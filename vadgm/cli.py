"""Command-line entry points for staged VaDGM development."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vadgm.base_model import (
    BaseGenerator,
    BaseModelConfig,
    evaluate_base_epoch,
    load_base_checkpoint,
    sample_base_graphs,
    save_base_checkpoint,
    train_base_epoch,
)
from vadgm.chemistry import AtomVocabulary
from vadgm.data import (
    DataConfig,
    MoleculeGraphDataset,
    SizeBucketBatchSampler,
    build_zinc_dataset,
    collate_graphs,
    fetch_zinc250k,
    load_manifest,
)
from vadgm.guidance_model import (
    GuidanceConfig,
    GuidanceNetwork,
    SourceCacheDataset,
    TargetSpecification,
    build_source_cache,
    cache_target_quantile,
    collate_source_cache,
    estimate_target_size_distribution,
    estimate_training_logp_scale,
    evaluate_guidance_epoch,
    load_guidance_checkpoint,
    load_source_cache,
    load_source_cache_metadata,
    model_state_sha256,
    property_weights,
    sample_guided_graphs,
    save_guidance_checkpoint,
    save_source_cache,
    train_guidance_epoch,
    training_size_distribution,
)
from vadgm.chemistry import CapacityEngine
from vadgm.diagnostics import EventPlan
from vadgm.sampler import result_to_dict, sample_vadgm_batch
from vadgm.evaluation import (
    evaluate_generation_file,
    export_evaluation_bundle,
)
from vadgm.paper_experiments import summarize_paper_experiments
from vadgm.utils import load_yaml, seed_everything, sha256_file
from vadgm.workflow import (
    artifact_paths,
    cache_config_hash,
    legacy_config_hashes,
    load_history,
    resume_config_hash,
    training_is_ready,
    write_history,
    write_json_atomic,
)


def _prepare_data(args: argparse.Namespace) -> int:
    if args.source.name.lower() == "zinc250k.csv" or not args.source.exists():
        fetch_zinc250k(args.source, show_progress=True)
    payload = load_yaml(args.config)
    config = DataConfig.from_mapping(payload)
    summary = build_zinc_dataset(
        args.source,
        args.output,
        config,
        show_progress=True,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


def _fetch_zinc250k(args: argparse.Namespace) -> int:
    summary = fetch_zinc250k(args.output, show_progress=True)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _training_precision(
    training: dict[str, object],
    device: torch.device,
) -> tuple[torch.dtype | None, torch.amp.GradScaler | None, str]:
    if device.type != "cuda" or not bool(training.get("amp", True)):
        return None, None, "float32"
    requested = str(training.get("amp_dtype", "bfloat16")).lower()
    if requested == "bfloat16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif requested in {"bfloat16", "float16"}:
        dtype = torch.float16
    else:
        raise ValueError("training.amp_dtype must be bfloat16 or float16")
    if bool(training.get("tf32", True)):
        torch.set_float32_matmul_precision("high")
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if dtype == torch.float16
        else None
    )
    return dtype, scaler, str(dtype).removeprefix("torch.")


def _size_bucket_loader(
    dataset: object,
    *,
    batch_size: int,
    collate_fn: object,
    training: dict[str, object],
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    sizes = getattr(dataset, "sizes", None)
    if not isinstance(sizes, tuple):
        raise TypeError("Size-bucket datasets must expose a sizes tuple")
    sampler = SizeBucketBatchSampler(
        sizes,
        batch_size,
        bucket_width=int(training.get("bucket_width", 2)),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )
    num_workers = int(training.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("training.num_workers cannot be negative")
    loader_kwargs: dict[str, object] = {
        "batch_sampler": sampler,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda" and bool(training.get("pin_memory", True)),
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    return DataLoader(dataset, **loader_kwargs)


def _dataset_is_ready(manifest: Path, vocabulary: Path) -> tuple[bool, str]:
    metadata_path = manifest.parent / "metadata.json"
    for path in (manifest, vocabulary, metadata_path):
        if not path.exists():
            return False, f"missing dataset artifact: {path}"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("manifest_sha256") != sha256_file(manifest):
        return False, "manifest hash does not match metadata.json"
    if metadata.get("vocabulary_sha256") != sha256_file(vocabulary):
        return False, "vocabulary hash does not match metadata.json"
    return True, "ready"


def _ensure_dataset(
    *,
    source: Path,
    manifest: Path,
    vocabulary: Path,
    data_config: Path,
) -> None:
    ready, reason = _dataset_is_ready(manifest, vocabulary)
    if ready:
        print(json.dumps({"stage": "data", "status": "reused", "reason": reason}))
        return
    expected_vocabulary = manifest.parent / "atom_vocabulary.json"
    if vocabulary.resolve() != expected_vocabulary.resolve():
        raise ValueError(
            "Automatic data preparation requires vocabulary next to manifest as "
            "atom_vocabulary.json"
        )
    existing = [path for path in (manifest, vocabulary, manifest.parent / "metadata.json") if path.exists()]
    if existing:
        raise ValueError(
            "Dataset directory is incomplete or inconsistent; use a new versioned "
            f"output directory instead of overwriting: {existing}"
        )
    if source.name.lower() == "zinc250k.csv":
        print(json.dumps({"stage": "zinc250k", "status": "checking source"}))
        fetch_zinc250k(source, show_progress=True)
    elif not source.exists():
        raise FileNotFoundError(f"Custom molecular source is missing: {source}")
    payload = load_yaml(data_config)
    summary = build_zinc_dataset(
        source,
        manifest.parent,
        DataConfig.from_mapping(payload),
        show_progress=True,
    )
    print(json.dumps({"stage": "data", "status": "built", **asdict(summary)}))


def _require_training_ready(
    best_checkpoint: Path,
    config_path: Path,
    *,
    artifact: str,
) -> None:
    ready, reason = training_is_ready(best_checkpoint, load_yaml(config_path))
    if not ready:
        raise RuntimeError(f"{artifact} is not ready for downstream use: {reason}")


def _resume_metadata_matches(
    observed: object,
    expected: dict[str, object],
    *,
    legacy_hashes: set[str],
) -> bool:
    if not isinstance(observed, dict):
        return False
    if "resume_config_hash" in observed:
        keys = (
            "manifest_sha256",
            "vocabulary_sha256",
            "resume_config_hash",
            "seed",
            "precision",
        )
    else:
        keys = ("manifest_sha256", "vocabulary_sha256", "seed", "precision")
        if observed.get("config_sha256") not in legacy_hashes:
            return False
    return all(observed.get(key) == expected.get(key) for key in keys)


def _resume_guidance_metadata_matches(
    observed: object,
    expected: dict[str, object],
    *,
    legacy_hashes: set[str],
) -> bool:
    if not isinstance(observed, dict):
        return False
    common = (
        "cache_sha256",
        "base_checkpoint_sha256",
        "manifest_sha256",
        "vocabulary_sha256",
        "seed",
    )
    keys = common
    if "resume_config_hash" in observed:
        keys = (*keys, "resume_config_hash", "precision")
    elif observed.get("config_sha256") not in legacy_hashes:
        return False
    return all(observed.get(key) == expected.get(key) for key in keys)


def _loader_generator(loader: DataLoader) -> torch.Generator | None:
    generator = getattr(loader.batch_sampler, "generator", None)
    return generator if isinstance(generator, torch.Generator) else None


def _training_runtime_metadata(
    train_generator: torch.Generator,
    train_loader: DataLoader,
    grad_scaler: torch.amp.GradScaler | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "training_generator_state": train_generator.get_state(),
    }
    loader_generator = _loader_generator(train_loader)
    if loader_generator is not None:
        metadata["loader_generator_state"] = loader_generator.get_state()
    if grad_scaler is not None:
        metadata["grad_scaler_state"] = grad_scaler.state_dict()
    return metadata


def _restore_training_runtime(
    metadata: object,
    train_generator: torch.Generator,
    train_loader: DataLoader,
    grad_scaler: torch.amp.GradScaler | None,
) -> None:
    if not isinstance(metadata, dict):
        return
    training_state = metadata.get("training_generator_state")
    if isinstance(training_state, torch.Tensor):
        train_generator.set_state(training_state.cpu())
    loader_state = metadata.get("loader_generator_state")
    loader_generator = _loader_generator(train_loader)
    if isinstance(loader_state, torch.Tensor) and loader_generator is not None:
        loader_generator.set_state(loader_state.cpu())
    scaler_state = metadata.get("grad_scaler_state")
    if isinstance(scaler_state, dict) and grad_scaler is not None:
        grad_scaler.load_state_dict(scaler_state)


def _train_base(args: argparse.Namespace) -> int:
    _ensure_dataset(
        source=args.source,
        manifest=args.manifest,
        vocabulary=args.vocabulary,
        data_config=args.data_config,
    )
    payload = load_yaml(args.config)
    model_payload = payload.get("model", {})
    training = payload.get("training", {})
    if not isinstance(model_payload, dict) or not isinstance(training, dict):
        raise ValueError("base config requires model and training mappings")
    seed = int(payload.get("seed", 2027))
    seed_everything(seed)
    device = _resolve_device(args.device)
    amp_dtype, grad_scaler, precision_name = _training_precision(training, device)
    vocabulary = AtomVocabulary.load(args.vocabulary)
    expected_base_metadata = {
        "manifest_sha256": sha256_file(args.manifest),
        "vocabulary_sha256": sha256_file(args.vocabulary),
        "config_sha256": sha256_file(args.config),
        "resume_config_hash": resume_config_hash(payload),
        "seed": seed,
        "precision": precision_name,
    }
    paths = artifact_paths(args.output)
    ready, ready_reason = training_is_ready(args.output, payload)
    if ready and args.resume is None:
        _, completed_payload = load_base_checkpoint(paths["best"], device="cpu")
        completed_epochs = int(completed_payload["epoch"]) + 1
        if _resume_metadata_matches(
            completed_payload.get("metadata", {}),
            expected_base_metadata,
            legacy_hashes=legacy_config_hashes(
                args.config,
                completed_epochs=completed_epochs,
            ),
        ):
            print(json.dumps({"status": "already_complete", "reason": ready_reason}))
            return 0
        raise ValueError(
            "Completion state exists, but the Base checkpoint lineage/configuration "
            "does not match the current data and YAML; use a new versioned output path"
        )
    start_epoch = 0
    best_loss = float("inf")
    resume_payload: dict[str, object] | None = None
    resume_path = args.resume
    if resume_path is None and paths["last"].exists():
        resume_path = paths["last"]
    elif resume_path is None and paths["best"].exists():
        resume_path = paths["best"]
    if resume_path:
        model, resume_payload = load_base_checkpoint(
            resume_path,
            device=device,
            restore_rng=True,
        )
        resume_metadata = resume_payload.get("metadata", {})
        checkpoint_epochs = int(resume_payload["epoch"]) + 1
        if not _resume_metadata_matches(
            resume_metadata,
            expected_base_metadata,
            legacy_hashes=legacy_config_hashes(
                args.config,
                completed_epochs=checkpoint_epochs,
            ),
        ):
            raise ValueError(
                "Resume checkpoint changes more than training duration/completion policy, "
                "or does not match manifest, vocabulary, seed, or precision"
            )
        start_epoch = checkpoint_epochs
        best_loss = float(resume_payload["best_validation_loss"])
    else:
        model = BaseGenerator(
            BaseModelConfig(
                atom_vocab_size=vocabulary.clean_size,
                hidden_dim=int(model_payload.get("hidden_dim", 256)),
                num_layers=int(model_payload.get("num_layers", 6)),
                dropout=float(model_payload.get("dropout", 0.1)),
                max_atoms=int(model_payload.get("max_atoms", 38)),
            )
        ).to(device)
    train_dataset = MoleculeGraphDataset.from_manifest(
        args.manifest,
        args.vocabulary,
        split="train",
        show_progress=True,
    )
    validation_dataset = MoleculeGraphDataset.from_manifest(
        args.manifest,
        args.vocabulary,
        split="validation",
        show_progress=True,
    )
    batch_size = int(training.get("batch_size", 128))
    train_loader = _size_bucket_loader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        training=training,
        device=device,
        shuffle=True,
        seed=seed + 101,
    )
    validation_loader = _size_bucket_loader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        training=training,
        device=device,
        shuffle=False,
        seed=seed + 102,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    if resume_payload is not None and resume_payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state"])

    train_generator = torch.Generator(device=device.type).manual_seed(seed + 1)
    if resume_payload is not None:
        _restore_training_runtime(
            resume_payload.get("metadata", {}),
            train_generator,
            train_loader,
            grad_scaler,
        )
    patience = int(training.get("patience", 30))
    epochs = int(training.get("epochs", 300))
    validation_interval = int(training.get("validation_interval", 1))
    if epochs <= 0 or patience <= 0 or validation_interval <= 0:
        raise ValueError("epochs, patience, and validation_interval must be positive")
    if start_epoch > epochs:
        raise ValueError(
            f"Configured epochs ({epochs}) cannot be lower than the checkpoint's "
            f"completed epochs ({start_epoch})"
        )
    edge_weight = float(training.get("edge_loss_weight", 1.0))
    epochs_without_improvement = int(
        resume_payload.get("metadata", {}).get("stale_validations", 0)
        if resume_payload is not None
        else 0
    )
    history: list[dict[str, object]] = load_history(
        paths["history"],
        before_epoch=start_epoch,
    )
    completed_epochs = start_epoch
    stopped_early = False
    epoch_iterator = tqdm(
        range(start_epoch, epochs),
        desc="Base training",
        unit="epoch",
    )
    for epoch in epoch_iterator:
        train_metrics = train_base_epoch(
            model,
            train_loader,
            optimizer,
            vocabulary,
            device=device,
            edge_loss_weight=edge_weight,
            generator=train_generator,
            progress_desc=f"Base epoch {epoch + 1} train",
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
        )
        should_validate = (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        validation_metrics = (
            evaluate_base_epoch(
                model,
                validation_loader,
                vocabulary,
                device=device,
                edge_loss_weight=edge_weight,
                generator=torch.Generator(device=device.type).manual_seed(seed + 2),
                progress_desc=f"Base epoch {epoch + 1} validation",
                amp_dtype=amp_dtype,
            )
            if should_validate
            else None
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": (
                validation_metrics["loss"] if validation_metrics is not None else None
            ),
        }
        history.append(row)
        write_history(paths["history"], history)
        print(json.dumps(row), flush=True)
        runtime_metadata = _training_runtime_metadata(
            train_generator,
            train_loader,
            grad_scaler,
        )
        should_stop = False
        if validation_metrics is not None and validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            save_base_checkpoint(
                args.output,
                model,
                optimizer,
                epoch=epoch,
                best_validation_loss=best_loss,
                metadata={**expected_base_metadata, **runtime_metadata},
            )
        elif validation_metrics is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_early = True
                should_stop = True
        save_base_checkpoint(
            paths["last"],
            model,
            optimizer,
            epoch=epoch,
            best_validation_loss=best_loss,
            metadata={
                **expected_base_metadata,
                **runtime_metadata,
                "stale_validations": epochs_without_improvement,
            },
        )
        completed_epochs = epoch + 1
        write_json_atomic(
            paths["state"],
            {
                "artifact": "base",
                "status": "running",
                "completed_epochs": completed_epochs,
                "target_epochs": epochs,
                "ready_for_downstream": False,
                "best_validation_loss": best_loss,
                "resume_config_hash": expected_base_metadata["resume_config_hash"],
                "best_checkpoint": str(paths["best"]),
                "last_checkpoint": str(paths["last"]),
                "history": str(paths["history"]),
            },
        )
        postfix = {"train": f"{train_metrics['loss']:.4f}"}
        if validation_metrics is not None:
            postfix["val"] = f"{validation_metrics['loss']:.4f}"
            postfix["best"] = f"{best_loss:.4f}"
        epoch_iterator.set_postfix(postfix)
        if should_stop:
            break
    if resume_payload is not None and start_epoch == epochs and not paths["last"].exists():
        save_base_checkpoint(
            paths["last"],
            model,
            optimizer,
            epoch=start_epoch - 1,
            best_validation_loss=best_loss,
            metadata={
                **expected_base_metadata,
                **_training_runtime_metadata(
                    train_generator,
                    train_loader,
                    grad_scaler,
                ),
                "stale_validations": epochs_without_improvement,
            },
        )
    ready_for_downstream = paths["best"].exists() and (
        stopped_early or completed_epochs >= epochs
    )
    write_json_atomic(
        paths["state"],
        {
            "artifact": "base",
            "status": "complete" if ready_for_downstream else "incomplete",
            "completed_epochs": completed_epochs,
            "target_epochs": epochs,
            "stopped_early": stopped_early,
            "completion_reason": (
                "early_stopping" if stopped_early else "target_epochs"
            ),
            "ready_for_downstream": ready_for_downstream,
            "best_validation_loss": best_loss,
            "resume_config_hash": expected_base_metadata["resume_config_hash"],
            "best_checkpoint": str(paths["best"]),
            "last_checkpoint": str(paths["last"]),
            "history": str(paths["history"]),
        },
    )
    print(
        json.dumps(
            {
                "best_validation_loss": best_loss,
                "completed_epochs": completed_epochs,
                "ready_for_downstream": ready_for_downstream,
            }
        )
    )
    return 0


def _sample_base(args: argparse.Namespace) -> int:
    _require_training_ready(
        args.checkpoint,
        args.base_config,
        artifact="Base generator",
    )
    device = _resolve_device(args.device)
    vocabulary = AtomVocabulary.load(args.vocabulary)
    model, payload = load_base_checkpoint(args.checkpoint, device=device)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get(
        "vocabulary_sha256"
    ) != sha256_file(args.vocabulary):
        raise ValueError("Base checkpoint does not use the supplied atom vocabulary")
    seed = int(args.seed)
    generator = torch.Generator(device=device.type).manual_seed(seed)
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    graphs, diagnostics = sample_base_graphs(
        model,
        vocabulary,
        sizes,
        steps=args.steps,
        generator=generator,
        device=device,
        progress_desc="Base sampling",
    )
    result = {
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "vocabulary_sha256": sha256_file(args.vocabulary),
        "seed": seed,
        "diagnostics": asdict(diagnostics),
        "graphs": [
            {
                "sample_id": f"base-{index:06d}",
                "atom_ids": list(graph.atom_ids),
                "bond_ids": list(graph.bond_ids),
            }
            for index, graph in enumerate(graphs)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _build_cache(args: argparse.Namespace) -> int:
    base_payload = load_yaml(args.base_config)
    base_ready, reason = training_is_ready(args.checkpoint, base_payload)
    if not base_ready:
        print(
            json.dumps(
                {
                    "stage": "base",
                    "status": "incomplete; training automatically",
                    "reason": reason,
                }
            )
        )
        _train_base(
            argparse.Namespace(
                manifest=args.manifest,
                vocabulary=args.vocabulary,
                output=args.checkpoint,
                config=args.base_config,
                resume=None,
                device=args.device,
                source=args.source,
                data_config=args.data_config,
            )
        )
    _require_training_ready(
        args.checkpoint,
        args.base_config,
        artifact="Base generator",
    )
    payload = load_yaml(args.config)
    cache_config = payload.get("cache", {})
    if not isinstance(cache_config, dict):
        raise ValueError("guidance config cache must be a mapping")
    seed = int(payload.get("seed", 2027))
    cache_metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if args.output.exists() and cache_metadata_path.exists():
        existing_metadata = load_source_cache_metadata(args.output)
        common_expected = {
            "base_checkpoint_sha256": sha256_file(args.checkpoint),
            "manifest_sha256": sha256_file(args.manifest),
            "vocabulary_sha256": sha256_file(args.vocabulary),
            "samples": int(cache_config.get("samples", 100_000)),
        }
        current_cache_hash = cache_config_hash(payload)
        exact_cache_contract = existing_metadata.get("cache_config_hash") == current_cache_hash
        legacy_cache_contract = all(
            existing_metadata.get(key) == value
            for key, value in {
                "seed": seed,
                "samples": int(cache_config.get("samples", 100_000)),
                "batch_size": int(cache_config.get("batch_size", 256)),
                "sampling_steps": int(cache_config.get("sampling_steps", 256)),
            }.items()
        )
        if all(
            existing_metadata.get(key) == value
            for key, value in common_expected.items()
        ) and (exact_cache_contract or legacy_cache_contract):
            print(json.dumps({"stage": "source_cache", "status": "reused"}))
            return 0
        print(
            json.dumps(
                {
                    "stage": "source_cache",
                    "status": "stale; rebuilding atomically",
                    "reason": "base or cache-generation contract changed",
                }
            )
        )
    if args.output.exists() != cache_metadata_path.exists():
        raise ValueError("Source cache data/metadata is incomplete; use a new output path")
    device = _resolve_device(args.device)
    vocabulary = AtomVocabulary.load(args.vocabulary)
    base, checkpoint = load_base_checkpoint(args.checkpoint, device=device)
    train_records = load_manifest(args.manifest, split="train")
    size_distribution = training_size_distribution(
        record.n_heavy_atoms for record in train_records
    )
    generator = torch.Generator(device=device.type).manual_seed(seed)
    records, metadata = build_source_cache(
        base,
        vocabulary,
        CapacityEngine(vocabulary),
        size_distribution,
        sample_count=int(cache_config.get("samples", 100_000)),
        batch_size=int(cache_config.get("batch_size", 256)),
        sampling_steps=int(cache_config.get("sampling_steps", 256)),
        generator=generator,
        device=device,
        show_progress=True,
    )
    metadata.update(
        {
            "base_checkpoint_sha256": sha256_file(args.checkpoint),
            "base_checkpoint_epoch": checkpoint["epoch"],
            "manifest_sha256": sha256_file(args.manifest),
            "vocabulary_sha256": sha256_file(args.vocabulary),
            "config_sha256": sha256_file(args.config),
            "cache_config_hash": cache_config_hash(payload),
            "batch_size": int(cache_config.get("batch_size", 256)),
            "seed": seed,
        }
    )
    save_source_cache(records, args.output, metadata, show_progress=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def _train_guidance(args: argparse.Namespace) -> int:
    _build_cache(
        argparse.Namespace(
            checkpoint=args.base_checkpoint,
            manifest=args.manifest,
            vocabulary=args.vocabulary,
            output=args.cache,
            config=args.config,
            base_config=args.base_config,
            source=args.source,
            data_config=args.data_config,
            device=args.device,
        )
    )
    payload = load_yaml(args.config)
    model_config = payload.get("model", {})
    training = payload.get("training", {})
    if not isinstance(model_config, dict) or not isinstance(training, dict):
        raise ValueError("guidance config requires model and training mappings")
    seed = int(payload.get("seed", 2027))
    seed_everything(seed)
    device = _resolve_device(args.device)
    amp_dtype, grad_scaler, precision_name = _training_precision(training, device)
    paths = artifact_paths(args.output)
    ready, ready_reason = training_is_ready(args.output, payload)
    cache_metadata = load_source_cache_metadata(args.cache)
    expected_base_hash = sha256_file(args.base_checkpoint)
    expected_manifest_hash = sha256_file(args.manifest)
    expected_vocabulary_hash = sha256_file(args.vocabulary)
    expected_config_hash = sha256_file(args.config)
    expected_resume_config_hash = resume_config_hash(payload)
    expected_cache_hash = sha256_file(args.cache)
    if cache_metadata.get("base_checkpoint_sha256") != expected_base_hash:
        raise ValueError("Source cache was not generated by the supplied base checkpoint")
    if cache_metadata.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("Source cache was not generated from the supplied manifest")
    if cache_metadata.get("vocabulary_sha256") != expected_vocabulary_hash:
        raise ValueError("Source cache does not use the supplied atom vocabulary")
    lineage_keys = {
        "cache_sha256": expected_cache_hash,
        "base_checkpoint_sha256": expected_base_hash,
        "manifest_sha256": expected_manifest_hash,
        "vocabulary_sha256": expected_vocabulary_hash,
    }
    existing_lineage_matches = False
    if paths["best"].exists():
        _, _, _, existing_payload = load_guidance_checkpoint(
            paths["best"],
            device="cpu",
        )
        existing_metadata = existing_payload.get("metadata", {})
        existing_lineage_matches = isinstance(existing_metadata, dict) and all(
            existing_metadata.get(key) == value for key, value in lineage_keys.items()
        )
    if ready and args.resume is None and existing_lineage_matches:
        print(json.dumps({"status": "already_complete", "reason": ready_reason}))
        return 0
    if paths["best"].exists() and not existing_lineage_matches:
        print(
            json.dumps(
                {
                    "stage": "guidance",
                    "status": "stale; restarting from scratch",
                    "reason": "base or source-cache lineage changed",
                }
            )
        )
    vocabulary = AtomVocabulary.load(args.vocabulary)
    cache_records = load_source_cache(
        args.cache,
        show_progress=True,
        total=int(cache_metadata["samples"]),
    )
    train_manifest = load_manifest(args.manifest, split="train")
    sigma = estimate_training_logp_scale(
        (record.smiles for record in train_manifest),
        show_progress=True,
        total=len(train_manifest),
    )
    target_value = cache_target_quantile(
        cache_records,
        float(payload.get("target_quantile", 0.8)),
    )
    target = TargetSpecification(
        property_name=str(payload.get("property", "logP")),
        target=target_value,
        gamma=float(payload.get("gamma", 4.0)),
        sigma=sigma,
    )
    weights = property_weights(cache_records, target)
    size_distribution = estimate_target_size_distribution(cache_records, weights)
    if len(cache_records) < 2:
        raise ValueError("Guidance training requires at least two cache records")
    permutation = torch.randperm(
        len(cache_records),
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    validation_count = max(1, int(0.1 * len(cache_records)))
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]

    def select(indices: list[int]) -> tuple[list[object], torch.Tensor]:
        return [cache_records[index] for index in indices], weights[indices]

    train_cache, train_weights = select(train_indices)
    validation_cache, validation_weights = select(validation_indices)
    batch_size = int(training.get("batch_size", 128))
    train_dataset = SourceCacheDataset(
        train_cache,
        train_weights,
        vocabulary,
        target.target,
        show_progress=True,
    )
    validation_dataset = SourceCacheDataset(
        validation_cache,
        validation_weights,
        vocabulary,
        target.target,
        show_progress=True,
    )
    train_loader = _size_bucket_loader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_source_cache,
        training=training,
        device=device,
        shuffle=True,
        seed=seed + 201,
    )
    validation_loader = _size_bucket_loader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=collate_source_cache,
        training=training,
        device=device,
        shuffle=False,
        seed=seed + 202,
    )
    start_epoch = 0
    resume_payload: dict[str, object] | None = None
    resume_path = args.resume
    if resume_path is None and existing_lineage_matches and paths["last"].exists():
        resume_path = paths["last"]
    elif resume_path is None and existing_lineage_matches and paths["best"].exists():
        resume_path = paths["best"]
    if resume_path:
        guidance, restored_target, restored_size, resume_payload = load_guidance_checkpoint(
            resume_path,
            device=device,
            restore_rng=True,
        )
        if restored_target != target or restored_size != size_distribution:
            raise ValueError("Resume checkpoint target or size distribution does not match inputs")
        resume_metadata = resume_payload.get("metadata", {})
        expected_resume_metadata = {
            "cache_sha256": expected_cache_hash,
            "base_checkpoint_sha256": expected_base_hash,
            "manifest_sha256": expected_manifest_hash,
            "vocabulary_sha256": expected_vocabulary_hash,
            "config_sha256": expected_config_hash,
            "resume_config_hash": expected_resume_config_hash,
            "seed": seed,
            "precision": precision_name,
        }
        if not _resume_guidance_metadata_matches(
            resume_metadata,
            expected_resume_metadata,
            legacy_hashes=legacy_config_hashes(
                args.config,
                completed_epochs=int(resume_payload["epoch"]) + 1,
            ),
        ):
            raise ValueError(
                "Resume guidance checkpoint does not match cache, base, manifest, "
                "vocabulary, config, or seed"
            )
        start_epoch = int(resume_payload["epoch"]) + 1
    else:
        guidance = GuidanceNetwork(
            GuidanceConfig(
                atom_vocab_size=vocabulary.clean_size,
                hidden_dim=int(model_config.get("hidden_dim", 256)),
                num_layers=int(model_config.get("num_layers", 6)),
                dropout=float(model_config.get("dropout", 0.1)),
                max_atoms=int(model_config.get("max_atoms", 38)),
                u_min=float(payload.get("u_min", -12.0)),
                u_max=float(payload.get("u_max", 8.0)),
            )
        ).to(device)
    optimizer = torch.optim.AdamW(
        guidance.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    if resume_payload is not None and resume_payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
    train_generator = torch.Generator(device=device.type).manual_seed(seed + 1)
    if resume_payload is not None:
        _restore_training_runtime(
            resume_payload.get("metadata", {}),
            train_generator,
            train_loader,
            grad_scaler,
        )
    epochs = int(training.get("epochs", 200))
    patience = int(training.get("patience", 30))
    validation_interval = int(training.get("validation_interval", 1))
    if epochs <= 0 or patience <= 0 or validation_interval <= 0:
        raise ValueError("epochs, patience, and validation_interval must be positive")
    if start_epoch > epochs:
        raise ValueError(
            f"Configured epochs ({epochs}) cannot be lower than the checkpoint's "
            f"completed epochs ({start_epoch})"
        )
    edge_weight = float(training.get("edge_loss_weight", 1.0))
    best_loss = (
        float(resume_payload.get("metadata", {}).get("best_validation_loss", math.inf))
        if resume_payload is not None
        else float("inf")
    )
    stale_epochs = int(
        resume_payload.get("metadata", {}).get("stale_validations", 0)
        if resume_payload is not None
        else 0
    )
    history: list[dict[str, object]] = load_history(
        paths["history"],
        before_epoch=start_epoch,
    )
    completed_epochs = start_epoch
    stopped_early = False
    epoch_iterator = tqdm(
        range(start_epoch, epochs),
        desc="Guidance training",
        unit="epoch",
    )
    for epoch in epoch_iterator:
        train_metrics = train_guidance_epoch(
            guidance,
            train_loader,
            optimizer,
            vocabulary,
            device=device,
            edge_loss_weight=edge_weight,
            generator=train_generator,
            progress_desc=f"Guidance epoch {epoch + 1} train",
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
        )
        should_validate = (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        validation_metrics = (
            evaluate_guidance_epoch(
                guidance,
                validation_loader,
                vocabulary,
                device=device,
                edge_loss_weight=edge_weight,
                generator=torch.Generator(device=device.type).manual_seed(seed + 2),
                progress_desc=f"Guidance epoch {epoch + 1} validation",
                amp_dtype=amp_dtype,
            )
            if should_validate
            else None
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": (
                validation_metrics["loss"] if validation_metrics is not None else None
            ),
        }
        history.append(row)
        write_history(paths["history"], history)
        print(json.dumps(row), flush=True)
        runtime_metadata = _training_runtime_metadata(
            train_generator,
            train_loader,
            grad_scaler,
        )
        should_stop = False
        if validation_metrics is not None and validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            stale_epochs = 0
            save_guidance_checkpoint(
                args.output,
                guidance,
                optimizer,
                epoch=epoch,
                target_specification=target,
                size_distribution=size_distribution,
                metadata={
                    "cache_sha256": expected_cache_hash,
                    "base_checkpoint_sha256": expected_base_hash,
                    "manifest_sha256": expected_manifest_hash,
                    "vocabulary_sha256": expected_vocabulary_hash,
                    "config_sha256": expected_config_hash,
                    "resume_config_hash": expected_resume_config_hash,
                    "best_validation_loss": best_loss,
                    "seed": seed,
                    "precision": precision_name,
                    **runtime_metadata,
                },
            )
        elif validation_metrics is not None:
            stale_epochs += 1
            if stale_epochs >= patience:
                stopped_early = True
                should_stop = True
        save_guidance_checkpoint(
            paths["last"],
            guidance,
            optimizer,
            epoch=epoch,
            target_specification=target,
            size_distribution=size_distribution,
            metadata={
                "cache_sha256": expected_cache_hash,
                "base_checkpoint_sha256": expected_base_hash,
                "manifest_sha256": expected_manifest_hash,
                "vocabulary_sha256": expected_vocabulary_hash,
                "config_sha256": expected_config_hash,
                "resume_config_hash": expected_resume_config_hash,
                "best_validation_loss": best_loss,
                "seed": seed,
                "precision": precision_name,
                **runtime_metadata,
                "stale_validations": stale_epochs,
            },
        )
        completed_epochs = epoch + 1
        write_json_atomic(
            paths["state"],
            {
                "artifact": "guidance",
                "status": "running",
                "completed_epochs": completed_epochs,
                "target_epochs": epochs,
                "ready_for_downstream": False,
                "best_validation_loss": best_loss,
                "resume_config_hash": expected_resume_config_hash,
                "best_checkpoint": str(paths["best"]),
                "last_checkpoint": str(paths["last"]),
                "history": str(paths["history"]),
            },
        )
        postfix = {"train": f"{train_metrics['loss']:.4f}"}
        if validation_metrics is not None:
            postfix["val"] = f"{validation_metrics['loss']:.4f}"
            postfix["best"] = f"{best_loss:.4f}"
        epoch_iterator.set_postfix(postfix)
        if should_stop:
            break
    if resume_payload is not None and start_epoch == epochs and not paths["last"].exists():
        save_guidance_checkpoint(
            paths["last"],
            guidance,
            optimizer,
            epoch=start_epoch - 1,
            target_specification=target,
            size_distribution=size_distribution,
            metadata={
                "cache_sha256": expected_cache_hash,
                "base_checkpoint_sha256": expected_base_hash,
                "manifest_sha256": expected_manifest_hash,
                "vocabulary_sha256": expected_vocabulary_hash,
                "config_sha256": expected_config_hash,
                "resume_config_hash": expected_resume_config_hash,
                "best_validation_loss": best_loss,
                "seed": seed,
                "precision": precision_name,
                **_training_runtime_metadata(
                    train_generator,
                    train_loader,
                    grad_scaler,
                ),
                "stale_validations": stale_epochs,
            },
        )
    ready_for_downstream = paths["best"].exists() and (
        stopped_early or completed_epochs >= epochs
    )
    write_json_atomic(
        paths["state"],
        {
            "artifact": "guidance",
            "status": "complete" if ready_for_downstream else "incomplete",
            "completed_epochs": completed_epochs,
            "target_epochs": epochs,
            "stopped_early": stopped_early,
            "completion_reason": (
                "early_stopping" if stopped_early else "target_epochs"
            ),
            "ready_for_downstream": ready_for_downstream,
            "best_validation_loss": best_loss,
            "resume_config_hash": expected_resume_config_hash,
            "best_checkpoint": str(paths["best"]),
            "last_checkpoint": str(paths["last"]),
            "history": str(paths["history"]),
        },
    )
    print(
        json.dumps(
            {
                "target": asdict(target),
                "size_distribution": asdict(size_distribution),
                "best_validation_loss": best_loss,
                "completed_epochs": completed_epochs,
                "ready_for_downstream": ready_for_downstream,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _sample_guided(args: argparse.Namespace) -> int:
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    _require_training_ready(
        args.base_checkpoint,
        args.base_config,
        artifact="Base generator",
    )
    _require_training_ready(
        args.guidance_checkpoint,
        args.guidance_config,
        artifact="Guidance network",
    )
    device = _resolve_device(args.device)
    vocabulary = AtomVocabulary.load(args.vocabulary)
    base, _ = load_base_checkpoint(args.base_checkpoint, device=device)
    guidance, target, size_distribution, guidance_payload = load_guidance_checkpoint(
        args.guidance_checkpoint,
        device=device,
    )
    guidance_metadata = guidance_payload.get("metadata", {})
    if guidance_metadata.get("base_checkpoint_sha256") != sha256_file(args.base_checkpoint):
        raise ValueError("Guidance checkpoint was not trained for the supplied base checkpoint")
    if guidance_metadata.get("vocabulary_sha256") != sha256_file(args.vocabulary):
        raise ValueError("Guidance checkpoint does not use the supplied atom vocabulary")
    size_generator = torch.Generator().manual_seed(args.seed)
    sizes = size_distribution.sample(args.samples, generator=size_generator)
    sample_generator = torch.Generator(device=device.type).manual_seed(args.seed + 1)
    base_state_before = model_state_sha256(base)
    guidance_state_before = model_state_sha256(guidance)
    graphs = []
    for start in tqdm(
        range(0, len(sizes), args.batch_size),
        desc="Guided sampling",
        unit="batch",
    ):
        batch_graphs, diagnostics = sample_guided_graphs(
            base,
            guidance,
            vocabulary,
            sizes[start : start + args.batch_size],
            target=target.target,
            steps=args.steps,
            generator=sample_generator,
            device=device,
            verify_model_state=False,
        )
        graphs.extend(batch_graphs)
    if base_state_before != model_state_sha256(base):
        raise RuntimeError("Frozen base parameters changed during guided sampling")
    if guidance_state_before != model_state_sha256(guidance):
        raise RuntimeError("Frozen guidance parameters changed during guided sampling")
    result = {
        "experiment": args.experiment,
        "condition": args.condition,
        "method": "Vanilla DGM",
        "target": asdict(target),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "guidance_checkpoint_sha256": sha256_file(args.guidance_checkpoint),
        "vocabulary_sha256": sha256_file(args.vocabulary),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "diagnostics": asdict(diagnostics),
        "graphs": [
            {
                "sample_id": f"guided-{index:06d}",
                "n_atoms": graph.n_atoms,
                "atom_ids": list(graph.atom_ids),
                "bond_ids": list(graph.bond_ids),
            }
            for index, graph in enumerate(graphs)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _sample_vadgm(args: argparse.Namespace) -> int:
    payload = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else payload.get("seed", 2027))
    delta_s = float(
        args.delta_s if args.delta_s is not None else payload.get("delta_s", 0.25)
    )
    record_trace = bool(payload.get("record_event_trace", True))
    record_event_plan = bool(payload.get("record_event_plan", False))
    record_conflict = bool(payload.get("record_joint_conflict", True))
    terminal_payload = payload.get("terminal_reachability", {})
    if not isinstance(terminal_payload, dict):
        raise ValueError("terminal_reachability must be a YAML mapping")
    enforce_terminal_reachability = bool(terminal_payload.get("enabled", True))
    relax_terminal_gate = bool(terminal_payload.get("relax_on_empty", True))
    defer_atom_commit = bool(terminal_payload.get("defer_atom_commit", True))
    if defer_atom_commit and not enforce_terminal_reachability:
        raise ValueError("defer_atom_commit requires terminal_reachability.enabled=true")
    if not enforce_terminal_reachability:
        method = "VaDGM (capacity-only)"
    elif not defer_atom_commit:
        method = "VaDGM (no terminal atom phase)"
    elif not relax_terminal_gate:
        method = "VaDGM (strict terminal gate)"
    else:
        method = "VaDGM"
    device = _resolve_device(args.device or payload.get("device"))
    _require_training_ready(
        args.base_checkpoint,
        args.base_config,
        artifact="Base generator",
    )
    _require_training_ready(
        args.guidance_checkpoint,
        args.guidance_config,
        artifact="Guidance network",
    )
    vocabulary = AtomVocabulary.load(args.vocabulary)
    capacity = CapacityEngine(vocabulary)
    base, _ = load_base_checkpoint(args.base_checkpoint, device=device)
    guidance, target, size_distribution, guidance_payload = load_guidance_checkpoint(
        args.guidance_checkpoint,
        device=device,
    )
    guidance_metadata = guidance_payload.get("metadata", {})
    if guidance_metadata.get("base_checkpoint_sha256") != sha256_file(args.base_checkpoint):
        raise ValueError("Guidance checkpoint was not trained for the supplied base checkpoint")
    if guidance_metadata.get("vocabulary_sha256") != sha256_file(args.vocabulary):
        raise ValueError("Guidance checkpoint does not use the supplied atom vocabulary")
    sample_count = int(args.samples or payload.get("samples", 10_000))
    batch_size = int(args.batch_size or payload.get("batch_size", 128))
    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    size_generator = torch.Generator().manual_seed(seed)
    plan_generator = torch.Generator().manual_seed(seed + 1)
    category_generator = torch.Generator().manual_seed(seed + 2)
    diagnostic_generator = (
        torch.Generator().manual_seed(seed + 3)
        if record_conflict
        else None
    )
    sizes = size_distribution.sample(sample_count, generator=size_generator)
    output_metadata = {
        "experiment": args.experiment,
        "condition": args.condition,
        "method": method,
        "target": asdict(target),
        "delta_s": delta_s,
        "seed": seed,
        "samples_requested": sample_count,
        "batch_size": batch_size,
        "record_event_plan": record_event_plan,
        "record_event_trace": record_trace,
        "record_joint_conflict": record_conflict,
        "terminal_reachability": {
            "enabled": enforce_terminal_reachability,
            "relax_on_empty": relax_terminal_gate,
            "defer_atom_commit": defer_atom_commit,
        },
        "config_sha256": sha256_file(args.config),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "guidance_checkpoint_sha256": sha256_file(args.guidance_checkpoint),
        "vocabulary_sha256": sha256_file(args.vocabulary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".part")
    base_state_before = model_state_sha256(base)
    guidance_state_before = model_state_sha256(guidance)
    neural_batch_forwards = 0
    try:
        with temporary_output.open("w", encoding="utf-8") as handle:
            handle.write("{\n")
            for key, value in output_metadata.items():
                handle.write(f"  {json.dumps(key)}: ")
                json.dump(value, handle, ensure_ascii=False)
                handle.write(",\n")
            handle.write('  "samples": [\n')
            with tqdm(total=sample_count, desc="VaDGM sampling", unit="mol") as progress:
                for start in range(0, sample_count, batch_size):
                    batch_sizes = sizes[start : start + batch_size]
                    plans = [
                        EventPlan.sample(size, delta_s, generator=plan_generator)
                        for size in batch_sizes
                    ]
                    if defer_atom_commit:
                        plans = [plan.with_terminal_atom_phase() for plan in plans]
                    results, batch_diagnostics = sample_vadgm_batch(
                        base,
                        guidance,
                        vocabulary,
                        capacity,
                        plans,
                        target=target.target,
                        category_generator=category_generator,
                        diagnostic_generator=diagnostic_generator,
                        device=device,
                        record_trace=record_trace,
                        verify_model_state=False,
                        enforce_terminal_reachability=enforce_terminal_reachability,
                        relax_terminal_gate=relax_terminal_gate,
                    )
                    neural_batch_forwards += batch_diagnostics.neural_batch_forwards
                    for offset, result in enumerate(results):
                        index = start + offset
                        result_payload = result_to_dict(
                            result,
                            include_event_plan=record_event_plan,
                            include_event_trace=record_trace,
                        )
                        result_payload["sample_id"] = f"vadgm-{index:06d}"
                        if index:
                            handle.write(",\n")
                        handle.write("    ")
                        json.dump(
                            result_payload,
                            handle,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    progress.update(len(results))
                    progress.set_postfix(batch=len(results))
            handle.write(
                "\n  ],\n"
                f'  "execution": {{"neural_batch_forwards":{neural_batch_forwards}}}\n'
                "}\n"
            )
        if base_state_before != model_state_sha256(base):
            raise RuntimeError("Frozen base parameters changed during VaDGM sampling")
        if guidance_state_before != model_state_sha256(guidance):
            raise RuntimeError("Frozen guidance parameters changed during VaDGM sampling")
        os.replace(temporary_output, args.output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()
    return 0


def _evaluate_results(args: argparse.Namespace) -> int:
    vocabulary = AtomVocabulary.load(args.vocabulary)
    records, metrics = evaluate_generation_file(
        args.input,
        vocabulary,
        method=args.method,
        target_override=args.target,
        hit_tolerance=args.hit_tolerance,
        show_progress=True,
    )
    outputs = export_evaluation_bundle(
        records,
        metrics,
        args.output_dir,
        render_examples=not args.no_render,
    )
    print(
        json.dumps(
            {"metrics": asdict(metrics), "outputs": outputs},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _summarize_paper(args: argparse.Namespace) -> int:
    outputs = summarize_paper_experiments(
        args.input_dir,
        args.vocabulary,
        args.training_manifest,
        args.output_dir,
        hit_tolerance=args.hit_tolerance,
    )
    print(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2))
    return 0


def _run_table1(args: argparse.Namespace) -> int:
    """Run the three Table 1 sampling conditions for configured seeds."""

    payload = load_yaml(args.config)
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Table 1 config paths must be a YAML mapping")
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Table 1 config settings must be a YAML mapping")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(int(seed) < 0 for seed in seeds):
        raise ValueError("Table 1 config seeds must be a non-empty list")
    output_dir = Path(paths.get("output_dir", "results/runs/paper/table1"))
    experiment = str(payload.get("experiment", "table1"))
    condition = str(payload.get("condition", "q80"))
    samples = int(settings.get("samples", 10_000))
    steps = int(settings.get("steps", 128))
    batch_size = int(settings.get("batch_size", 256))
    device = settings.get("device")
    if samples <= 0 or steps <= 0 or batch_size <= 0:
        raise ValueError("Table 1 samples, steps, and batch_size must be positive")

    common = {
        "base_checkpoint": Path(paths["base_checkpoint"]),
        "guidance_checkpoint": Path(paths["guidance_checkpoint"]),
        "vocabulary": Path(paths["vocabulary"]),
        "base_config": Path(paths.get("base_config", "configs/base.yaml")),
        "guidance_config": Path(paths.get("guidance_config", "configs/guidance.yaml")),
        "device": device,
    }
    sampler_configs = payload.get("sampler_configs", {})
    if not isinstance(sampler_configs, dict):
        raise ValueError("Table 1 sampler_configs must be a YAML mapping")
    capacity_config = Path(
        sampler_configs.get("capacity_only", "configs/sampling_capacity_only.yaml")
    )
    vadgm_config = Path(sampler_configs.get("vadgm", "configs/sampling.yaml"))
    overwrite = bool(settings.get("overwrite", False))

    def reusable(path: Path, seed: int) -> bool:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(existing, dict):
            return False
        if (
            existing.get("experiment") != experiment
            or existing.get("condition") != condition
            or int(existing.get("seed", -1)) != seed
        ):
            return False
        collection = existing.get("graphs", existing.get("samples"))
        return isinstance(collection, list) and len(collection) == samples

    for seed_value in seeds:
        seed = int(seed_value)
        jobs = (
            (
                "vanilla",
                output_dir / f"vanilla_q80_seed{seed}.json",
                _sample_guided,
                argparse.Namespace(
                    **common,
                    samples=samples,
                    steps=steps,
                    batch_size=batch_size,
                    seed=seed,
                    experiment=experiment,
                    condition=condition,
                    output=output_dir / f"vanilla_q80_seed{seed}.json",
                ),
            ),
            (
                "capacity-only",
                output_dir / f"capacity_q80_seed{seed}.json",
                _sample_vadgm,
                argparse.Namespace(
                    **common,
                    samples=samples,
                    batch_size=batch_size,
                    seed=seed,
                    delta_s=None,
                    experiment=experiment,
                    condition=condition,
                    output=output_dir / f"capacity_q80_seed{seed}.json",
                    config=capacity_config,
                ),
            ),
            (
                "VaDGM",
                output_dir / f"vadgm_q80_seed{seed}.json",
                _sample_vadgm,
                argparse.Namespace(
                    **common,
                    samples=samples,
                    batch_size=batch_size,
                    seed=seed,
                    delta_s=None,
                    experiment=experiment,
                    condition=condition,
                    output=output_dir / f"vadgm_q80_seed{seed}.json",
                    config=vadgm_config,
                ),
            ),
        )
        for label, output, handler, job_args in jobs:
            if output.exists() and not overwrite:
                if reusable(output, seed):
                    print(json.dumps({"condition": label, "seed": seed, "status": "reused", "output": str(output)}))
                    continue
                raise ValueError(
                    f"Existing Table 1 output is incomplete or uses a different sample count: {output}. "
                    "Set overwrite: true or move the stale file before sampling."
                )
            print(json.dumps({"condition": label, "seed": seed, "status": "sampling", "output": str(output)}), flush=True)
            handler(job_args)
    return 0


def _experiment_payload(
    config_path: Path,
    *,
    label: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], list[int]]:
    """Load the shared compact schema used by the remaining paper experiments."""

    payload = load_yaml(config_path)
    paths = payload.get("paths", {})
    settings = payload.get("settings", {})
    seeds = payload.get("seeds")
    if not isinstance(paths, dict):
        raise ValueError(f"{label} config paths must be a YAML mapping")
    if not isinstance(settings, dict):
        raise ValueError(f"{label} config settings must be a YAML mapping")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{label} config seeds must be a non-empty list")
    parsed_seeds = [int(seed) for seed in seeds]
    if any(seed < 0 for seed in parsed_seeds) or len(set(parsed_seeds)) != len(parsed_seeds):
        raise ValueError(f"{label} config seeds must be distinct non-negative integers")
    return payload, paths, settings, parsed_seeds


def _completed_experiment_output(
    path: Path,
    *,
    experiment: str,
    condition: str,
    seed: int,
    samples: int,
) -> bool:
    """Return whether an existing JSON is complete and belongs to the requested cell."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    collection = payload.get("graphs", payload.get("samples"))
    return (
        payload.get("experiment") == experiment
        and payload.get("condition") == condition
        and int(payload.get("seed", -1)) == seed
        and isinstance(collection, list)
        and len(collection) == samples
    )


def _run_or_reuse_vadgm(
    *,
    output: Path,
    overwrite: bool,
    experiment: str,
    condition: str,
    seed: int,
    samples: int,
    job_args: argparse.Namespace,
) -> None:
    if output.exists() and not overwrite:
        if _completed_experiment_output(
            output,
            experiment=experiment,
            condition=condition,
            seed=seed,
            samples=samples,
        ):
            print(
                json.dumps(
                    {
                        "experiment": experiment,
                        "condition": condition,
                        "seed": seed,
                        "status": "reused",
                        "output": str(output),
                    }
                )
            )
            return
        raise ValueError(
            f"Existing output is incomplete or belongs to another experiment cell: {output}. "
            "Set overwrite: true or move the stale file before sampling."
        )
    print(
        json.dumps(
            {
                "experiment": experiment,
                "condition": condition,
                "seed": seed,
                "status": "sampling",
                "output": str(output),
            }
        ),
        flush=True,
    )
    _sample_vadgm(job_args)


def _vadgm_common_args(
    paths: dict[str, object],
    settings: dict[str, object],
) -> dict[str, object]:
    return {
        "base_checkpoint": Path(paths["base_checkpoint"]),
        "vocabulary": Path(paths["vocabulary"]),
        "base_config": Path(paths.get("base_config", "configs/base.yaml")),
        "samples": int(settings.get("samples", 10_000)),
        "batch_size": int(settings.get("batch_size", 128)),
        "device": settings.get("device"),
    }


def _run_table2(args: argparse.Namespace) -> int:
    """Run only the missing no-terminal-atom-phase row of Table 2."""

    payload, paths, settings, seeds = _experiment_payload(args.config, label="Table 2")
    common = _vadgm_common_args(paths, settings)
    samples = int(common["samples"])
    batch_size = int(common["batch_size"])
    if samples <= 0 or batch_size <= 0:
        raise ValueError("Table 2 samples and batch_size must be positive")
    experiment = str(payload.get("experiment", "table2"))
    condition = str(payload.get("condition", "q80"))
    output_dir = Path(paths.get("output_dir", "results/runs/paper/table2"))
    sampler_config = Path(
        paths.get("sampler_config", "configs/sampling_no_terminal_phase.yaml")
    )
    guidance_checkpoint = Path(paths["guidance_checkpoint"])
    guidance_config = Path(paths.get("guidance_config", "configs/guidance.yaml"))
    overwrite = bool(settings.get("overwrite", False))

    for seed in seeds:
        output = output_dir / f"no_terminal_phase_{condition}_seed{seed}.json"
        _run_or_reuse_vadgm(
            output=output,
            overwrite=overwrite,
            experiment=experiment,
            condition=condition,
            seed=seed,
            samples=samples,
            job_args=argparse.Namespace(
                **common,
                guidance_checkpoint=guidance_checkpoint,
                guidance_config=guidance_config,
                config=sampler_config,
                seed=seed,
                delta_s=None,
                experiment=experiment,
                condition=condition,
                output=output,
            ),
        )
    return 0


def _require_reused_cell(
    pattern: str,
    *,
    experiment: str,
    condition: str,
    seeds: Sequence[int],
    samples: int,
) -> None:
    for seed in seeds:
        path = Path(pattern.format(seed=seed))
        if not _completed_experiment_output(
            path,
            experiment=experiment,
            condition=condition,
            seed=seed,
            samples=samples,
        ):
            raise ValueError(
                f"Required reusable result is missing or incomplete: {path}. "
                "Finish Table 1 before running this experiment."
            )
        print(
            json.dumps(
                {
                    "experiment": experiment,
                    "condition": condition,
                    "seed": seed,
                    "status": "reused_from_table1",
                    "output": str(path),
                }
            )
        )


def _figure3_training_args(
    *,
    paths: dict[str, object],
    settings: dict[str, object],
    guidance_checkpoint: Path,
    guidance_config: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        cache=Path(paths["source_cache"]),
        base_checkpoint=Path(paths["base_checkpoint"]),
        manifest=Path(paths["manifest"]),
        vocabulary=Path(paths["vocabulary"]),
        output=guidance_checkpoint,
        config=guidance_config,
        base_config=Path(paths.get("base_config", "configs/base.yaml")),
        source=Path(paths.get("source", "data/raw/zinc250k.csv")),
        data_config=Path(paths.get("data_config", "configs/data.yaml")),
        resume=None,
        device=settings.get("device"),
    )


def _train_figure3(args: argparse.Namespace) -> int:
    """Train the non-Q80 guidance checkpoints required by Figure 3."""

    payload, paths, settings, _ = _experiment_payload(args.config, label="Figure 3")
    targets = payload.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("Figure 3 config targets must be a non-empty YAML mapping")
    trained = 0
    for condition, raw_target in targets.items():
        if not isinstance(raw_target, dict):
            raise ValueError(f"Figure 3 target {condition} must be a YAML mapping")
        if raw_target.get("reuse_output_pattern"):
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "stage": "guidance_training",
                        "status": "not_required",
                        "reason": "Table 1 Q80 checkpoint and samples are reused",
                    }
                )
            )
            continue
        guidance_checkpoint = Path(raw_target["guidance_checkpoint"])
        guidance_config = Path(raw_target["guidance_config"])
        print(
            json.dumps(
                {
                    "condition": condition,
                    "stage": "guidance_training",
                    "status": "checking_or_training",
                    "output": str(guidance_checkpoint),
                }
            ),
            flush=True,
        )
        _train_guidance(
            _figure3_training_args(
                paths=paths,
                settings=settings,
                guidance_checkpoint=guidance_checkpoint,
                guidance_config=guidance_config,
            )
        )
        trained += 1
    if trained == 0:
        raise ValueError("Figure 3 config contains no trainable guidance target")
    return 0


def _run_figure3(args: argparse.Namespace) -> int:
    """Sample Figure 3 after all target guidance checkpoints pass readiness checks."""

    payload, paths, settings, seeds = _experiment_payload(args.config, label="Figure 3")
    targets = payload.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("Figure 3 config targets must be a non-empty YAML mapping")
    common = _vadgm_common_args(paths, settings)
    samples = int(common["samples"])
    batch_size = int(common["batch_size"])
    if samples <= 0 or batch_size <= 0:
        raise ValueError("Figure 3 samples and batch_size must be positive")
    experiment = str(payload.get("experiment", "figure3"))
    output_dir = Path(paths.get("output_dir", "results/runs/paper/figure3"))
    sampler_config = Path(paths.get("sampler_config", "configs/sampling.yaml"))
    overwrite = bool(settings.get("overwrite", False))

    for condition, raw_target in targets.items():
        if not isinstance(raw_target, dict):
            raise ValueError(f"Figure 3 target {condition} must be a YAML mapping")
        reuse_pattern = raw_target.get("reuse_output_pattern")
        if reuse_pattern:
            _require_reused_cell(
                str(reuse_pattern),
                experiment=str(raw_target.get("reuse_experiment", "table1")),
                condition=str(raw_target.get("reuse_condition", condition)),
                seeds=seeds,
                samples=samples,
            )
            continue

        guidance_checkpoint = Path(raw_target["guidance_checkpoint"])
        guidance_config = Path(raw_target["guidance_config"])
        _require_training_ready(
            guidance_checkpoint,
            guidance_config,
            artifact=f"Figure 3 {condition} guidance",
        )
        for seed in seeds:
            output = output_dir / f"{condition}_seed{seed}.json"
            _run_or_reuse_vadgm(
                output=output,
                overwrite=overwrite,
                experiment=experiment,
                condition=str(condition),
                seed=seed,
                samples=samples,
                job_args=argparse.Namespace(
                    **common,
                    guidance_checkpoint=guidance_checkpoint,
                    guidance_config=guidance_config,
                    config=sampler_config,
                    seed=seed,
                    delta_s=None,
                    experiment=experiment,
                    condition=str(condition),
                    output=output,
                ),
            )
    return 0


def _delta_tag(value: float) -> str:
    return format(value, ".8g").replace(".", "p")


def _run_figure4(args: argparse.Namespace) -> int:
    """Run the delta-s/NFE sweep used by the compact mechanism Figure 4."""

    payload, paths, settings, seeds = _experiment_payload(args.config, label="Figure 4")
    deltas = payload.get("delta_s")
    if not isinstance(deltas, list) or not deltas:
        raise ValueError("Figure 4 delta_s must be a non-empty list")
    delta_values = [float(value) for value in deltas]
    if any(value <= 0 for value in delta_values) or len(set(delta_values)) != len(delta_values):
        raise ValueError("Figure 4 delta_s values must be distinct and positive")
    common = _vadgm_common_args(paths, settings)
    samples = int(common["samples"])
    batch_size = int(common["batch_size"])
    if samples <= 0 or batch_size <= 0:
        raise ValueError("Figure 4 samples and batch_size must be positive")
    experiment = str(payload.get("experiment", "figure4"))
    condition_prefix = str(payload.get("condition_prefix", "delta_s="))
    output_dir = Path(paths.get("output_dir", "results/runs/paper/figure4"))
    sampler_config = Path(paths.get("sampler_config", "configs/sampling.yaml"))
    guidance_checkpoint = Path(paths["guidance_checkpoint"])
    guidance_config = Path(paths.get("guidance_config", "configs/guidance.yaml"))
    overwrite = bool(settings.get("overwrite", False))
    reuse = payload.get("reuse", {})
    if not isinstance(reuse, dict):
        raise ValueError("Figure 4 reuse must be a YAML mapping")
    reused_delta = float(reuse["delta_s"]) if reuse.get("delta_s") is not None else None
    reuse_pattern = str(reuse.get("output_pattern", ""))

    for delta in delta_values:
        condition = f"{condition_prefix}{format(delta, '.8g')}"
        if reused_delta is not None and math.isclose(delta, reused_delta, abs_tol=1e-12):
            if not reuse_pattern:
                raise ValueError("Figure 4 reused delta_s requires reuse.output_pattern")
            _require_reused_cell(
                reuse_pattern,
                experiment=str(reuse.get("experiment", "table1")),
                condition=str(reuse.get("condition", "q80")),
                seeds=seeds,
                samples=samples,
            )
            continue
        for seed in seeds:
            output = output_dir / f"delta_{_delta_tag(delta)}_seed{seed}.json"
            _run_or_reuse_vadgm(
                output=output,
                overwrite=overwrite,
                experiment=experiment,
                condition=condition,
                seed=seed,
                samples=samples,
                job_args=argparse.Namespace(
                    **common,
                    guidance_checkpoint=guidance_checkpoint,
                    guidance_config=guidance_config,
                    config=sampler_config,
                    seed=seed,
                    delta_s=delta,
                    experiment=experiment,
                    condition=condition,
                    output=output,
                ),
            )
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    common_base = argparse.Namespace(
        manifest=args.data_output / "manifest.csv",
        vocabulary=args.data_output / "atom_vocabulary.json",
        output=args.base_output,
        config=args.base_config,
        resume=None,
        device=args.device,
        source=args.source,
        data_config=args.data_config,
    )
    if args.through == "data":
        _ensure_dataset(
            source=args.source,
            manifest=common_base.manifest,
            vocabulary=common_base.vocabulary,
            data_config=args.data_config,
        )
    else:
        _train_base(common_base)
    if args.through in {"cache", "guidance"}:
        _build_cache(
            argparse.Namespace(
                checkpoint=args.base_output,
                manifest=common_base.manifest,
                vocabulary=common_base.vocabulary,
                output=args.cache_output,
                config=args.guidance_config,
                base_config=args.base_config,
                source=args.source,
                data_config=args.data_config,
                device=args.device,
            )
        )
    if args.through == "guidance":
        _train_guidance(
            argparse.Namespace(
                cache=args.cache_output,
                base_checkpoint=args.base_output,
                manifest=common_base.manifest,
                vocabulary=common_base.vocabulary,
                output=args.guidance_output,
                config=args.guidance_config,
                base_config=args.base_config,
                source=args.source,
                data_config=args.data_config,
                resume=None,
                device=args.device,
            )
        )
    print(
        json.dumps(
            {
                "through": args.through,
                "data": str(args.data_output),
                "base_best": str(args.base_output),
                "base_last": str(artifact_paths(args.base_output)["last"]),
                "source_cache": str(args.cache_output),
                "guidance_best": str(args.guidance_output),
                "guidance_last": str(artifact_paths(args.guidance_output)["last"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vadgm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser(
        "fetch-zinc250k",
        help="Download and verify the pinned ZINC250K benchmark source",
    )
    fetch.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/zinc250k.csv"),
    )
    fetch.set_defaults(handler=_fetch_zinc250k)

    prepare = subparsers.add_parser(
        "prepare-data",
        help="Build the deterministic ZINC100K manifest and atom vocabulary",
    )
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    prepare.set_defaults(handler=_prepare_data)

    train_base = subparsers.add_parser(
        "train-base",
        help="Train the absorbing-mask base molecular graph generator",
    )
    train_base.add_argument("--manifest", type=Path, required=True)
    train_base.add_argument("--vocabulary", type=Path, required=True)
    train_base.add_argument("--output", type=Path, required=True)
    train_base.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    train_base.add_argument("--resume", type=Path)
    train_base.add_argument("--source", type=Path, default=Path("data/raw/zinc250k.csv"))
    train_base.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    train_base.add_argument("--device")
    train_base.set_defaults(handler=_train_base)

    sample_base = subparsers.add_parser(
        "sample-base",
        help="Sample token graphs from a trained base checkpoint",
    )
    sample_base.add_argument("--checkpoint", type=Path, required=True)
    sample_base.add_argument("--vocabulary", type=Path, required=True)
    sample_base.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    sample_base.add_argument("--sizes", required=True, help="Comma-separated heavy-atom counts")
    sample_base.add_argument("--steps", type=int, default=128)
    sample_base.add_argument("--seed", type=int, default=2027)
    sample_base.add_argument("--output", type=Path, required=True)
    sample_base.add_argument("--device")
    sample_base.set_defaults(handler=_sample_base)

    build_cache = subparsers.add_parser(
        "build-cache",
        help="Generate a versioned source cache from the frozen base model",
    )
    build_cache.add_argument("--checkpoint", type=Path, required=True)
    build_cache.add_argument("--manifest", type=Path, required=True)
    build_cache.add_argument("--vocabulary", type=Path, required=True)
    build_cache.add_argument("--output", type=Path, required=True)
    build_cache.add_argument("--config", type=Path, default=Path("configs/guidance.yaml"))
    build_cache.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    build_cache.add_argument("--source", type=Path, default=Path("data/raw/zinc250k.csv"))
    build_cache.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    build_cache.add_argument("--device")
    build_cache.set_defaults(handler=_build_cache)

    train_guidance = subparsers.add_parser(
        "train-guidance",
        help="Train logP guidance with the Bregman objective",
    )
    train_guidance.add_argument("--cache", type=Path, required=True)
    train_guidance.add_argument("--base-checkpoint", type=Path, required=True)
    train_guidance.add_argument("--manifest", type=Path, required=True)
    train_guidance.add_argument("--vocabulary", type=Path, required=True)
    train_guidance.add_argument("--output", type=Path, required=True)
    train_guidance.add_argument("--config", type=Path, default=Path("configs/guidance.yaml"))
    train_guidance.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    train_guidance.add_argument("--source", type=Path, default=Path("data/raw/zinc250k.csv"))
    train_guidance.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    train_guidance.add_argument("--resume", type=Path)
    train_guidance.add_argument("--device")
    train_guidance.set_defaults(handler=_train_guidance)

    sample_guided = subparsers.add_parser(
        "sample-guided",
        help="Run vanilla DGM sampling with frozen base and guidance models",
    )
    sample_guided.add_argument("--base-checkpoint", type=Path, required=True)
    sample_guided.add_argument("--guidance-checkpoint", type=Path, required=True)
    sample_guided.add_argument("--vocabulary", type=Path, required=True)
    sample_guided.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    sample_guided.add_argument(
        "--guidance-config",
        type=Path,
        default=Path("configs/guidance.yaml"),
    )
    sample_guided.add_argument("--samples", type=int, required=True)
    sample_guided.add_argument("--steps", type=int, default=128)
    sample_guided.add_argument("--batch-size", type=int, default=256)
    sample_guided.add_argument("--seed", type=int, default=2027)
    sample_guided.add_argument("--experiment", default="unspecified")
    sample_guided.add_argument("--condition", default="unspecified")
    sample_guided.add_argument("--output", type=Path, required=True)
    sample_guided.add_argument("--device")
    sample_guided.set_defaults(handler=_sample_guided)

    vadgm_sample = subparsers.add_parser(
        "sample-vadgm",
        help="Run finite-bin VaDGM with dynamic atom-bond capacity gates",
    )
    vadgm_sample.add_argument("--base-checkpoint", type=Path, required=True)
    vadgm_sample.add_argument("--guidance-checkpoint", type=Path, required=True)
    vadgm_sample.add_argument("--vocabulary", type=Path, required=True)
    vadgm_sample.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    vadgm_sample.add_argument(
        "--guidance-config",
        type=Path,
        default=Path("configs/guidance.yaml"),
    )
    vadgm_sample.add_argument("--samples", type=int)
    vadgm_sample.add_argument("--seed", type=int)
    vadgm_sample.add_argument("--delta-s", type=float)
    vadgm_sample.add_argument("--experiment", default="unspecified")
    vadgm_sample.add_argument("--condition", default="unspecified")
    vadgm_sample.add_argument(
        "--batch-size",
        type=int,
        help="Molecules processed together; defaults to sampling.yaml",
    )
    vadgm_sample.add_argument("--output", type=Path, required=True)
    vadgm_sample.add_argument("--config", type=Path, default=Path("configs/sampling.yaml"))
    vadgm_sample.add_argument("--device")
    vadgm_sample.set_defaults(handler=_sample_vadgm)

    evaluate = subparsers.add_parser(
        "evaluate-results",
        help="Evaluate generated token graphs and export E1-E4 result tables",
    )
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--vocabulary", type=Path, required=True)
    evaluate.add_argument("--method", required=True)
    evaluate.add_argument("--target", type=float)
    evaluate.add_argument("--hit-tolerance", type=float, default=0.5)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--no-render", action="store_true")
    evaluate.set_defaults(handler=_evaluate_results)

    paper_summary = subparsers.add_parser(
        "summarize-paper",
        help="Export Table 1, Figure 3, Table 2, and Figure 4 Source Data",
    )
    paper_summary.add_argument("--input-dir", type=Path, required=True)
    paper_summary.add_argument("--vocabulary", type=Path, required=True)
    paper_summary.add_argument("--training-manifest", type=Path, required=True)
    paper_summary.add_argument("--output-dir", type=Path, required=True)
    paper_summary.add_argument("--hit-tolerance", type=float, default=0.5)
    paper_summary.set_defaults(handler=_summarize_paper)

    table1 = subparsers.add_parser(
        "run-table1",
        help="Run Table 1 Vanilla DGM, capacity-only, and full VaDGM sampling",
    )
    table1.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/table1.yaml"),
    )
    table1.set_defaults(handler=_run_table1)

    table2 = subparsers.add_parser(
        "run-table2",
        help="Run the missing no-terminal-atom-phase Table 2 condition",
    )
    table2.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/table2.yaml"),
    )
    table2.set_defaults(handler=_run_table2)

    figure3 = subparsers.add_parser(
        "run-figure3",
        help="Sample Q50/Q80/Q90 after the Figure 3 guidance models are ready",
    )
    figure3.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/figure3.yaml"),
    )
    figure3.set_defaults(handler=_run_figure3)

    figure3_training = subparsers.add_parser(
        "train-figure3",
        help="Train or resume the Q50/Q90 guidance models required by Figure 3",
    )
    figure3_training.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/figure3.yaml"),
    )
    figure3_training.set_defaults(handler=_train_figure3)

    figure4 = subparsers.add_parser(
        "run-figure4",
        help="Run the delta-s/NFE mechanism and efficiency sweep",
    )
    figure4.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/figure4.yaml"),
    )
    figure4.set_defaults(handler=_run_figure4)

    pipeline = subparsers.add_parser(
        "run-pipeline",
        help="Run prerequisites automatically through a requested training stage",
    )
    pipeline.add_argument(
        "--through",
        choices=("data", "base", "cache", "guidance"),
        default="guidance",
    )
    pipeline.add_argument("--source", type=Path, default=Path("data/raw/zinc250k.csv"))
    pipeline.add_argument(
        "--data-output",
        type=Path,
        default=Path("data/processed/zinc100k_v1"),
    )
    pipeline.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    pipeline.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    pipeline.add_argument(
        "--base-output",
        type=Path,
        default=Path("artifacts/base/best.pt"),
    )
    pipeline.add_argument(
        "--cache-output",
        type=Path,
        default=Path("data/source_cache/logp_q80.jsonl"),
    )
    pipeline.add_argument(
        "--guidance-config",
        type=Path,
        default=Path("configs/guidance.yaml"),
    )
    pipeline.add_argument(
        "--guidance-output",
        type=Path,
        default=Path("artifacts/guidance/logp_q80_best.pt"),
    )
    pipeline.add_argument("--device")
    pipeline.set_defaults(handler=_run_pipeline)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
