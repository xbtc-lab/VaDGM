"""Source-cache construction and DGM property guidance."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen
from torch import nn
from tqdm.auto import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset

from vadgm.base_model import (
    BaseGenerator,
    BaseModelConfig,
    BaseSampleDiagnostics,
    EquivariantGraphLayer,
    TimeEmbedding,
    dense_bonds_to_upper,
    sample_base_graphs,
    upper_bonds_to_dense,
)
from vadgm.chemistry import (
    AtomVocabulary,
    BondToken,
    CapacityEngine,
    GraphCodec,
    TokenGraph,
)
from vadgm.diffusion import LinearRevealSchedule, corrupt_batch, sample_training_times


@dataclass(frozen=True)
class SourceCacheRecord:
    sample_id: str
    n_atoms: int
    atom_ids: tuple[int, ...]
    bond_ids: tuple[int, ...]
    decode_ok: bool
    capacity_safe: bool
    connected: bool
    smiles: str | None
    logp: float | None
    error: str | None = None

    @property
    def chemical_support(self) -> bool:
        return self.decode_ok and self.capacity_safe and self.connected and self.logp is not None


@dataclass(frozen=True)
class TargetSpecification:
    property_name: str
    target: float
    gamma: float
    sigma: float

    def __post_init__(self) -> None:
        if self.property_name != "logP":
            raise ValueError("The first VaDGM implementation supports logP only")
        if self.gamma <= 0 or self.sigma <= 0:
            raise ValueError("gamma and sigma must be positive")


@dataclass(frozen=True)
class SizeDistribution:
    sizes: tuple[int, ...]
    probabilities: tuple[float, ...]
    normalizer: float
    effective_sample_size: float

    def __post_init__(self) -> None:
        if len(self.sizes) != len(self.probabilities) or not self.sizes:
            raise ValueError("Size distribution has inconsistent support")
        if any(value < 0 for value in self.probabilities):
            raise ValueError("Size probabilities cannot be negative")
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Size probabilities must sum to one")

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> list[int]:
        if count <= 0:
            raise ValueError("count must be positive")
        probabilities = torch.tensor(self.probabilities, dtype=torch.float64)
        indices = torch.multinomial(probabilities, count, replacement=True, generator=generator)
        return [self.sizes[int(index)] for index in indices]


@dataclass(frozen=True)
class GuidanceConfig:
    atom_vocab_size: int
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.1
    max_atoms: int = 38
    u_min: float = -12.0
    u_max: float = 8.0

    def __post_init__(self) -> None:
        if self.u_min >= self.u_max:
            raise ValueError("u_min must be smaller than u_max")


@dataclass(frozen=True)
class GuidanceLoss:
    total: torch.Tensor
    atom: torch.Tensor
    bond: torch.Tensor


@dataclass(frozen=True)
class GuidedSampleDiagnostics:
    steps: int
    base_forwards: int
    guidance_forwards: int


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def evaluate_cache_graph(
    graph: TokenGraph,
    codec: GraphCodec,
    capacity_engine: CapacityEngine,
    *,
    sample_id: str,
) -> SourceCacheRecord:
    capacity_safe = False
    try:
        capacity_safe = capacity_engine.clean_graph_is_capacity_safe(graph)
    except (ValueError, RuntimeError):
        pass
    try:
        molecule = codec.decode_molecule(graph, sanitize=True)
        fragments = Chem.GetMolFrags(molecule)
        connected = len(fragments) == 1
        smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        logp = float(Crippen.MolLogP(molecule))
        return SourceCacheRecord(
            sample_id=sample_id,
            n_atoms=graph.n_atoms,
            atom_ids=graph.atom_ids,
            bond_ids=graph.bond_ids,
            decode_ok=True,
            capacity_safe=capacity_safe,
            connected=connected,
            smiles=smiles,
            logp=logp,
        )
    except Exception as error:  # RDKit raises several non-unified exception types.
        return SourceCacheRecord(
            sample_id=sample_id,
            n_atoms=graph.n_atoms,
            atom_ids=graph.atom_ids,
            bond_ids=graph.bond_ids,
            decode_ok=False,
            capacity_safe=capacity_safe,
            connected=False,
            smiles=None,
            logp=None,
            error=f"{type(error).__name__}: {error}",
        )


def build_source_cache(
    base: BaseGenerator,
    vocabulary: AtomVocabulary,
    capacity_engine: CapacityEngine,
    size_probabilities: Mapping[int, float],
    *,
    sample_count: int,
    batch_size: int,
    sampling_steps: int,
    generator: torch.Generator,
    device: torch.device | str,
    show_progress: bool = False,
) -> tuple[list[SourceCacheRecord], dict[str, object]]:
    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("sample_count and batch_size must be positive")
    sizes = tuple(sorted(size_probabilities))
    probabilities = torch.tensor(
        [size_probabilities[size] for size in sizes],
        dtype=torch.float64,
        device=device,
    )
    if torch.any(probabilities < 0) or not torch.isclose(
        probabilities.sum(),
        torch.tensor(1.0, dtype=torch.float64, device=device),
    ):
        raise ValueError("Training size probabilities must be non-negative and sum to one")
    codec = GraphCodec(vocabulary)
    before = model_state_sha256(base)
    base.eval()
    records: list[SourceCacheRecord] = []
    total_forwards = 0
    with tqdm(
        total=sample_count,
        desc="Building source cache",
        unit="mol",
        disable=not show_progress,
    ) as progress:
        while len(records) < sample_count:
            current = min(batch_size, sample_count - len(records))
            sampled_indices = torch.multinomial(
                probabilities,
                current,
                replacement=True,
                generator=generator,
            )
            batch_sizes = [sizes[int(index)] for index in sampled_indices]
            graphs, diagnostics = sample_base_graphs(
                base,
                vocabulary,
                batch_sizes,
                steps=sampling_steps,
                generator=generator,
                device=device,
            )
            total_forwards += diagnostics.model_forwards
            for graph in graphs:
                sample_id = f"cache-{len(records):09d}"
                records.append(
                    evaluate_cache_graph(
                        graph,
                        codec,
                        capacity_engine,
                        sample_id=sample_id,
                    )
                )
            progress.update(len(graphs))
    after = model_state_sha256(base)
    if before != after:
        raise RuntimeError("Frozen base parameters changed while building source cache")
    metadata = {
        "samples": len(records),
        "valid_samples": sum(record.decode_ok for record in records),
        "chemical_support_samples": sum(record.chemical_support for record in records),
        "sampling_steps": sampling_steps,
        "model_forwards": total_forwards,
        "base_state_sha256": before,
    }
    return records, metadata


def save_source_cache(
    records: Sequence[SourceCacheRecord],
    path: str | Path,
    metadata: Mapping[str, object],
    *,
    show_progress: bool = False,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    metadata_path = target.with_suffix(target.suffix + ".meta.json")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in tqdm(
            records,
            desc="Writing source cache",
            unit="mol",
            disable=not show_progress,
        ):
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    temporary_metadata.write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    os.replace(temporary_metadata, metadata_path)


def load_source_cache(
    path: str | Path,
    *,
    show_progress: bool = False,
    total: int | None = None,
) -> list[SourceCacheRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in tqdm(
            handle,
            total=total,
            desc="Loading source cache",
            unit="mol",
            disable=not show_progress,
        ):
            payload = json.loads(line)
            payload["atom_ids"] = tuple(payload["atom_ids"])
            payload["bond_ids"] = tuple(payload["bond_ids"])
            records.append(SourceCacheRecord(**payload))
    return records


def load_source_cache_metadata(path: str | Path) -> dict[str, object]:
    metadata_path = Path(path).with_suffix(Path(path).suffix + ".meta.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Source-cache metadata is missing: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Source-cache metadata must be a JSON object")
    return payload


def training_size_distribution(n_atoms: Iterable[int]) -> dict[int, float]:
    counts: dict[int, int] = {}
    total = 0
    for value in n_atoms:
        count = int(value)
        counts[count] = counts.get(count, 0) + 1
        total += 1
    if total == 0:
        raise ValueError("Cannot estimate a size distribution from no molecules")
    return {size: count / total for size, count in sorted(counts.items())}


def estimate_training_logp_scale(
    smiles: Iterable[str],
    *,
    show_progress: bool = False,
    total: int | None = None,
) -> float:
    values = []
    for value in tqdm(
        smiles,
        total=total,
        desc="Computing training logP",
        unit="mol",
        disable=not show_progress,
    ):
        molecule = Chem.MolFromSmiles(value)
        if molecule is not None:
            values.append(float(Crippen.MolLogP(molecule)))
    if len(values) < 2:
        raise ValueError("At least two valid training molecules are needed for logP scale")
    scale = float(np.std(values, ddof=1))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Training logP scale must be finite and positive")
    return scale


def cache_target_quantile(records: Sequence[SourceCacheRecord], quantile: float) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must lie in [0, 1]")
    values = [record.logp for record in records if record.chemical_support]
    if not values:
        raise ValueError("Source cache has no chemically supported logP values")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def property_weights(
    records: Sequence[SourceCacheRecord],
    specification: TargetSpecification,
) -> torch.Tensor:
    weights = []
    denominator = 2.0 * specification.sigma**2
    for record in records:
        if not record.chemical_support:
            weights.append(0.0)
            continue
        error = float(record.logp) - specification.target
        weights.append(math.exp(-specification.gamma * error * error / denominator))
    return torch.tensor(weights, dtype=torch.float32)


def estimate_target_size_distribution(
    records: Sequence[SourceCacheRecord],
    weights: torch.Tensor,
) -> SizeDistribution:
    if len(records) != weights.numel():
        raise ValueError("records and weights must have the same length")
    if torch.any(weights < 0) or torch.any(~torch.isfinite(weights)):
        raise ValueError("weights must be finite and non-negative")
    normalizer = float(weights.sum())
    squared_sum = float((weights * weights).sum())
    if normalizer <= 0 or squared_sum <= 0:
        raise ValueError("Target has zero source-cache support")
    totals: dict[int, float] = {}
    for record, weight in zip(records, weights.tolist()):
        totals[record.n_atoms] = totals.get(record.n_atoms, 0.0) + weight
    supported = [(size, value) for size, value in sorted(totals.items()) if value > 0]
    return SizeDistribution(
        sizes=tuple(size for size, _ in supported),
        probabilities=tuple(value / normalizer for _, value in supported),
        normalizer=normalizer / len(records),
        effective_sample_size=normalizer**2 / squared_sum,
    )


class GuidanceNetwork(nn.Module):
    """Predict log-guidance u for all clean atom and bond candidates."""

    def __init__(self, config: GuidanceConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.atom_embedding = nn.Embedding(config.atom_vocab_size + 2, hidden)
        self.bond_embedding = nn.Embedding(BondToken.CLEAN_SIZE + 2, hidden)
        self.time_embedding = TimeEmbedding(hidden)
        self.size_embedding = nn.Embedding(config.max_atoms + 1, hidden)
        self.target_embedding = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.layers = nn.ModuleList(
            EquivariantGraphLayer(hidden, config.dropout)
            for _ in range(config.num_layers)
        )
        self.atom_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.atom_vocab_size),
        )
        self.bond_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, BondToken.CLEAN_SIZE),
        )

    def forward(
        self,
        atom_ids: torch.Tensor,
        bond_ids: torch.Tensor,
        atom_valid_mask: torch.Tensor,
        bond_valid_mask: torch.Tensor,
        time: torch.Tensor,
        target: torch.Tensor,
        n_atoms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_atoms = atom_ids.shape
        if max_atoms > self.config.max_atoms or torch.any(n_atoms > self.config.max_atoms):
            raise ValueError("Graph size exceeds guidance max_atoms")
        if target.shape != (batch_size,):
            raise ValueError("target must contain one property value per graph")
        dense_bonds = upper_bonds_to_dense(
            bond_ids,
            n_atoms,
            max_atoms=max_atoms,
        )
        node_mask = atom_valid_mask.bool()
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        diagonal = torch.eye(max_atoms, dtype=torch.bool, device=atom_ids.device)
        pair_mask = pair_mask & ~diagonal[None]
        condition = (
            self.time_embedding(time)
            + self.size_embedding(n_atoms)
            + self.target_embedding(target[:, None])
        )
        nodes = (self.atom_embedding(atom_ids) + condition[:, None]) * node_mask[..., None]
        edges = (
            self.bond_embedding(dense_bonds) + condition[:, None, None]
        ) * pair_mask[..., None]
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, node_mask, pair_mask)
        atom_u = self.atom_head(nodes).masked_fill(~node_mask[..., None], 0.0)
        dense_bond_u = self.bond_head(edges)
        bond_u = dense_bonds_to_upper(
            dense_bond_u,
            n_atoms,
            output_length=bond_ids.shape[1],
            pad_value=0,
        )
        bond_u = bond_u.masked_fill(~bond_valid_mask[..., None], 0.0)
        return atom_u, bond_u

    def clipped_log_guidance(
        self,
        *args: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_u, bond_u = self.forward(*args)
        return (
            atom_u.clamp(self.config.u_min, self.config.u_max),
            bond_u.clamp(self.config.u_min, self.config.u_max),
        )


def bregman_guidance_loss(
    atom_u: torch.Tensor,
    bond_u: torch.Tensor,
    clean_atom_ids: torch.Tensor,
    clean_bond_ids: torch.Tensor,
    atom_valid_mask: torch.Tensor,
    bond_valid_mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    u_min: float,
    u_max: float,
    edge_loss_weight: float = 1.0,
) -> GuidanceLoss:
    if weights.shape != (atom_u.shape[0],):
        raise ValueError("weights must contain one value per graph")
    if torch.any(~torch.isfinite(weights)) or torch.any((weights < 0) | (weights > 1)):
        raise ValueError("Bregman weights must be finite and lie in [0, 1]")
    # Keep the custom exp-based Bregman risk in FP32 under autocast. The network
    # still runs in BF16/FP16, while this sensitive reduction retains FP32 range.
    clipped_atom = atom_u.float().clamp(u_min, u_max)
    clipped_bond = bond_u.float().clamp(u_min, u_max)
    safe_atoms = torch.where(atom_valid_mask, clean_atom_ids, 0)
    safe_bonds = torch.where(bond_valid_mask, clean_bond_ids, 0)
    selected_atom = clipped_atom.gather(-1, safe_atoms[..., None]).squeeze(-1)
    if bond_u.shape[1] == 0:
        selected_bond = torch.zeros_like(safe_bonds, dtype=atom_u.dtype)
    else:
        selected_bond = clipped_bond.gather(-1, safe_bonds[..., None]).squeeze(-1)
    weight_column = weights[:, None]
    atom_values = torch.exp(selected_atom) - weight_column * selected_atom
    bond_values = torch.exp(selected_bond) - weight_column * selected_bond
    atom_per_graph = (atom_values * atom_valid_mask).sum(1) / atom_valid_mask.sum(1).clamp_min(1)
    bond_per_graph = (bond_values * bond_valid_mask).sum(1) / bond_valid_mask.sum(1).clamp_min(1)
    atom_loss = atom_per_graph.mean()
    bond_loss = bond_per_graph.mean()
    return GuidanceLoss(
        total=atom_loss + edge_loss_weight * bond_loss,
        atom=atom_loss,
        bond=bond_loss,
    )


class SourceCacheDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        records: Sequence[SourceCacheRecord],
        weights: torch.Tensor,
        vocabulary: AtomVocabulary,
        target: float,
        *,
        show_progress: bool = False,
    ) -> None:
        if len(records) != weights.numel():
            raise ValueError("records and weights length mismatch")
        self.records = tuple(records)
        self.weights = weights.float().clone()
        self.vocabulary = vocabulary
        self.target = float(target)
        self.sizes = tuple(record.n_atoms for record in self.records)
        max_atoms = max(self.sizes, default=0)
        max_bonds = max_atoms * (max_atoms - 1) // 2
        self._atom_ids = torch.full(
            (len(self.records), max_atoms),
            vocabulary.pad_id,
            dtype=torch.int16,
        )
        self._bond_ids = torch.full(
            (len(self.records), max_bonds),
            BondToken.PAD,
            dtype=torch.int16,
        )
        for index, record in enumerate(
            tqdm(
                self.records,
                desc="Packing source-cache graphs",
                unit="mol",
                disable=not show_progress,
            )
        ):
            atom_count = record.n_atoms
            bond_count = atom_count * (atom_count - 1) // 2
            self._atom_ids[index, :atom_count] = torch.tensor(
                record.atom_ids,
                dtype=torch.int16,
            )
            self._bond_ids[index, :bond_count] = torch.tensor(
                record.bond_ids,
                dtype=torch.int16,
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        atom_count = record.n_atoms
        bond_count = atom_count * (atom_count - 1) // 2
        return {
            "sample_id": record.sample_id,
            "n_atoms": atom_count,
            "atom_ids": self._atom_ids[index, :atom_count].long(),
            "bond_ids": self._bond_ids[index, :bond_count].long(),
            "atom_pad_id": self.vocabulary.pad_id,
            "weight": self.weights[index],
            "target": self.target,
        }


def collate_source_cache(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty cache batch")
    max_atoms = max(int(sample["n_atoms"]) for sample in samples)
    max_bonds = max_atoms * (max_atoms - 1) // 2
    batch_size = len(samples)
    atom_pad = int(samples[0]["atom_pad_id"])
    atoms = torch.full((batch_size, max_atoms), atom_pad, dtype=torch.long)
    bonds = torch.full((batch_size, max_bonds), BondToken.PAD, dtype=torch.long)
    atom_mask = torch.zeros_like(atoms, dtype=torch.bool)
    bond_mask = torch.zeros_like(bonds, dtype=torch.bool)
    for row, sample in enumerate(samples):
        source_atoms = sample["atom_ids"]
        source_bonds = sample["bond_ids"]
        if not isinstance(source_atoms, torch.Tensor) or not isinstance(source_bonds, torch.Tensor):
            raise TypeError("Cache samples must contain tensor token IDs")
        atoms[row, : source_atoms.numel()] = source_atoms
        bonds[row, : source_bonds.numel()] = source_bonds
        atom_mask[row, : source_atoms.numel()] = True
        bond_mask[row, : source_bonds.numel()] = True
    return {
        "sample_id": [str(sample["sample_id"]) for sample in samples],
        "n_atoms": torch.tensor([int(sample["n_atoms"]) for sample in samples]),
        "atom_ids": atoms,
        "bond_ids": bonds,
        "atom_mask": atom_mask,
        "bond_mask": bond_mask,
        "weight": torch.tensor([float(sample["weight"]) for sample in samples]),
        "target": torch.tensor([float(sample["target"]) for sample in samples]),
    }


def train_guidance_epoch(
    model: GuidanceNetwork,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    vocabulary: AtomVocabulary,
    *,
    device: torch.device | str,
    edge_loss_weight: float = 1.0,
    generator: torch.Generator | None = None,
    progress_desc: str | None = None,
    amp_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    samples = 0
    batches_iterable = (
        tqdm(loader, desc=progress_desc, unit="batch", leave=False)
        if progress_desc
        else loader
    )
    for batch in batches_iterable:
        tensors = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        time = sample_training_times(
            tensors["atom_ids"].shape[0],
            device=device,
            generator=generator,
        )
        corrupted = corrupt_batch(
            tensors["atom_ids"],
            tensors["bond_ids"],
            tensors["atom_mask"],
            tensors["bond_mask"],
            time,
            atom_mask_id=vocabulary.mask_id,
            atom_pad_id=vocabulary.pad_id,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            atom_u, bond_u = model(
                corrupted.atom_ids,
                corrupted.bond_ids,
                tensors["atom_mask"],
                tensors["bond_mask"],
                time,
                tensors["target"],
                tensors["n_atoms"],
            )
            losses = bregman_guidance_loss(
                atom_u,
                bond_u,
                tensors["atom_ids"],
                tensors["bond_ids"],
                tensors["atom_mask"],
                tensors["bond_mask"],
                tensors["weight"],
                u_min=model.config.u_min,
                u_max=model.config.u_max,
                edge_loss_weight=edge_loss_weight,
            )
        if grad_scaler is not None:
            grad_scaler.scale(losses.total).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            losses.total.backward()
            optimizer.step()
        batch_size = tensors["atom_ids"].shape[0]
        totals += torch.stack(
            (losses.total.detach(), losses.atom.detach(), losses.bond.detach())
        ).to(dtype=torch.float64) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("Guidance loader produced no batches")
    means = (totals / samples).cpu().tolist()
    return dict(zip(("loss", "atom_loss", "bond_loss"), means))


@torch.no_grad()
def evaluate_guidance_epoch(
    model: GuidanceNetwork,
    loader: Iterable[Mapping[str, object]],
    vocabulary: AtomVocabulary,
    *,
    device: torch.device | str,
    edge_loss_weight: float = 1.0,
    generator: torch.Generator | None = None,
    progress_desc: str | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    samples = 0
    batches_iterable = (
        tqdm(loader, desc=progress_desc, unit="batch", leave=False)
        if progress_desc
        else loader
    )
    for batch in batches_iterable:
        tensors = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        time = sample_training_times(
            tensors["atom_ids"].shape[0],
            device=device,
            generator=generator,
        )
        corrupted = corrupt_batch(
            tensors["atom_ids"],
            tensors["bond_ids"],
            tensors["atom_mask"],
            tensors["bond_mask"],
            time,
            atom_mask_id=vocabulary.mask_id,
            atom_pad_id=vocabulary.pad_id,
            generator=generator,
        )
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            atom_u, bond_u = model(
                corrupted.atom_ids,
                corrupted.bond_ids,
                tensors["atom_mask"],
                tensors["bond_mask"],
                time,
                tensors["target"],
                tensors["n_atoms"],
            )
            losses = bregman_guidance_loss(
                atom_u,
                bond_u,
                tensors["atom_ids"],
                tensors["bond_ids"],
                tensors["atom_mask"],
                tensors["bond_mask"],
                tensors["weight"],
                u_min=model.config.u_min,
                u_max=model.config.u_max,
                edge_loss_weight=edge_loss_weight,
            )
        batch_size = tensors["atom_ids"].shape[0]
        totals += torch.stack((losses.total, losses.atom, losses.bond)).to(
            dtype=torch.float64
        ) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("Guidance validation loader produced no batches")
    means = (totals / samples).cpu().tolist()
    return dict(zip(("loss", "atom_loss", "bond_loss"), means))


def save_guidance_checkpoint(
    path: str | Path,
    model: GuidanceNetwork,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    target_specification: TargetSpecification,
    size_distribution: SizeDistribution,
    metadata: Mapping[str, object] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "target_specification": asdict(target_specification),
        "size_distribution": asdict(size_distribution),
        "torch_rng_state": torch.get_rng_state(),
        "metadata": dict(metadata or {}),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    temporary = target.with_suffix(target.suffix + ".part")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_guidance_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    restore_rng: bool = False,
) -> tuple[GuidanceNetwork, TargetSpecification, SizeDistribution, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = GuidanceNetwork(GuidanceConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    if restore_rng and "torch_rng_state" in payload:
        # ``map_location='cuda'`` also maps the saved CPU RNG tensor to CUDA,
        # but PyTorch's default-generator API requires a CPU ByteTensor.
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["cuda_rng_state_all"]]
            )
    target = TargetSpecification(**payload["target_specification"])
    size = SizeDistribution(**payload["size_distribution"])
    return model, target, size, payload


def combine_dgm_log_scores(
    base_atom_logits: torch.Tensor,
    base_bond_logits: torch.Tensor,
    atom_u: torch.Tensor,
    bond_u: torch.Tensor,
    *,
    u_min: float,
    u_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Form log p_base + clipped u without multiplying in linear space."""

    return (
        F.log_softmax(base_atom_logits, dim=-1) + atom_u.clamp(u_min, u_max),
        F.log_softmax(base_bond_logits, dim=-1) + bond_u.clamp(u_min, u_max),
    )


def _sample_from_log_scores(
    log_scores: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if log_scores.numel() == 0:
        return torch.empty(
            log_scores.shape[:-1],
            dtype=torch.long,
            device=log_scores.device,
        )
    flat = log_scores.reshape(-1, log_scores.shape[-1])
    probabilities = torch.softmax(flat, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).reshape(
        log_scores.shape[:-1]
    )


@torch.no_grad()
def sample_guided_graphs(
    base: BaseGenerator,
    guidance: GuidanceNetwork,
    vocabulary: AtomVocabulary,
    n_atoms: Sequence[int],
    *,
    target: float,
    steps: int,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    progress_desc: str | None = None,
    verify_model_state: bool = True,
) -> tuple[list[TokenGraph], GuidedSampleDiagnostics]:
    """Vanilla DGM sampler used to verify guidance before adding VaDGM gates."""

    if steps <= 0 or not n_atoms or min(n_atoms) <= 0:
        raise ValueError("A positive step count and graph sizes are required")
    max_atoms = max(n_atoms)
    if max_atoms > min(base.config.max_atoms, guidance.config.max_atoms):
        raise ValueError("Requested graph exceeds a model maximum")
    batch_size = len(n_atoms)
    max_bonds = max_atoms * (max_atoms - 1) // 2
    n_tensor = torch.tensor(n_atoms, dtype=torch.long, device=device)
    target_tensor = torch.full(
        (batch_size,),
        float(target),
        dtype=torch.float32,
        device=device,
    )
    atom_valid = torch.arange(max_atoms, device=device)[None] < n_tensor[:, None]
    edge_counts = n_tensor * (n_tensor - 1) // 2
    bond_valid = torch.arange(max_bonds, device=device)[None] < edge_counts[:, None]
    atom_ids = torch.full(
        (batch_size, max_atoms),
        vocabulary.pad_id,
        dtype=torch.long,
        device=device,
    )
    atom_ids[atom_valid] = vocabulary.mask_id
    bond_ids = torch.full(
        (batch_size, max_bonds),
        BondToken.PAD,
        dtype=torch.long,
        device=device,
    )
    bond_ids[bond_valid] = BondToken.MASK
    schedule = LinearRevealSchedule()
    base_hash = model_state_sha256(base) if verify_model_state else None
    guidance_hash = model_state_sha256(guidance) if verify_model_state else None
    base.eval()
    guidance.eval()
    step_iterable = (
        tqdm(range(steps), desc=progress_desc, unit="step", leave=False)
        if progress_desc
        else range(steps)
    )
    for step in step_iterable:
        time = torch.full(
            (batch_size,),
            step / steps,
            dtype=torch.float32,
            device=device,
        )
        next_time = torch.full_like(time, (step + 1) / steps)
        base_atom, base_bond = base(
            atom_ids,
            bond_ids,
            atom_valid,
            bond_valid,
            time,
            n_tensor,
        )
        atom_u, bond_u = guidance(
            atom_ids,
            bond_ids,
            atom_valid,
            bond_valid,
            time,
            target_tensor,
            n_tensor,
        )
        atom_scores, bond_scores = combine_dgm_log_scores(
            base_atom,
            base_bond,
            atom_u,
            bond_u,
            u_min=guidance.config.u_min,
            u_max=guidance.config.u_max,
        )
        atom_candidates = _sample_from_log_scores(atom_scores, generator)
        bond_candidates = _sample_from_log_scores(bond_scores, generator)
        probability = schedule.conditional_reveal_probability(time, next_time)
        reveal_atoms = (
            (atom_ids == vocabulary.mask_id)
            & atom_valid
            & (torch.rand(atom_ids.shape, device=device, generator=generator) < probability[:, None])
        )
        reveal_bonds = (
            (bond_ids == BondToken.MASK)
            & bond_valid
            & (torch.rand(bond_ids.shape, device=device, generator=generator) < probability[:, None])
        )
        atom_ids[reveal_atoms] = atom_candidates[reveal_atoms]
        bond_ids[reveal_bonds] = bond_candidates[reveal_bonds]
    if torch.any(atom_ids[atom_valid] == vocabulary.mask_id) or torch.any(
        bond_ids[bond_valid] == BondToken.MASK
    ):
        raise RuntimeError("Guided sampling ended with MASK tokens")
    if verify_model_state and base_hash != model_state_sha256(base):
        raise RuntimeError("Base parameters changed during guided sampling")
    if verify_model_state and guidance_hash != model_state_sha256(guidance):
        raise RuntimeError("Guidance parameters changed during sampling")
    graphs = []
    for row, count in enumerate(n_atoms):
        edge_count = count * (count - 1) // 2
        graphs.append(
            TokenGraph(
                tuple(int(value) for value in atom_ids[row, :count].cpu()),
                tuple(int(value) for value in bond_ids[row, :edge_count].cpu()),
            )
        )
    return graphs, GuidedSampleDiagnostics(steps, steps, steps)
