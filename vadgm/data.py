"""Deterministic ZINC100K construction and graph datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from rdkit import Chem
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm

from vadgm.chemistry import (
    AtomVocabulary,
    BondToken,
    CapacityEngine,
    DEFAULT_ALLOWED_VALENCES,
    GraphCodec,
    TokenGraph,
    atom_token,
    parse_and_filter_smiles,
)
from vadgm.utils import sha256_file, stable_json_hash


ZINC250K_SOURCE_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "37b9f96470d4471c0593cffefa448e0a8a184ef6/"
    "models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)
ZINC250K_SOURCE_SHA256 = (
    "35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae"
)
ZINC250K_SOURCE_ROWS = 249_455
DATA_PROTOCOL_VERSION = "zinc100k-v2-token-covered-split"


@dataclass(frozen=True)
class SourceDownloadSummary:
    source_url: str
    output_path: str
    sha256: str
    rows: int
    reused_existing: bool


def _validate_zinc250k_source(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> tuple[str, int]:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(
            f"ZINC250K SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("ZINC250K source has no CSV header")
        normalized = {name.strip().lower() for name in reader.fieldnames}
        if "smiles" not in normalized and "canonical_smiles" not in normalized:
            raise ValueError("ZINC250K source has no SMILES column")
        rows = sum(1 for _ in reader)
    if rows != expected_rows:
        raise ValueError(
            f"ZINC250K row-count mismatch: expected {expected_rows}, got {rows}"
        )
    return digest, rows


def fetch_zinc250k(
    output_path: str | Path,
    *,
    source_url: str = ZINC250K_SOURCE_URL,
    expected_sha256: str = ZINC250K_SOURCE_SHA256,
    expected_rows: int = ZINC250K_SOURCE_ROWS,
    show_progress: bool = False,
) -> SourceDownloadSummary:
    """Download and verify the pinned ZINC250K benchmark source."""

    output = Path(output_path).resolve()
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    if output.exists():
        digest, rows = _validate_zinc250k_source(
            output,
            expected_sha256=expected_sha256,
            expected_rows=expected_rows,
        )
        summary = SourceDownloadSummary(
            source_url=source_url,
            output_path=str(output),
            sha256=digest,
            rows=rows,
            reused_existing=True,
        )
        metadata_path.write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".part",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = urllib.request.Request(
                source_url,
                headers={"User-Agent": "VaDGM-ZINC250K-fetch/0.1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total_bytes = int(response.headers.get("Content-Length", 0)) or None
                with tqdm(
                    total=total_bytes,
                    desc="Downloading ZINC250K",
                    unit="B",
                    unit_scale=True,
                    disable=not show_progress,
                ) as progress:
                    while chunk := response.read(1024 * 1024):
                        temporary.write(chunk)
                        progress.update(len(chunk))
        digest, rows = _validate_zinc250k_source(
            temporary_path,
            expected_sha256=expected_sha256,
            expected_rows=expected_rows,
        )
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    summary = SourceDownloadSummary(
        source_url=source_url,
        output_path=str(output),
        sha256=digest,
        rows=rows,
        reused_existing=False,
    )
    metadata_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


@dataclass(frozen=True)
class DataConfig:
    seed: int = 2027
    sample_size: int = 100_000
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    max_heavy_atoms: int = 38
    allowed_atomic_numbers: tuple[int, ...] = (5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
    require_single_component: bool = True
    reject_radicals: bool = True

    def __post_init__(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Dataset split fractions must sum to 1")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) < 0:
            raise ValueError("Dataset split fractions cannot be negative")

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "DataConfig":
        split = payload.get("split", {})
        if not isinstance(split, dict):
            raise ValueError("data config split must be a mapping")
        return cls(
            seed=int(payload.get("seed", 2027)),
            sample_size=int(payload.get("sample_size", 100_000)),
            train_fraction=float(split.get("train", 0.8)),
            validation_fraction=float(split.get("validation", 0.1)),
            test_fraction=float(split.get("test", 0.1)),
            max_heavy_atoms=int(payload.get("max_heavy_atoms", 38)),
            allowed_atomic_numbers=tuple(
                int(value)
                for value in payload.get(
                    "allowed_atomic_numbers",
                    (5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53),
                )
            ),
            require_single_component=bool(payload.get("require_single_component", True)),
            reject_radicals=bool(payload.get("reject_radicals", True)),
        )


@dataclass(frozen=True)
class MoleculeRecord:
    sample_id: str
    smiles: str
    split: str
    n_heavy_atoms: int


@dataclass(frozen=True)
class BuildSummary:
    protocol_version: str
    source_path: str
    source_sha256: str
    config_hash: str
    seen_rows: int
    accepted_unique: int
    rejected_rows: int
    duplicate_rows: int
    selected_rows: int
    split_counts: dict[str, int]
    manifest_sha256: str
    vocabulary_sha256: str


def _stable_rank(smiles: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{smiles}".encode("utf-8")).hexdigest()


def _sample_id(smiles: str) -> str:
    return "zinc-" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:16]


def iter_source_smiles(path: str | Path) -> Iterator[str]:
    """Read a SMILES column from CSV/TSV or the first field of a text file."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"No header found in {source}")
            normalized = {name.strip().lower(): name for name in reader.fieldnames}
            smiles_field = next(
                (normalized[name] for name in ("smiles", "canonical_smiles") if name in normalized),
                None,
            )
            if smiles_field is None:
                raise ValueError(f"No SMILES column found in {source}")
            for row in reader:
                value = row.get(smiles_field)
                if value:
                    yield value
        return

    with source.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped.split()[0]


def _split_labels_with_train_token_coverage(
    selected_smiles: Sequence[str],
    molecules: dict[str, Chem.Mol],
    config: DataConfig,
) -> dict[str, str]:
    """Assign fixed-size splits while keeping every atom token in training."""

    n_train = int(config.sample_size * config.train_fraction)
    n_validation = int(config.sample_size * config.validation_fraction)
    token_representatives: dict[object, str] = {}
    for smiles in selected_smiles:
        for atom in molecules[smiles].GetAtoms():
            token_representatives.setdefault(atom_token(atom), smiles)
    required_train = set(token_representatives.values())
    if len(required_train) > n_train:
        raise ValueError(
            "Training split is too small to cover every selected atom token: "
            f"{len(required_train)} representatives for {n_train} slots"
        )
    train = set(required_train)
    for smiles in selected_smiles:
        if len(train) == n_train:
            break
        train.add(smiles)
    remaining = [smiles for smiles in selected_smiles if smiles not in train]
    validation = set(remaining[:n_validation])
    return {
        smiles: (
            "train"
            if smiles in train
            else "validation"
            if smiles in validation
            else "test"
        )
        for smiles in selected_smiles
    }


def _write_manifest(path: Path, records: Sequence[MoleculeRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "smiles", "split", "n_heavy_atoms"],
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def load_manifest(path: str | Path, split: str | None = None) -> list[MoleculeRecord]:
    records: list[MoleculeRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = MoleculeRecord(
                sample_id=row["sample_id"],
                smiles=row["smiles"],
                split=row["split"],
                n_heavy_atoms=int(row["n_heavy_atoms"]),
            )
            if split is None or record.split == split:
                records.append(record)
    return records


def build_zinc_dataset(
    source_path: str | Path,
    output_dir: str | Path,
    config: DataConfig,
    *,
    show_progress: bool = False,
) -> BuildSummary:
    """Filter first, then deterministically select and split the ZINC subset.

    Existing outputs are never overwritten.  A protocol change must create a
    new versioned output directory, matching the research change-control rule.
    """

    source = Path(source_path).resolve()
    destination = Path(output_dir).resolve()
    manifest_path = destination / "manifest.csv"
    vocabulary_path = destination / "atom_vocabulary.json"
    metadata_path = destination / "metadata.json"
    for path in (manifest_path, vocabulary_path, metadata_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite versioned dataset artifact: {path}"
            )

    accepted: dict[str, Chem.Mol] = {}
    seen_rows = 0
    rejected_rows = 0
    duplicate_rows = 0
    allowed = set(config.allowed_atomic_numbers)
    source_sha256 = sha256_file(source)
    expected_source_rows = (
        ZINC250K_SOURCE_ROWS
        if source_sha256 == ZINC250K_SOURCE_SHA256
        else None
    )
    for raw_smiles in tqdm(
        iter_source_smiles(source),
        total=expected_source_rows,
        desc="Filtering ZINC source",
        unit="mol",
        disable=not show_progress,
    ):
        seen_rows += 1
        try:
            canonical, molecule = parse_and_filter_smiles(
                raw_smiles,
                allowed_atomic_numbers=allowed,
                allowed_charge_states=set(DEFAULT_ALLOWED_VALENCES),
                max_heavy_atoms=config.max_heavy_atoms,
                require_single_component=config.require_single_component,
                reject_radicals=config.reject_radicals,
            )
        except (ValueError, RuntimeError):
            rejected_rows += 1
            continue
        if canonical in accepted:
            duplicate_rows += 1
            continue
        accepted[canonical] = molecule

    if len(accepted) < config.sample_size:
        raise ValueError(
            f"Only {len(accepted)} unique molecules passed the protocol; "
            f"{config.sample_size} are required"
        )

    selected_smiles = sorted(
        accepted,
        key=lambda smiles: (_stable_rank(smiles, config.seed), smiles),
    )[: config.sample_size]
    split_by_smiles = _split_labels_with_train_token_coverage(
        selected_smiles,
        accepted,
        config,
    )
    records = [
        MoleculeRecord(
            sample_id=_sample_id(smiles),
            smiles=smiles,
            split=split_by_smiles[smiles],
            n_heavy_atoms=accepted[smiles].GetNumHeavyAtoms(),
        )
        for smiles in selected_smiles
    ]

    train_molecules = [
        accepted[record.smiles] for record in records if record.split == "train"
    ]
    vocabulary = AtomVocabulary.build(train_molecules)
    codec = GraphCodec(vocabulary)
    capacity = CapacityEngine(vocabulary)

    # Held-out token coverage and encoded capacity are hard protocol checks.
    for record in tqdm(
        records,
        desc="Validating ZINC100K",
        unit="mol",
        disable=not show_progress,
    ):
        graph = codec.encode_molecule(accepted[record.smiles], record.smiles)
        if not capacity.clean_graph_is_capacity_safe(graph):
            raise ValueError(
                f"Selected molecule violates the declared capacity protocol: {record.smiles}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, records)
    vocabulary.save(vocabulary_path)
    split_counts = {
        name: sum(record.split == name for record in records)
        for name in ("train", "validation", "test")
    }
    summary_without_output_hashes = {
        "protocol_version": DATA_PROTOCOL_VERSION,
        "source_path": str(source),
        "source_sha256": source_sha256,
        "config_hash": stable_json_hash(
            {
                "protocol_version": DATA_PROTOCOL_VERSION,
                "config": asdict(config),
            }
        ),
        "seen_rows": seen_rows,
        "accepted_unique": len(accepted),
        "rejected_rows": rejected_rows,
        "duplicate_rows": duplicate_rows,
        "selected_rows": len(records),
        "split_counts": split_counts,
    }
    summary = BuildSummary(
        **summary_without_output_hashes,
        manifest_sha256=sha256_file(manifest_path),
        vocabulary_sha256=sha256_file(vocabulary_path),
    )
    metadata_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


class MoleculeGraphDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        records: Sequence[MoleculeRecord],
        codec: GraphCodec,
        *,
        show_progress: bool = False,
    ) -> None:
        self.records = tuple(records)
        self.codec = codec
        self.sizes = tuple(record.n_heavy_atoms for record in self.records)
        max_atoms = max(self.sizes, default=0)
        max_bonds = max_atoms * (max_atoms - 1) // 2
        self._atom_ids = torch.full(
            (len(self.records), max_atoms),
            codec.atom_vocabulary.pad_id,
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
                desc="Pre-encoding molecular graphs",
                unit="mol",
                disable=not show_progress,
            )
        ):
            graph = codec.encode_smiles(record.smiles)
            if graph.n_atoms != record.n_heavy_atoms:
                raise ValueError(
                    f"Manifest size mismatch for {record.sample_id}: "
                    f"{record.n_heavy_atoms} != {graph.n_atoms}"
                )
            atom_count = graph.n_atoms
            bond_count = atom_count * (atom_count - 1) // 2
            self._atom_ids[index, :atom_count] = torch.tensor(
                graph.atom_ids,
                dtype=torch.int16,
            )
            self._bond_ids[index, :bond_count] = torch.tensor(
                graph.bond_ids,
                dtype=torch.int16,
            )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        vocabulary_path: str | Path,
        *,
        split: str,
        show_progress: bool = False,
    ) -> "MoleculeGraphDataset":
        vocabulary = AtomVocabulary.load(vocabulary_path)
        records = load_manifest(manifest_path, split=split)
        return cls(
            records,
            GraphCodec(vocabulary),
            show_progress=show_progress,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        atom_count = self.sizes[index]
        bond_count = atom_count * (atom_count - 1) // 2
        return {
            "sample_id": record.sample_id,
            "smiles": record.smiles,
            "n_atoms": atom_count,
            "atom_ids": self._atom_ids[index, :atom_count].long(),
            "bond_ids": self._bond_ids[index, :bond_count].long(),
            "atom_pad_id": self.codec.atom_vocabulary.pad_id,
        }


class SizeBucketBatchSampler(Sampler[list[int]]):
    """Group similarly sized graphs to reduce quadratic dense padding."""

    def __init__(
        self,
        sizes: Sequence[int],
        batch_size: int,
        *,
        bucket_width: int = 2,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size <= 0 or bucket_width <= 0:
            raise ValueError("batch_size and bucket_width must be positive")
        if not sizes or min(sizes) <= 0:
            raise ValueError("sizes must contain positive graph sizes")
        self.batch_size = int(batch_size)
        self.bucket_width = int(bucket_width)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator
        self.sizes = tuple(int(size) for size in sizes)
        buckets: dict[int, list[int]] = {}
        for index, size in enumerate(self.sizes):
            bucket = (int(size) - 1) // self.bucket_width
            buckets.setdefault(bucket, []).append(index)
        self._buckets = tuple(tuple(indices) for _, indices in sorted(buckets.items()))

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        leftovers: list[int] = []
        for source_indices in self._buckets:
            indices = list(source_indices)
            if self.shuffle:
                order = torch.randperm(len(indices), generator=self.generator).tolist()
                indices = [indices[position] for position in order]
            full_stop = len(indices) - len(indices) % self.batch_size
            batches.extend(
                indices[start : start + self.batch_size]
                for start in range(0, full_stop, self.batch_size)
            )
            leftovers.extend(indices[full_stop:])
        leftovers.sort(key=lambda index: self.sizes[index])
        for start in range(0, len(leftovers), self.batch_size):
            batch = leftovers[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[position] for position in order]
        yield from batches

    def __len__(self) -> int:
        total = sum(len(indices) for indices in self._buckets)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


def collate_graphs(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    max_atoms = max(int(sample["n_atoms"]) for sample in samples)
    max_bonds = max_atoms * (max_atoms - 1) // 2
    batch_size = len(samples)
    atom_pad_ids = {int(sample["atom_pad_id"]) for sample in samples}
    if len(atom_pad_ids) != 1:
        raise ValueError("All samples in a batch must use the same frozen atom vocabulary")
    atom_pad_id = atom_pad_ids.pop()
    atom_ids = torch.full((batch_size, max_atoms), atom_pad_id, dtype=torch.long)
    bond_ids = torch.full((batch_size, max_bonds), BondToken.PAD, dtype=torch.long)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    bond_mask = torch.zeros((batch_size, max_bonds), dtype=torch.bool)
    n_atoms = torch.empty(batch_size, dtype=torch.long)

    for row, sample in enumerate(samples):
        atoms = sample["atom_ids"]
        bonds = sample["bond_ids"]
        if not isinstance(atoms, torch.Tensor) or not isinstance(bonds, torch.Tensor):
            raise TypeError("Dataset samples must contain tensor atom_ids and bond_ids")
        atom_count = atoms.numel()
        bond_count = bonds.numel()
        atom_ids[row, :atom_count] = atoms
        bond_ids[row, :bond_count] = bonds
        atom_mask[row, :atom_count] = True
        bond_mask[row, :bond_count] = True
        n_atoms[row] = atom_count

    return {
        "sample_id": [str(sample["sample_id"]) for sample in samples],
        "smiles": [str(sample["smiles"]) for sample in samples],
        "n_atoms": n_atoms,
        "atom_ids": atom_ids,
        "bond_ids": bond_ids,
        "atom_mask": atom_mask,
        "bond_mask": bond_mask,
    }


def vocabulary_from_smiles(smiles: Iterable[str]) -> AtomVocabulary:
    molecules = []
    for value in smiles:
        _, molecule = parse_and_filter_smiles(value)
        molecules.append(molecule)
    tokens = sorted({atom_token(atom) for molecule in molecules for atom in molecule.GetAtoms()})
    return AtomVocabulary(tokens)
