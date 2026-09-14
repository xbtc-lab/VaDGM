from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vadgm.chemistry import AtomVocabulary, GraphCodec, parse_and_filter_smiles
from vadgm.data import (
    DataConfig,
    MoleculeGraphDataset,
    MoleculeRecord,
    SizeBucketBatchSampler,
    build_zinc_dataset,
    collate_graphs,
    fetch_zinc250k,
    load_manifest,
    vocabulary_from_smiles,
)


def test_fetch_zinc250k_validates_and_reuses_local_source(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("smiles,logP\nC,0.1\nCC,0.2\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "raw" / "zinc250k.csv"

    first = fetch_zinc250k(
        output,
        source_url=source.as_uri(),
        expected_sha256=digest,
        expected_rows=2,
    )
    second = fetch_zinc250k(
        output,
        source_url=source.as_uri(),
        expected_sha256=digest,
        expected_rows=2,
    )

    assert first.rows == 2 and not first.reused_existing
    assert second.rows == 2 and second.reused_existing
    assert output.read_bytes() == source.read_bytes()
    assert output.with_suffix(".csv.meta.json").exists()


def test_codec_round_trip_for_kekulized_aromatic_and_aliphatic_graphs() -> None:
    source = ["CCO", "C=C", "c1ccccc1", "[NH4+]"]
    vocabulary = vocabulary_from_smiles(source)
    codec = GraphCodec(vocabulary)

    for smiles in source:
        graph = codec.encode_smiles(smiles)
        decoded = codec.decode_smiles(graph)
        _, expected_molecule = parse_and_filter_smiles(smiles)
        _, decoded_molecule = parse_and_filter_smiles(decoded)
        expected = codec.encode_molecule(expected_molecule)
        observed = codec.encode_molecule(decoded_molecule)
        assert observed.atom_ids == expected.atom_ids
        assert observed.bond_ids == expected.bond_ids


def test_dataset_and_collate_preserve_upper_triangle_storage() -> None:
    smiles = ["CCO", "CCN"]
    vocabulary = vocabulary_from_smiles(smiles)
    codec = GraphCodec(vocabulary)
    records = [
        MoleculeRecord(f"sample-{index}", value, "train", 3)
        for index, value in enumerate(smiles)
    ]
    dataset = MoleculeGraphDataset(records, codec)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_graphs)))

    assert batch["atom_ids"].shape == (2, 3)
    assert batch["bond_ids"].shape == (2, 3)
    assert torch.all(batch["atom_mask"])
    assert torch.all(batch["bond_mask"])
    assert batch["n_atoms"].tolist() == [3, 3]


def test_size_bucket_sampler_covers_every_sample_without_wide_padding() -> None:
    sizes = (1, 2, 2, 3, 4, 5, 6)
    sampler = SizeBucketBatchSampler(
        sizes,
        batch_size=2,
        bucket_width=2,
        shuffle=False,
    )
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(sizes)))
    assert len(flattened) == len(set(flattened))
    assert len(batches) == len(sampler)
    assert sum(len(batch) < sampler.batch_size for batch in batches) == 1
    assert all(
        max(sizes[index] for index in batch)
        - min(sizes[index] for index in batch)
        <= 1
        for batch in batches
    )


def test_zinc_subset_build_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "zinc.smi"
    source.write_text(
        "\n".join(
            [
                "C1CC1",
                "C1CCC1",
                "C1CCCC1",
                "C1CCCCC1",
                "C1CCCCCC1",
                "C1CCCCCCC1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = DataConfig(
        seed=17,
        sample_size=5,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        allowed_atomic_numbers=(6,),
    )

    first = build_zinc_dataset(source, tmp_path / "first", config)
    second = build_zinc_dataset(source, tmp_path / "second", config)
    first_records = load_manifest(tmp_path / "first" / "manifest.csv")
    second_records = load_manifest(tmp_path / "second" / "manifest.csv")

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.vocabulary_sha256 == second.vocabulary_sha256
    assert first_records == second_records
    assert first.split_counts == {"train": 3, "validation": 1, "test": 1}
    assert len({record.sample_id for record in first_records}) == 5


def test_zinc_split_keeps_all_selected_atom_tokens_in_training(tmp_path: Path) -> None:
    source = tmp_path / "zinc.smi"
    source.write_text(
        "C1CC1\nC1CCC1\nC1CCCC1\nC1CCCCC1\n[PH4+]\n",
        encoding="utf-8",
    )
    config = DataConfig(
        seed=19,
        sample_size=5,
        train_fraction=0.4,
        validation_fraction=0.2,
        test_fraction=0.4,
        allowed_atomic_numbers=(6, 15),
    )
    output = tmp_path / "output"
    build_zinc_dataset(source, output, config)
    records = load_manifest(output / "manifest.csv")
    vocabulary = AtomVocabulary.load(output / "atom_vocabulary.json")
    codec = GraphCodec(vocabulary)

    assert sum(record.split == "train" for record in records) == 2
    assert any(record.smiles == "[PH4+]" and record.split == "train" for record in records)
    for record in records:
        codec.encode_smiles(record.smiles)


def test_atom_vocabulary_rejects_unknown_held_out_token() -> None:
    _, train_molecule = parse_and_filter_smiles("C1CC1")
    vocabulary = AtomVocabulary.build([train_molecule])
    codec = GraphCodec(vocabulary)

    try:
        codec.encode_smiles("CC")
    except KeyError as error:
        assert "outside the frozen vocabulary" in str(error)
    else:
        raise AssertionError("Expected a held-out atom token mismatch")
