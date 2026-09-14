from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vadgm.base_model import (
    BaseGenerator,
    BaseModelConfig,
    base_denoising_loss,
    dense_bonds_to_upper,
    load_base_checkpoint,
    sample_base_graphs,
    save_base_checkpoint,
    train_base_epoch,
    upper_bonds_to_dense,
)
from vadgm.chemistry import BondToken, GraphCodec
from vadgm.data import vocabulary_from_smiles
from vadgm.data import MoleculeGraphDataset, MoleculeRecord, collate_graphs
from vadgm.diffusion import LinearRevealSchedule, corrupt_batch


def _clean_batch(smiles: str) -> tuple[object, dict[str, torch.Tensor]]:
    vocabulary = vocabulary_from_smiles([smiles])
    graph = GraphCodec(vocabulary).encode_smiles(smiles)
    atoms = torch.tensor([graph.atom_ids], dtype=torch.long)
    bonds = torch.tensor([graph.bond_ids], dtype=torch.long)
    return vocabulary, {
        "atom_ids": atoms,
        "bond_ids": bonds,
        "atom_mask": torch.ones_like(atoms, dtype=torch.bool),
        "bond_mask": torch.ones_like(bonds, dtype=torch.bool),
        "n_atoms": torch.tensor([graph.n_atoms]),
    }


def test_linear_schedule_hazard_inverse_and_endpoints() -> None:
    schedule = LinearRevealSchedule()
    time = torch.tensor([0.0, 0.2, 0.7, 0.99])
    assert torch.allclose(schedule.inverse_hazard(schedule.hazard(time)), time)
    assert schedule.kappa(torch.tensor([0.0, 1.0])).tolist() == [0.0, 1.0]


def test_absorbing_corruption_has_correct_endpoints() -> None:
    vocabulary, batch = _clean_batch("CCO")
    for time_value, expect_clean in ((0.0, False), (1.0, True)):
        corrupted = corrupt_batch(
            batch["atom_ids"],
            batch["bond_ids"],
            batch["atom_mask"],
            batch["bond_mask"],
            torch.tensor([time_value]),
            atom_mask_id=vocabulary.mask_id,
            atom_pad_id=vocabulary.pad_id,
        )
        if expect_clean:
            assert torch.equal(corrupted.atom_ids, batch["atom_ids"])
            assert torch.equal(corrupted.bond_ids, batch["bond_ids"])
        else:
            assert torch.all(corrupted.atom_ids == vocabulary.mask_id)
            assert torch.all(corrupted.bond_ids == BondToken.MASK)


def test_base_output_contains_only_clean_category_heads() -> None:
    vocabulary, batch = _clean_batch("CCO")
    model = BaseGenerator(
        BaseModelConfig(vocabulary.clean_size, hidden_dim=32, num_layers=2, max_atoms=8)
    )
    corrupted = corrupt_batch(
        batch["atom_ids"],
        batch["bond_ids"],
        batch["atom_mask"],
        batch["bond_mask"],
        torch.tensor([0.4]),
        atom_mask_id=vocabulary.mask_id,
        atom_pad_id=vocabulary.pad_id,
        generator=torch.Generator().manual_seed(3),
    )
    atom_logits, bond_logits = model(
        corrupted.atom_ids,
        corrupted.bond_ids,
        batch["atom_mask"],
        batch["bond_mask"],
        corrupted.time,
        batch["n_atoms"],
    )
    assert atom_logits.shape == (1, 3, vocabulary.clean_size)
    assert bond_logits.shape == (1, 3, BondToken.CLEAN_SIZE)


def test_vectorized_packed_bond_round_trip_supports_mixed_graph_sizes() -> None:
    n_atoms = torch.tensor([2, 3, 4])
    packed = torch.tensor(
        [
            [
                1,
                BondToken.PAD,
                BondToken.PAD,
                BondToken.PAD,
                BondToken.PAD,
                BondToken.PAD,
            ],
            [1, 2, 3, BondToken.PAD, BondToken.PAD, BondToken.PAD],
            [1, 2, 3, 0, 1, 2],
        ]
    )
    dense = upper_bonds_to_dense(packed, n_atoms, max_atoms=4)
    restored = dense_bonds_to_upper(dense, n_atoms, output_length=6)
    assert torch.equal(restored, packed)


def test_base_model_is_permutation_equivariant_and_edge_symmetric() -> None:
    vocabulary, batch = _clean_batch("CCO")
    model = BaseGenerator(
        BaseModelConfig(
            vocabulary.clean_size,
            hidden_dim=24,
            num_layers=2,
            dropout=0.0,
            max_atoms=8,
        )
    ).eval()
    atom_ids = batch["atom_ids"].clone()
    atom_ids[0, 1] = vocabulary.mask_id
    bond_ids = batch["bond_ids"].clone()
    bond_ids[0, 1] = BondToken.MASK
    time = torch.tensor([0.37])
    original_atom, original_bond = model(
        atom_ids,
        bond_ids,
        batch["atom_mask"],
        batch["bond_mask"],
        time,
        batch["n_atoms"],
    )

    permutation = torch.tensor([2, 0, 1])
    dense = upper_bonds_to_dense(bond_ids, batch["n_atoms"], max_atoms=3)
    permuted_dense = dense[:, permutation][:, :, permutation]
    permuted_bonds = dense_bonds_to_upper(
        permuted_dense,
        batch["n_atoms"],
        output_length=3,
    )
    permuted_atom, permuted_bond = model(
        atom_ids[:, permutation],
        permuted_bonds,
        batch["atom_mask"],
        batch["bond_mask"],
        time,
        batch["n_atoms"],
    )
    assert torch.allclose(permuted_atom, original_atom[:, permutation], atol=1e-6)

    original_pairs = [(0, 1), (0, 2), (1, 2)]
    original_lookup = {pair: index for index, pair in enumerate(original_pairs)}
    for new_index, (i, j) in enumerate(original_pairs):
        old_pair = tuple(sorted((int(permutation[i]), int(permutation[j]))))
        assert torch.allclose(
            permuted_bond[:, new_index],
            original_bond[:, original_lookup[old_pair]],
            atol=1e-6,
        )


def test_node_and_edge_losses_are_separately_normalized() -> None:
    vocabulary, batch = _clean_batch("CCO")
    atom_logits = torch.zeros(1, 3, vocabulary.clean_size)
    bond_logits = torch.zeros(1, 3, BondToken.CLEAN_SIZE)
    partial_atoms = torch.full_like(batch["atom_ids"], vocabulary.mask_id)
    partial_bonds = torch.full_like(batch["bond_ids"], BondToken.MASK)
    losses = base_denoising_loss(
        atom_logits,
        bond_logits,
        batch["atom_ids"],
        batch["bond_ids"],
        partial_atoms,
        partial_bonds,
        batch["atom_mask"],
        batch["bond_mask"],
        atom_mask_id=vocabulary.mask_id,
    )
    assert torch.allclose(losses.atom, torch.log(torch.tensor(float(vocabulary.clean_size))))
    assert torch.allclose(losses.bond, torch.log(torch.tensor(float(BondToken.CLEAN_SIZE))))
    assert losses.masked_atoms == 3
    assert losses.masked_bonds == 3


def test_tiny_graph_can_overfit_and_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(11)
    vocabulary, batch = _clean_batch("C1CC1")
    model = BaseGenerator(
        BaseModelConfig(
            vocabulary.clean_size,
            hidden_dim=24,
            num_layers=1,
            dropout=0.0,
            max_atoms=8,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    partial_atoms = torch.full_like(batch["atom_ids"], vocabulary.mask_id)
    partial_bonds = torch.full_like(batch["bond_ids"], BondToken.MASK)

    def loss_value() -> torch.Tensor:
        atom_logits, bond_logits = model(
            partial_atoms,
            partial_bonds,
            batch["atom_mask"],
            batch["bond_mask"],
            torch.zeros(1),
            batch["n_atoms"],
        )
        return base_denoising_loss(
            atom_logits,
            bond_logits,
            batch["atom_ids"],
            batch["bond_ids"],
            partial_atoms,
            partial_bonds,
            batch["atom_mask"],
            batch["bond_mask"],
            atom_mask_id=vocabulary.mask_id,
        ).total

    initial = float(loss_value().detach())
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
    final = float(loss_value().detach())
    assert final < initial * 0.1

    checkpoint = tmp_path / "base.pt"
    save_base_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=35,
        best_validation_loss=final,
        metadata={"vocabulary_size": vocabulary.clean_size},
    )
    restored, payload = load_base_checkpoint(checkpoint, restore_rng=True)
    restored.eval()
    model.eval()
    with torch.no_grad():
        expected = model(
            partial_atoms,
            partial_bonds,
            batch["atom_mask"],
            batch["bond_mask"],
            torch.zeros(1),
            batch["n_atoms"],
        )
        observed = restored(
            partial_atoms,
            partial_bonds,
            batch["atom_mask"],
            batch["bond_mask"],
            torch.zeros(1),
            batch["n_atoms"],
        )
    assert all(torch.equal(left, right) for left, right in zip(expected, observed))
    assert payload["epoch"] == 35


def test_base_sampler_reveals_every_valid_coordinate() -> None:
    vocabulary = vocabulary_from_smiles(["C1CC1"])
    model = BaseGenerator(
        BaseModelConfig(
            vocabulary.clean_size,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            max_atoms=8,
        )
    )
    graphs, diagnostics = sample_base_graphs(
        model,
        vocabulary,
        [1, 3, 4],
        steps=4,
        generator=torch.Generator().manual_seed(9),
    )
    assert all(graph.is_clean(vocabulary.clean_size) for graph in graphs)
    assert diagnostics.model_forwards == 4
    assert diagnostics.revealed_atoms == 8
    assert diagnostics.revealed_bonds == 9


def test_single_atom_graph_has_zero_bond_loss_and_can_be_sampled() -> None:
    vocabulary = vocabulary_from_smiles(["C"])
    model = BaseGenerator(
        BaseModelConfig(vocabulary.clean_size, hidden_dim=16, num_layers=1, max_atoms=4)
    )
    atoms = torch.tensor([[vocabulary.mask_id]])
    empty_bonds = torch.empty((1, 0), dtype=torch.long)
    atom_logits, bond_logits = model(
        atoms,
        empty_bonds,
        torch.ones_like(atoms, dtype=torch.bool),
        torch.empty((1, 0), dtype=torch.bool),
        torch.zeros(1),
        torch.ones(1, dtype=torch.long),
    )
    losses = base_denoising_loss(
        atom_logits,
        bond_logits,
        torch.zeros_like(atoms),
        empty_bonds,
        atoms,
        empty_bonds,
        torch.ones_like(atoms, dtype=torch.bool),
        torch.empty((1, 0), dtype=torch.bool),
        atom_mask_id=vocabulary.mask_id,
    )
    assert losses.bond.item() == 0.0
    graphs, _ = sample_base_graphs(model, vocabulary, [1], steps=2)
    assert graphs[0].is_clean(vocabulary.clean_size)


def test_training_epoch_runs_through_dataset_pipeline() -> None:
    vocabulary = vocabulary_from_smiles(["C1CC1"])
    records = [
        MoleculeRecord(f"ring-{index}", "C1CC1", "train", 3)
        for index in range(4)
    ]
    loader = DataLoader(
        MoleculeGraphDataset(records, GraphCodec(vocabulary)),
        batch_size=2,
        collate_fn=collate_graphs,
    )
    model = BaseGenerator(
        BaseModelConfig(
            vocabulary.clean_size,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            max_atoms=8,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = train_base_epoch(
        model,
        loader,
        optimizer,
        vocabulary,
        device="cpu",
        generator=torch.Generator().manual_seed(4),
    )
    assert set(metrics) == {"loss", "atom_loss", "bond_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
