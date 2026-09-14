from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vadgm.base_model import (
    BaseGenerator,
    BaseModelConfig,
    dense_bonds_to_upper,
    upper_bonds_to_dense,
)
from vadgm.chemistry import (
    AtomToken,
    AtomVocabulary,
    BondToken,
    CapacityEngine,
    GraphCodec,
    TokenGraph,
)
from vadgm.data import vocabulary_from_smiles
from vadgm.guidance_model import (
    GuidanceConfig,
    GuidanceNetwork,
    SizeDistribution,
    SourceCacheDataset,
    SourceCacheRecord,
    TargetSpecification,
    bregman_guidance_loss,
    build_source_cache,
    cache_target_quantile,
    collate_source_cache,
    combine_dgm_log_scores,
    estimate_target_size_distribution,
    evaluate_cache_graph,
    load_guidance_checkpoint,
    load_source_cache,
    load_source_cache_metadata,
    model_state_sha256,
    property_weights,
    sample_guided_graphs,
    save_guidance_checkpoint,
    save_source_cache,
    train_guidance_epoch,
)


def _record(
    sample_id: str,
    n_atoms: int,
    logp: float | None,
    *,
    supported: bool = True,
) -> SourceCacheRecord:
    return SourceCacheRecord(
        sample_id=sample_id,
        n_atoms=n_atoms,
        atom_ids=(0,) * n_atoms,
        bond_ids=(BondToken.NO_BOND,) * (n_atoms * (n_atoms - 1) // 2),
        decode_ok=supported,
        capacity_safe=supported,
        connected=supported,
        smiles="C" if supported else None,
        logp=logp if supported else None,
        error=None if supported else "invalid",
    )


def test_property_weights_keep_invalid_records_with_zero_weight() -> None:
    records = [_record("at-target", 2, 1.0), _record("away", 3, 2.0), _record("bad", 4, None, supported=False)]
    specification = TargetSpecification("logP", target=1.0, gamma=2.0, sigma=1.0)
    weights = property_weights(records, specification)
    assert weights[0].item() == 1.0
    assert 0.0 < weights[1].item() < 1.0
    assert weights[2].item() == 0.0
    assert len(weights) == len(records)


def test_target_size_distribution_and_ess_match_manual_values() -> None:
    records = [_record("a", 2, 0.0), _record("b", 2, 0.0), _record("c", 3, 0.0)]
    weights = torch.tensor([1.0, 0.5, 0.5])
    distribution = estimate_target_size_distribution(records, weights)
    assert distribution.sizes == (2, 3)
    assert distribution.probabilities == (0.75, 0.25)
    assert distribution.normalizer == 2.0 / 3.0
    assert distribution.effective_sample_size == 4.0 / 1.5


def test_cache_quantile_uses_only_chemical_support() -> None:
    records = [_record("a", 1, 1.0), _record("b", 1, 3.0), _record("bad", 1, None, supported=False)]
    assert cache_target_quantile(records, 0.5) == 2.0


def test_bregman_risk_targets_conditional_mean_and_zero_boundary() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=0.15)
    clean = torch.zeros((2, 1), dtype=torch.long)
    valid = torch.ones((2, 1), dtype=torch.bool)
    empty_long = torch.empty((2, 0), dtype=torch.long)
    empty_bool = torch.empty((2, 0), dtype=torch.bool)
    weights = torch.tensor([0.2, 0.8])
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        atom_u = parameter.expand(2, 1, 1)
        loss = bregman_guidance_loss(
            atom_u,
            torch.empty((2, 0, 4)),
            clean,
            empty_long,
            valid,
            empty_bool,
            weights,
            u_min=-8.0,
            u_max=4.0,
        ).total
        loss.backward()
        optimizer.step()
    assert abs(torch.exp(parameter).item() - 0.5) < 0.02

    zero_weight = torch.zeros(2)
    at_boundary = bregman_guidance_loss(
        torch.full((2, 1, 1), -8.0),
        torch.empty((2, 0, 4)),
        clean,
        empty_long,
        valid,
        empty_bool,
        zero_weight,
        u_min=-8.0,
        u_max=4.0,
    ).total
    above_boundary = bregman_guidance_loss(
        torch.full((2, 1, 1), -7.0),
        torch.empty((2, 0, 4)),
        clean,
        empty_long,
        valid,
        empty_bool,
        zero_weight,
        u_min=-8.0,
        u_max=4.0,
    ).total
    assert at_boundary < above_boundary


def test_bregman_reduction_uses_float32_for_low_precision_logits() -> None:
    atom_u = torch.zeros((2, 1, 1), dtype=torch.bfloat16)
    bond_u = torch.empty((2, 0, 4), dtype=torch.bfloat16)
    clean_atoms = torch.zeros((2, 1), dtype=torch.long)
    clean_bonds = torch.empty((2, 0), dtype=torch.long)
    atom_mask = torch.ones((2, 1), dtype=torch.bool)
    bond_mask = torch.empty((2, 0), dtype=torch.bool)
    loss = bregman_guidance_loss(
        atom_u,
        bond_u,
        clean_atoms,
        clean_bonds,
        atom_mask,
        bond_mask,
        torch.tensor([0.0, 1.0]),
        u_min=-12.0,
        u_max=8.0,
    )
    assert loss.total.dtype == torch.float32


def test_guidance_outputs_all_candidate_heads_and_clips_consistently() -> None:
    vocabulary = vocabulary_from_smiles(["CCO"])
    graph = GraphCodec(vocabulary).encode_smiles("CCO")
    atoms = torch.tensor([graph.atom_ids])
    bonds = torch.tensor([graph.bond_ids])
    model = GuidanceNetwork(
        GuidanceConfig(
            vocabulary.clean_size,
            hidden_dim=24,
            num_layers=1,
            dropout=0.0,
            max_atoms=8,
            u_min=-0.1,
            u_max=0.1,
        )
    )
    atom_u, bond_u = model.clipped_log_guidance(
        atoms,
        bonds,
        torch.ones_like(atoms, dtype=torch.bool),
        torch.ones_like(bonds, dtype=torch.bool),
        torch.tensor([0.5]),
        torch.tensor([1.2]),
        torch.tensor([3]),
    )
    assert atom_u.shape == (1, 3, vocabulary.clean_size)
    assert bond_u.shape == (1, 3, BondToken.CLEAN_SIZE)
    assert atom_u.min() >= -0.1 and atom_u.max() <= 0.1
    assert bond_u.min() >= -0.1 and bond_u.max() <= 0.1


def test_guidance_is_permutation_equivariant() -> None:
    vocabulary = vocabulary_from_smiles(["CCO"])
    graph = GraphCodec(vocabulary).encode_smiles("CCO")
    atoms = torch.tensor([graph.atom_ids])
    bonds = torch.tensor([graph.bond_ids])
    valid_atoms = torch.ones_like(atoms, dtype=torch.bool)
    valid_bonds = torch.ones_like(bonds, dtype=torch.bool)
    n_atoms = torch.tensor([3])
    model = GuidanceNetwork(
        GuidanceConfig(
            vocabulary.clean_size,
            hidden_dim=20,
            num_layers=2,
            dropout=0.0,
            max_atoms=5,
        )
    ).eval()
    original_atom, original_bond = model(
        atoms,
        bonds,
        valid_atoms,
        valid_bonds,
        torch.tensor([0.3]),
        torch.tensor([1.1]),
        n_atoms,
    )
    permutation = torch.tensor([2, 0, 1])
    dense = upper_bonds_to_dense(bonds, n_atoms, max_atoms=3)
    permuted_dense = dense[:, permutation][:, :, permutation]
    permuted_bonds = dense_bonds_to_upper(permuted_dense, n_atoms, output_length=3)
    permuted_atom, permuted_bond = model(
        atoms[:, permutation],
        permuted_bonds,
        valid_atoms,
        valid_bonds,
        torch.tensor([0.3]),
        torch.tensor([1.1]),
        n_atoms,
    )
    assert torch.allclose(permuted_atom, original_atom[:, permutation], atol=1e-6)
    pairs = [(0, 1), (0, 2), (1, 2)]
    lookup = {pair: index for index, pair in enumerate(pairs)}
    for new_index, (i, j) in enumerate(pairs):
        old_pair = tuple(sorted((int(permutation[i]), int(permutation[j]))))
        assert torch.allclose(
            permuted_bond[:, new_index],
            original_bond[:, lookup[old_pair]],
            atol=1e-6,
        )


def test_log_score_combination_recovers_base_when_u_is_zero() -> None:
    base_atom = torch.randn(2, 3, 5)
    base_bond = torch.randn(2, 4, 4)
    atom_score, bond_score = combine_dgm_log_scores(
        base_atom,
        base_bond,
        torch.zeros_like(base_atom),
        torch.zeros_like(base_bond),
        u_min=-10,
        u_max=10,
    )
    assert torch.allclose(atom_score, torch.log_softmax(base_atom, -1))
    assert torch.allclose(bond_score, torch.log_softmax(base_bond, -1))


def test_cache_evaluation_separates_capacity_from_rdkit_validity() -> None:
    vocabulary = AtomVocabulary([AtomToken(6, 0, 4)])
    codec = GraphCodec(vocabulary)
    engine = CapacityEngine(vocabulary)
    graph = TokenGraph((0, 0), (BondToken.SINGLE,))
    record = evaluate_cache_graph(graph, codec, engine, sample_id="invalid")
    assert not record.capacity_safe
    assert not record.chemical_support


def test_source_cache_generation_keeps_base_frozen_and_round_trips(tmp_path: Path) -> None:
    vocabulary = vocabulary_from_smiles(["C"])
    base = BaseGenerator(
        BaseModelConfig(vocabulary.clean_size, hidden_dim=12, num_layers=1, max_atoms=2)
    )
    before = model_state_sha256(base)
    records, metadata = build_source_cache(
        base,
        vocabulary,
        CapacityEngine(vocabulary),
        {1: 1.0},
        sample_count=4,
        batch_size=2,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(7),
        device="cpu",
    )
    assert len(records) == 4
    assert before == model_state_sha256(base) == metadata["base_state_sha256"]
    path = tmp_path / "cache.jsonl"
    save_source_cache(records, path, metadata)
    assert load_source_cache(path) == records
    assert load_source_cache_metadata(path) == metadata


def test_guidance_training_does_not_change_base_and_checkpoint_round_trip(tmp_path: Path) -> None:
    vocabulary = vocabulary_from_smiles(["C"])
    base = BaseGenerator(
        BaseModelConfig(vocabulary.clean_size, hidden_dim=12, num_layers=1, max_atoms=2)
    )
    base_hash = model_state_sha256(base)
    records = [_record(f"cache-{index}", 1, 0.5) for index in range(4)]
    weights = torch.tensor([0.2, 0.4, 0.6, 0.8])
    loader = DataLoader(
        SourceCacheDataset(records, weights, vocabulary, target=0.5),
        batch_size=2,
        collate_fn=collate_source_cache,
    )
    guidance = GuidanceNetwork(
        GuidanceConfig(vocabulary.clean_size, hidden_dim=12, num_layers=1, max_atoms=2)
    )
    optimizer = torch.optim.Adam(guidance.parameters(), lr=1e-3)
    metrics = train_guidance_epoch(
        guidance,
        loader,
        optimizer,
        vocabulary,
        device="cpu",
        generator=torch.Generator().manual_seed(8),
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert model_state_sha256(base) == base_hash

    checkpoint = tmp_path / "guidance.pt"
    target = TargetSpecification("logP", 0.5, 4.0, 1.0)
    size = SizeDistribution((1,), (1.0,), 0.5, 3.0)
    save_guidance_checkpoint(
        checkpoint,
        guidance,
        optimizer,
        epoch=1,
        target_specification=target,
        size_distribution=size,
    )
    restored, restored_target, restored_size, payload = load_guidance_checkpoint(
        checkpoint,
        restore_rng=True,
    )
    assert model_state_sha256(restored) == model_state_sha256(guidance)
    assert restored_target == target
    assert restored_size == size
    assert payload["epoch"] == 1


def test_guided_sampler_reveals_all_coordinates_without_updating_models() -> None:
    vocabulary = vocabulary_from_smiles(["C1CC1"])
    base = BaseGenerator(
        BaseModelConfig(vocabulary.clean_size, hidden_dim=12, num_layers=1, max_atoms=5)
    )
    guidance = GuidanceNetwork(
        GuidanceConfig(vocabulary.clean_size, hidden_dim=12, num_layers=1, max_atoms=5)
    )
    base_hash = model_state_sha256(base)
    guidance_hash = model_state_sha256(guidance)
    graphs, diagnostics = sample_guided_graphs(
        base,
        guidance,
        vocabulary,
        [1, 3],
        target=1.0,
        steps=3,
        generator=torch.Generator().manual_seed(5),
    )
    assert all(graph.is_clean(vocabulary.clean_size) for graph in graphs)
    assert diagnostics.base_forwards == diagnostics.guidance_forwards == 3
    assert model_state_sha256(base) == base_hash
    assert model_state_sha256(guidance) == guidance_hash
