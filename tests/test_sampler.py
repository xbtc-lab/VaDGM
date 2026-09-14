from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from vadgm.chemistry import (
    AtomToken,
    AtomVocabulary,
    BondToken,
    CapacityEngine,
    TokenGraph,
)
from vadgm.diagnostics import (
    Event,
    EventPlan,
    Proposal,
    all_coordinates,
    detect_joint_conflict,
)
from vadgm.sampler import result_to_dict, sample_vadgm, sample_vadgm_batch


def _vocabulary() -> AtomVocabulary:
    return AtomVocabulary([AtomToken(6, 0, 0), AtomToken(6, 0, 4)])


def _manual_plan(n_atoms: int, *, same_bin: bool) -> EventPlan:
    coordinates = all_coordinates(n_atoms)
    if same_bin:
        delta_s = 1.0
        clocks = [0.01 * (index + 1) for index in range(len(coordinates))]
    else:
        delta_s = 0.01
        clocks = [0.005 + 0.02 * index for index in range(len(coordinates))]
    events = tuple(
        Event(coordinate, clock, int(clock // delta_s))
        for coordinate, clock in zip(coordinates, clocks)
    )
    return EventPlan(n_atoms, delta_s, events)


class FixedBase(nn.Module):
    def __init__(self, atom_vocab_size: int, *, nan: bool = False) -> None:
        super().__init__()
        self.atom_vocab_size = atom_vocab_size
        self.nan = nan
        self.config = SimpleNamespace(max_atoms=16)

    def forward(
        self,
        atom_ids: torch.Tensor,
        bond_ids: torch.Tensor,
        atom_valid_mask: torch.Tensor,
        bond_valid_mask: torch.Tensor,
        time: torch.Tensor,
        n_atoms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fill = torch.nan if self.nan else -1000.0
        atoms = torch.full(
            (*atom_ids.shape, self.atom_vocab_size),
            fill,
            device=atom_ids.device,
        )
        bonds = torch.full(
            (*bond_ids.shape, BondToken.CLEAN_SIZE),
            fill,
            device=bond_ids.device,
        )
        if not self.nan:
            atoms[..., 1] = 1000.0  # Prefer zero-capacity CH4.
            bonds[..., BondToken.NO_BOND] = 0.0
            bonds[..., BondToken.SINGLE] = 1000.0
        return atoms, bonds


class ZeroGuidance(nn.Module):
    def __init__(self, atom_vocab_size: int) -> None:
        super().__init__()
        self.atom_vocab_size = atom_vocab_size
        self.config = SimpleNamespace(max_atoms=16, u_min=-12.0, u_max=8.0)

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
        return (
            torch.zeros((*atom_ids.shape, self.atom_vocab_size), device=atom_ids.device),
            torch.zeros((*bond_ids.shape, BondToken.CLEAN_SIZE), device=bond_ids.device),
        )


def test_event_plan_is_reproducible_and_covers_every_coordinate() -> None:
    first = EventPlan.sample(
        4,
        0.2,
        generator=torch.Generator().manual_seed(21),
    )
    second = EventPlan.sample(
        4,
        0.2,
        generator=torch.Generator().manual_seed(21),
    )
    assert first == second
    assert len(first.events) == 4 + 6
    assert len({event.coordinate for event in first.events}) == 10
    assert all(
        left.clock < right.clock for left, right in zip(first.events, first.events[1:])
    )


def test_terminal_atom_phase_preserves_coordinates_and_orders_atoms_last() -> None:
    original = EventPlan.sample(
        4,
        0.2,
        generator=torch.Generator().manual_seed(22),
    )
    transformed = original.with_terminal_atom_phase()
    assert {event.coordinate for event in transformed.events} == {
        event.coordinate for event in original.events
    }
    bond_clocks = [event.clock for event in transformed.events if event.coordinate.kind == "bond"]
    atom_clocks = [event.clock for event in transformed.events if event.coordinate.kind == "atom"]
    assert max(bond_clocks) < min(atom_clocks)
    assert len({event.bin_index for event in transformed.events if event.coordinate.kind == "atom"}) == 1


def test_bond_bond_joint_conflict_matches_definition() -> None:
    vocabulary = AtomVocabulary([AtomToken(6, 0, 0)])
    capacity = CapacityEngine(vocabulary)
    graph = TokenGraph(
        atom_ids=(0, 0, 0, 0),
        # (0,1)=triple; (0,2)/(0,3) remain masked.
        bond_ids=(3, BondToken.MASK, BondToken.MASK, 0, 0, 0),
    )
    coordinates = all_coordinates(4)
    proposals = (
        Proposal(coordinates[4 + 1], BondToken.SINGLE),
        Proposal(coordinates[4 + 2], BondToken.SINGLE),
    )
    conflict = detect_joint_conflict(graph, proposals, capacity)
    assert conflict.occurred
    assert conflict.conflict_type == "bond-bond"
    assert conflict.affected_atoms == (0,)
    assert conflict.excess_by_atom == ((0, 1),)


def test_atom_bond_joint_conflict_is_detected_separately() -> None:
    vocabulary = _vocabulary()
    capacity = CapacityEngine(vocabulary)
    graph = TokenGraph(
        atom_ids=(vocabulary.mask_id, vocabulary.mask_id),
        bond_ids=(BondToken.MASK,),
    )
    coordinates = all_coordinates(2)
    low_capacity_atom = vocabulary.encode(AtomToken(6, 0, 4))
    conflict = detect_joint_conflict(
        graph,
        (
            Proposal(coordinates[0], low_capacity_atom),
            Proposal(coordinates[2], BondToken.SINGLE),
        ),
        capacity,
    )
    assert conflict.occurred
    assert conflict.conflict_type == "atom-bond"
    assert conflict.affected_atoms == (0,)


def test_single_or_noncompeting_proposals_do_not_create_joint_conflict() -> None:
    vocabulary = AtomVocabulary([AtomToken(6, 0, 0)])
    capacity = CapacityEngine(vocabulary)
    graph = TokenGraph(
        atom_ids=(0, 0, 0),
        bond_ids=(BondToken.MASK, BondToken.MASK, BondToken.MASK),
    )
    coordinate = all_coordinates(3)[3]
    conflict = detect_joint_conflict(
        graph,
        (Proposal(coordinate, BondToken.TRIPLE),),
        capacity,
    )
    assert not conflict.occurred


def test_vadgm_dynamic_gate_preserves_capacity_with_one_forward_per_bin() -> None:
    vocabulary = _vocabulary()
    capacity = CapacityEngine(vocabulary)
    plan = _manual_plan(3, same_bin=True)
    result = sample_vadgm(
        FixedBase(vocabulary.clean_size),
        ZeroGuidance(vocabulary.clean_size),
        vocabulary,
        capacity,
        plan,
        target=1.0,
        category_generator=torch.Generator().manual_seed(2),
        diagnostic_generator=torch.Generator().manual_seed(3),
    )
    assert result.graph.is_clean(vocabulary.clean_size)
    assert capacity.clean_graph_is_capacity_safe(result.graph)
    assert result.diagnostics.nonempty_bins == 1
    assert result.diagnostics.base_forwards == 1
    assert result.diagnostics.guidance_forwards == 1
    assert result.diagnostics.path_capacity_violations == 0
    assert any(conflict.occurred for conflict in result.diagnostics.joint_conflicts)


def test_all_nonfinite_logits_use_only_declared_safe_fallbacks() -> None:
    vocabulary = _vocabulary()
    capacity = CapacityEngine(vocabulary)
    plan = _manual_plan(3, same_bin=True)
    result = sample_vadgm(
        FixedBase(vocabulary.clean_size, nan=True),
        ZeroGuidance(vocabulary.clean_size),
        vocabulary,
        capacity,
        plan,
        target=0.0,
        category_generator=torch.Generator().manual_seed(4),
    )
    expected_events = 3 + 3
    assert result.diagnostics.all_nonfinite_fallbacks == expected_events
    assert result.diagnostics.atom_fallbacks == 3
    assert result.diagnostics.bond_fallbacks == 3
    assert all(token == BondToken.NO_BOND for token in result.graph.bond_ids)
    max_capacity = max(capacity.capacity(index) for index in range(vocabulary.clean_size))
    assert all(capacity.capacity(token) == max_capacity for token in result.graph.atom_ids)
    assert capacity.clean_graph_is_capacity_safe(result.graph)


def test_diagnostic_rng_does_not_change_generated_graph() -> None:
    vocabulary = _vocabulary()
    capacity = CapacityEngine(vocabulary)
    plan = _manual_plan(3, same_bin=True)
    without_diagnostic = sample_vadgm(
        FixedBase(vocabulary.clean_size),
        ZeroGuidance(vocabulary.clean_size),
        vocabulary,
        capacity,
        plan,
        target=1.0,
        category_generator=torch.Generator().manual_seed(7),
    )
    with_diagnostic = sample_vadgm(
        FixedBase(vocabulary.clean_size),
        ZeroGuidance(vocabulary.clean_size),
        vocabulary,
        capacity,
        plan,
        target=1.0,
        category_generator=torch.Generator().manual_seed(7),
        diagnostic_generator=torch.Generator().manual_seed(8),
    )
    assert with_diagnostic.graph == without_diagnostic.graph


def test_one_event_per_bin_refreshes_models_once_per_event() -> None:
    vocabulary = _vocabulary()
    plan = _manual_plan(2, same_bin=False)
    result = sample_vadgm(
        FixedBase(vocabulary.clean_size),
        ZeroGuidance(vocabulary.clean_size),
        vocabulary,
        CapacityEngine(vocabulary),
        plan,
        target=1.0,
        category_generator=torch.Generator().manual_seed(9),
    )
    assert result.diagnostics.nonempty_bins == len(plan.events)
    assert result.diagnostics.base_forwards == len(plan.events)
    assert result.diagnostics.guidance_forwards == len(plan.events)
    assert all(0.0 <= trace.diffusion_time < 1.0 for trace in result.diagnostics.event_trace)


def test_compact_result_serialization_keeps_scientific_diagnostics() -> None:
    vocabulary = _vocabulary()
    plan = _manual_plan(2, same_bin=True)
    result = sample_vadgm(
        FixedBase(vocabulary.clean_size),
        ZeroGuidance(vocabulary.clean_size),
        vocabulary,
        CapacityEngine(vocabulary),
        plan,
        target=1.0,
        category_generator=torch.Generator().manual_seed(15),
        diagnostic_generator=torch.Generator().manual_seed(16),
    )
    payload = result_to_dict(
        result,
        include_event_plan=False,
        include_event_trace=False,
    )
    assert "event_plan" not in payload
    assert "event_trace" not in payload["diagnostics"]
    assert payload["diagnostics"]["path_capacity_violations"] == 0
    assert "joint_conflicts" in payload["diagnostics"]


def test_batched_vadgm_size_one_matches_serial_sampler() -> None:
    vocabulary = _vocabulary()
    capacity = CapacityEngine(vocabulary)
    plan = _manual_plan(3, same_bin=True)
    serial = sample_vadgm(
        FixedBase(vocabulary.clean_size),
        ZeroGuidance(vocabulary.clean_size),
        vocabulary,
        capacity,
        plan,
        target=1.0,
        category_generator=torch.Generator().manual_seed(31),
        diagnostic_generator=torch.Generator().manual_seed(32),
    )
    batched, execution = sample_vadgm_batch(
        FixedBase(vocabulary.clean_size),
        ZeroGuidance(vocabulary.clean_size),
        vocabulary,
        capacity,
        [plan],
        target=1.0,
        category_generator=torch.Generator().manual_seed(31),
        diagnostic_generator=torch.Generator().manual_seed(32),
    )
    assert batched == [serial]
    assert execution.samples == 1
    assert execution.neural_batch_forwards == serial.diagnostics.nonempty_bins


def test_batched_vadgm_preserves_each_path_and_batches_frontiers() -> None:
    vocabulary = _vocabulary()
    capacity = CapacityEngine(vocabulary)
    plans = [
        EventPlan.sample(3, 0.25, generator=torch.Generator().manual_seed(seed))
        for seed in (41, 42, 43, 44)
    ]
    results, execution = sample_vadgm_batch(
        FixedBase(vocabulary.clean_size),
        ZeroGuidance(vocabulary.clean_size),
        vocabulary,
        capacity,
        plans,
        target=1.0,
        category_generator=torch.Generator().manual_seed(45),
        diagnostic_generator=torch.Generator().manual_seed(46),
        record_trace=False,
    )
    assert len(results) == len(plans)
    assert all(capacity.clean_graph_is_capacity_safe(result.graph) for result in results)
    assert all(result.diagnostics.path_capacity_violations == 0 for result in results)
    assert all(not result.diagnostics.event_trace for result in results)
    assert execution.max_active_batch == len(plans)
    assert execution.neural_batch_forwards < sum(
        result.diagnostics.base_forwards for result in results
    )


def test_terminal_reachability_sampler_records_and_completes_valence() -> None:
    vocabulary = _vocabulary()
    capacity = CapacityEngine(vocabulary)
    result = sample_vadgm(
        FixedBase(vocabulary.clean_size),
        ZeroGuidance(vocabulary.clean_size),
        vocabulary,
        capacity,
        _manual_plan(3, same_bin=True),
        target=1.0,
        category_generator=torch.Generator().manual_seed(51),
        enforce_terminal_reachability=True,
        relax_terminal_gate=False,
    )

    assert result.diagnostics.terminal_reachability_enabled
    assert result.diagnostics.terminal_gate_relaxations == 0
    assert result.diagnostics.final_terminal_complete
    assert capacity.clean_graph_is_terminal_complete(result.graph)
