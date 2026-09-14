"""Reference finite-bin VaDGM sampler with dynamic capacity gating."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import torch

from vadgm.base_model import BaseGenerator
from vadgm.chemistry import AtomVocabulary, BondToken, CapacityEngine, TokenGraph
from vadgm.diagnostics import (
    Coordinate,
    Event,
    EventPlan,
    JointConflict,
    Proposal,
    detect_joint_conflict,
)
from vadgm.diffusion import LinearRevealSchedule
from vadgm.guidance_model import (
    GuidanceNetwork,
    combine_dgm_log_scores,
    model_state_sha256,
)


@dataclass(frozen=True)
class EventTrace:
    bin_index: int
    clock: float
    diffusion_time: float
    coordinate_kind: str
    coordinate_index: int
    token_id: int
    feasible_categories: int
    finite_feasible_categories: int
    gate_deleted_categories: int
    terminal_gate_deleted_categories: int
    terminal_gate_relaxed: bool
    nonfinite_feasible_logits: int
    used_fallback: bool
    load_before: tuple[int, ...]
    load_after: tuple[int, ...]


@dataclass(frozen=True)
class VaDGMDiagnostics:
    nonempty_bins: int
    base_forwards: int
    guidance_forwards: int
    events: int
    gate_deleted_categories: int
    terminal_reachability_enabled: bool
    terminal_gate_deleted_categories: int
    terminal_gate_relaxations: int
    nonfinite_feasible_logits: int
    all_nonfinite_fallbacks: int
    atom_fallbacks: int
    bond_fallbacks: int
    forced_max_capacity_atoms: int
    path_capacity_violations: int
    final_capacity_safe: bool
    final_terminal_complete: bool
    event_trace: tuple[EventTrace, ...]
    joint_conflicts: tuple[JointConflict, ...]


@dataclass(frozen=True)
class VaDGMResult:
    graph: TokenGraph
    event_plan: EventPlan
    diagnostics: VaDGMDiagnostics


@dataclass(frozen=True)
class VaDGMBatchDiagnostics:
    samples: int
    neural_batch_forwards: int
    max_active_batch: int


@dataclass
class _VaDGMState:
    event_plan: EventPlan
    graph: TokenGraph
    nonempty_bins: tuple[tuple[Event, ...], ...]
    next_bin: int = 0
    traces: list[EventTrace] = field(default_factory=list)
    conflicts: list[JointConflict] = field(default_factory=list)
    gate_deleted_total: int = 0
    terminal_gate_deleted_total: int = 0
    terminal_gate_relaxations: int = 0
    nonfinite_total: int = 0
    all_nonfinite: int = 0
    atom_fallbacks: int = 0
    bond_fallbacks: int = 0
    forced_atoms: int = 0
    path_violations: int = 0


@dataclass(frozen=True)
class _GateDecision:
    mask: torch.Tensor
    terminal_deleted: int = 0
    terminal_relaxed: bool = False


def _coordinate_gate(
    graph: TokenGraph,
    coordinate: Coordinate,
    capacity: CapacityEngine,
    *,
    enforce_terminal_reachability: bool = False,
    relax_terminal_gate: bool = True,
    loads: Sequence[int] | None = None,
    masked_incident: Sequence[int] | None = None,
) -> _GateDecision:
    if coordinate.kind == "atom":
        capacity_mask = capacity.atom_gate(graph, coordinate.index, loads)
    else:
        capacity_mask = capacity.bond_gate(graph, coordinate.index, loads)
    if not enforce_terminal_reachability:
        return _GateDecision(capacity_mask)
    if coordinate.kind == "atom":
        terminal_mask = capacity.atom_terminal_gate(
            graph,
            coordinate.index,
            loads=loads,
            masked_incident=masked_incident,
        )
    else:
        terminal_mask = capacity.bond_terminal_gate(
            graph,
            coordinate.index,
            loads=loads,
            masked_incident=masked_incident,
        )
    combined = capacity_mask & terminal_mask
    terminal_deleted = int((capacity_mask & ~terminal_mask).sum().item())
    if bool(combined.any()):
        return _GateDecision(combined, terminal_deleted, False)
    if relax_terminal_gate and bool(capacity_mask.any()):
        return _GateDecision(capacity_mask, terminal_deleted, True)
    raise RuntimeError(
        "Terminal reachability gate removed every capacity-feasible category"
    )


def _fallback_token(
    graph: TokenGraph,
    coordinate: Coordinate,
    capacity: CapacityEngine,
    gate: torch.Tensor | None = None,
) -> tuple[int, bool]:
    active_gate = gate
    if coordinate.kind == "bond":
        if active_gate is None:
            active_gate = capacity.bond_gate(graph, coordinate.index)
        feasible_bonds = [
            bond_id
            for bond_id in range(BondToken.CLEAN_SIZE)
            if bool(active_gate[bond_id])
        ]
        if not feasible_bonds:
            raise RuntimeError("Bond fallback encountered an empty feasible set")
        # 优先选择 no-bond；若终态 gate 要求补足负载，则选择最小可行键级。
        token = (
            BondToken.NO_BOND
            if BondToken.NO_BOND in feasible_bonds
            else min(feasible_bonds, key=lambda item: BondToken.ORDERS[item])
        )
        return token, False
    feasible = capacity.feasible_atom_ids(graph, coordinate.index)
    if active_gate is not None:
        feasible = tuple(atom_id for atom_id in feasible if bool(active_gate[atom_id]))
    if not feasible:
        raise RuntimeError("Atom fallback encountered an empty feasible set")
    maximum = max(capacity.capacity(atom_id) for atom_id in feasible)
    candidates = [
        atom_id for atom_id in feasible if capacity.capacity(atom_id) == maximum
    ]
    # Atom IDs follow the frozen lexicographic token order, so this tie break is
    # deterministic and independent of the coordinate/node index.
    return min(candidates), True


def _sample_finite_feasible(
    log_scores: torch.Tensor,
    gate: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[int | None, int, int, int]:
    gate = gate.to(device=log_scores.device)
    finite_feasible = gate & torch.isfinite(log_scores)
    feasible_count = int(gate.sum().item())
    finite_count = int(finite_feasible.sum().item())
    nonfinite_count = int((gate & ~torch.isfinite(log_scores)).sum().item())
    gate_deleted = int((~gate).sum().item())
    if finite_count == 0:
        return None, feasible_count, nonfinite_count, gate_deleted
    masked = log_scores.masked_fill(~finite_feasible, -torch.inf)
    log_normalizer = torch.logsumexp(masked, dim=-1)
    probabilities = torch.exp(masked - log_normalizer)
    token = int(torch.multinomial(probabilities, 1, generator=generator).item())
    return token, feasible_count, nonfinite_count, gate_deleted


def sample_static_gated_proposals(
    bin_start_graph: TokenGraph,
    events: Sequence[Event],
    cached_scores: Mapping[Coordinate, torch.Tensor],
    capacity: CapacityEngine,
    *,
    generator: torch.Generator,
    enforce_terminal_reachability: bool = False,
    relax_terminal_gate: bool = True,
) -> tuple[Proposal, ...]:
    """Draw actual stale local-gated proposals without changing carrier state."""

    proposals = []
    loads = capacity.loads(bin_start_graph)
    masked_incident = (
        capacity.masked_incident_counts(bin_start_graph)
        if enforce_terminal_reachability
        else None
    )
    for event in events:
        coordinate = event.coordinate
        decision = _coordinate_gate(
            bin_start_graph,
            coordinate,
            capacity,
            enforce_terminal_reachability=enforce_terminal_reachability,
            relax_terminal_gate=relax_terminal_gate,
            loads=loads,
            masked_incident=masked_incident,
        )
        token, _, _, _ = _sample_finite_feasible(
            cached_scores[coordinate],
            decision.mask,
            generator=generator,
        )
        if token is None:
            token, _ = _fallback_token(
                bin_start_graph,
                coordinate,
                capacity,
                decision.mask,
            )
        proposals.append(Proposal(coordinate, token))
    return tuple(proposals)


def _model_inputs(
    graph: TokenGraph,
    vocabulary: AtomVocabulary,
    diffusion_time: float,
    target: float,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    atom_ids = torch.tensor([graph.atom_ids], dtype=torch.long, device=device)
    bond_ids = torch.tensor([graph.bond_ids], dtype=torch.long, device=device)
    return {
        "atom_ids": atom_ids,
        "bond_ids": bond_ids,
        "atom_valid_mask": torch.ones_like(atom_ids, dtype=torch.bool),
        "bond_valid_mask": torch.ones_like(bond_ids, dtype=torch.bool),
        "time": torch.tensor([diffusion_time], dtype=torch.float32, device=device),
        "target": torch.tensor([target], dtype=torch.float32, device=device),
        "n_atoms": torch.tensor([graph.n_atoms], dtype=torch.long, device=device),
    }


def _batched_model_inputs(
    graphs: Sequence[TokenGraph],
    vocabulary: AtomVocabulary,
    diffusion_times: Sequence[float],
    target: float,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Pad partial graphs for one neural forward over several hazard bins."""

    if not graphs or len(graphs) != len(diffusion_times):
        raise ValueError("graphs and diffusion_times require the same positive length")
    batch_size = len(graphs)
    max_atoms = max(graph.n_atoms for graph in graphs)
    max_bonds = max_atoms * (max_atoms - 1) // 2
    atom_ids = torch.full(
        (batch_size, max_atoms),
        vocabulary.pad_id,
        dtype=torch.long,
        device=device,
    )
    bond_ids = torch.full(
        (batch_size, max_bonds),
        BondToken.PAD,
        dtype=torch.long,
        device=device,
    )
    atom_valid = torch.zeros_like(atom_ids, dtype=torch.bool)
    bond_valid = torch.zeros_like(bond_ids, dtype=torch.bool)
    for row, graph in enumerate(graphs):
        atom_count = graph.n_atoms
        bond_count = len(graph.bond_ids)
        atom_ids[row, :atom_count] = torch.tensor(
            graph.atom_ids,
            dtype=torch.long,
            device=device,
        )
        bond_ids[row, :bond_count] = torch.tensor(
            graph.bond_ids,
            dtype=torch.long,
            device=device,
        )
        atom_valid[row, :atom_count] = True
        bond_valid[row, :bond_count] = True
    return {
        "atom_ids": atom_ids,
        "bond_ids": bond_ids,
        "atom_valid_mask": atom_valid,
        "bond_valid_mask": bond_valid,
        "time": torch.tensor(diffusion_times, dtype=torch.float32, device=device),
        "target": torch.full(
            (batch_size,),
            float(target),
            dtype=torch.float32,
            device=device,
        ),
        "n_atoms": torch.tensor(
            [graph.n_atoms for graph in graphs],
            dtype=torch.long,
            device=device,
        ),
    }


def _commit_cached_bin(
    state: _VaDGMState,
    events: Sequence[Event],
    cached_scores: Mapping[Coordinate, torch.Tensor],
    capacity: CapacityEngine,
    *,
    category_generator: torch.Generator,
    diagnostic_generator: torch.Generator | None,
    diffusion_time: float,
    record_trace: bool,
    enforce_terminal_reachability: bool,
    relax_terminal_gate: bool,
) -> None:
    """Apply one stale-score bin while refreshing the gate after every event."""

    bin_start_graph = state.graph
    if diagnostic_generator is not None:
        shadow = sample_static_gated_proposals(
            bin_start_graph,
            events,
            cached_scores,
            capacity,
            generator=diagnostic_generator,
            # joint-conflict 始终以原始 capacity-only 静态 gate 定义，
            # 避免启用终态约束后悄然改变 Figure 1 的诊断口径。
            enforce_terminal_reachability=False,
        )
        state.conflicts.append(
            detect_joint_conflict(bin_start_graph, shadow, capacity)
        )

    for event in events:
        coordinate = event.coordinate
        loads_before = capacity.loads(state.graph)
        masked_incident = (
            capacity.masked_incident_counts(state.graph)
            if enforce_terminal_reachability
            else None
        )
        decision = _coordinate_gate(
            state.graph,
            coordinate,
            capacity,
            enforce_terminal_reachability=enforce_terminal_reachability,
            relax_terminal_gate=relax_terminal_gate,
            loads=loads_before,
            masked_incident=masked_incident,
        )
        token, feasible_count, nonfinite_count, gate_deleted = _sample_finite_feasible(
            cached_scores[coordinate],
            decision.mask,
            generator=category_generator,
        )
        used_fallback = token is None
        if used_fallback:
            token, forced = _fallback_token(
                state.graph,
                coordinate,
                capacity,
                decision.mask,
            )
            state.all_nonfinite += 1
            if coordinate.kind == "atom":
                state.atom_fallbacks += 1
                state.forced_atoms += int(forced)
            else:
                state.bond_fallbacks += 1
        if coordinate.kind == "atom":
            if state.graph.atom_ids[coordinate.index] != capacity.atom_vocabulary.mask_id:
                raise RuntimeError("Atom coordinate was committed more than once")
            state.graph = state.graph.with_atom(coordinate.index, int(token))
        else:
            if state.graph.bond_ids[coordinate.index] != BondToken.MASK:
                raise RuntimeError("Bond coordinate was committed more than once")
            state.graph = state.graph.with_bond(coordinate.index, int(token))
        state.gate_deleted_total += gate_deleted
        state.terminal_gate_deleted_total += decision.terminal_deleted
        state.terminal_gate_relaxations += int(decision.terminal_relaxed)
        state.nonfinite_total += nonfinite_count
        if not capacity.invariant_holds(state.graph):
            state.path_violations += 1
            raise RuntimeError("VaDGM dynamic commit violated the path invariant")
        if record_trace:
            state.traces.append(
                EventTrace(
                    bin_index=event.bin_index,
                    clock=event.clock,
                    diffusion_time=diffusion_time,
                    coordinate_kind=coordinate.kind,
                    coordinate_index=coordinate.index,
                    token_id=int(token),
                    feasible_categories=feasible_count,
                    finite_feasible_categories=feasible_count - nonfinite_count,
                    gate_deleted_categories=gate_deleted,
                    terminal_gate_deleted_categories=decision.terminal_deleted,
                    terminal_gate_relaxed=decision.terminal_relaxed,
                    nonfinite_feasible_logits=nonfinite_count,
                    used_fallback=used_fallback,
                    load_before=loads_before,
                    load_after=capacity.loads(state.graph),
                )
            )


@torch.no_grad()
def sample_vadgm(
    base: BaseGenerator,
    guidance: GuidanceNetwork,
    vocabulary: AtomVocabulary,
    capacity: CapacityEngine,
    event_plan: EventPlan,
    *,
    target: float,
    category_generator: torch.Generator,
    diagnostic_generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    record_trace: bool = True,
    verify_model_state: bool = True,
    enforce_terminal_reachability: bool = False,
    relax_terminal_gate: bool = True,
) -> VaDGMResult:
    """Sample one graph using one neural forward pair per non-empty hazard bin."""

    graph = TokenGraph.all_mask(event_plan.n_atoms, vocabulary.mask_id)
    if not capacity.invariant_holds(graph):
        raise RuntimeError("All-MASK initial graph violates the capacity invariant")
    base_before = model_state_sha256(base) if verify_model_state else None
    guidance_before = model_state_sha256(guidance) if verify_model_state else None
    base.eval()
    guidance.eval()
    schedule = LinearRevealSchedule()
    traces: list[EventTrace] = []
    conflicts: list[JointConflict] = []
    gate_deleted_total = 0
    terminal_gate_deleted_total = 0
    terminal_gate_relaxations = 0
    nonfinite_total = 0
    all_nonfinite = 0
    atom_fallbacks = 0
    bond_fallbacks = 0
    forced_atoms = 0
    path_violations = 0
    nonempty_bins = event_plan.nonempty_bins()

    for events in nonempty_bins:
        bin_start_graph = graph
        first_clock = events[0].clock
        diffusion_time = float(
            schedule.inverse_hazard(torch.tensor(first_clock, dtype=torch.float64)).item()
        )
        inputs = _model_inputs(
            bin_start_graph,
            vocabulary,
            diffusion_time,
            target,
            device,
        )
        base_atom, base_bond = base(
            inputs["atom_ids"],
            inputs["bond_ids"],
            inputs["atom_valid_mask"],
            inputs["bond_valid_mask"],
            inputs["time"],
            inputs["n_atoms"],
        )
        atom_u, bond_u = guidance(
            inputs["atom_ids"],
            inputs["bond_ids"],
            inputs["atom_valid_mask"],
            inputs["bond_valid_mask"],
            inputs["time"],
            inputs["target"],
            inputs["n_atoms"],
        )
        atom_scores, bond_scores = combine_dgm_log_scores(
            base_atom,
            base_bond,
            atom_u,
            bond_u,
            u_min=guidance.config.u_min,
            u_max=guidance.config.u_max,
        )
        cached_scores: dict[Coordinate, torch.Tensor] = {}
        for event in events:
            coordinate = event.coordinate
            if coordinate.kind == "atom":
                cached_scores[coordinate] = atom_scores[0, coordinate.index].clone()
            else:
                cached_scores[coordinate] = bond_scores[0, coordinate.index].clone()

        if diagnostic_generator is not None:
            shadow = sample_static_gated_proposals(
                bin_start_graph,
                events,
                cached_scores,
                capacity,
                generator=diagnostic_generator,
                enforce_terminal_reachability=False,
            )
            conflicts.append(detect_joint_conflict(bin_start_graph, shadow, capacity))

        for event in events:
            coordinate = event.coordinate
            loads_before = capacity.loads(graph)
            masked_incident = (
                capacity.masked_incident_counts(graph)
                if enforce_terminal_reachability
                else None
            )
            decision = _coordinate_gate(
                graph,
                coordinate,
                capacity,
                enforce_terminal_reachability=enforce_terminal_reachability,
                relax_terminal_gate=relax_terminal_gate,
                loads=loads_before,
                masked_incident=masked_incident,
            )
            token, feasible_count, nonfinite_count, gate_deleted = _sample_finite_feasible(
                cached_scores[coordinate],
                decision.mask,
                generator=category_generator,
            )
            used_fallback = token is None
            if used_fallback:
                token, forced = _fallback_token(
                    graph,
                    coordinate,
                    capacity,
                    decision.mask,
                )
                all_nonfinite += 1
                if coordinate.kind == "atom":
                    atom_fallbacks += 1
                    forced_atoms += int(forced)
                else:
                    bond_fallbacks += 1
            if coordinate.kind == "atom":
                if graph.atom_ids[coordinate.index] != vocabulary.mask_id:
                    raise RuntimeError("Atom coordinate was committed more than once")
                graph = graph.with_atom(coordinate.index, int(token))
            else:
                if graph.bond_ids[coordinate.index] != BondToken.MASK:
                    raise RuntimeError("Bond coordinate was committed more than once")
                graph = graph.with_bond(coordinate.index, int(token))
            gate_deleted_total += gate_deleted
            terminal_gate_deleted_total += decision.terminal_deleted
            terminal_gate_relaxations += int(decision.terminal_relaxed)
            nonfinite_total += nonfinite_count
            if not capacity.invariant_holds(graph):
                path_violations += 1
                raise RuntimeError("VaDGM dynamic commit violated the path invariant")
            if record_trace:
                traces.append(
                    EventTrace(
                        bin_index=event.bin_index,
                        clock=event.clock,
                        diffusion_time=diffusion_time,
                        coordinate_kind=coordinate.kind,
                        coordinate_index=coordinate.index,
                        token_id=int(token),
                        feasible_categories=feasible_count,
                        finite_feasible_categories=feasible_count - nonfinite_count,
                        gate_deleted_categories=gate_deleted,
                        terminal_gate_deleted_categories=decision.terminal_deleted,
                        terminal_gate_relaxed=decision.terminal_relaxed,
                        nonfinite_feasible_logits=nonfinite_count,
                        used_fallback=used_fallback,
                        load_before=loads_before,
                        load_after=capacity.loads(graph),
                    )
                )

    if not graph.is_clean(vocabulary.clean_size):
        raise RuntimeError("VaDGM ended before all coordinates were revealed")
    final_safe = capacity.clean_graph_is_capacity_safe(graph)
    final_terminal_complete = capacity.clean_graph_is_terminal_complete(graph)
    if not final_safe:
        raise RuntimeError("VaDGM produced a final encoded-capacity violation")
    if verify_model_state and base_before != model_state_sha256(base):
        raise RuntimeError("Frozen base parameters changed during VaDGM sampling")
    if verify_model_state and guidance_before != model_state_sha256(guidance):
        raise RuntimeError("Frozen guidance parameters changed during VaDGM sampling")
    diagnostics = VaDGMDiagnostics(
        nonempty_bins=len(nonempty_bins),
        base_forwards=len(nonempty_bins),
        guidance_forwards=len(nonempty_bins),
        events=len(event_plan.events),
        gate_deleted_categories=gate_deleted_total,
        terminal_reachability_enabled=enforce_terminal_reachability,
        terminal_gate_deleted_categories=terminal_gate_deleted_total,
        terminal_gate_relaxations=terminal_gate_relaxations,
        nonfinite_feasible_logits=nonfinite_total,
        all_nonfinite_fallbacks=all_nonfinite,
        atom_fallbacks=atom_fallbacks,
        bond_fallbacks=bond_fallbacks,
        forced_max_capacity_atoms=forced_atoms,
        path_capacity_violations=path_violations,
        final_capacity_safe=final_safe,
        final_terminal_complete=final_terminal_complete,
        event_trace=tuple(traces),
        joint_conflicts=tuple(conflicts),
    )
    return VaDGMResult(graph, event_plan, diagnostics)


@torch.no_grad()
def sample_vadgm_batch(
    base: BaseGenerator,
    guidance: GuidanceNetwork,
    vocabulary: AtomVocabulary,
    capacity: CapacityEngine,
    event_plans: Sequence[EventPlan],
    *,
    target: float,
    category_generator: torch.Generator,
    diagnostic_generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    record_trace: bool = True,
    verify_model_state: bool = True,
    enforce_terminal_reachability: bool = False,
    relax_terminal_gate: bool = True,
) -> tuple[list[VaDGMResult], VaDGMBatchDiagnostics]:
    """Sample several molecules by batching one frontier bin per molecule.

    Neural scores are computed together on ``device`` and transferred once per
    frontier round to CPU. Each molecule still commits its own bin in event
    clock order with a freshly evaluated dynamic capacity gate. The scheduling
    changes random-number consumption relative to the serial implementation,
    but not the VaDGM transition distribution or per-sample NFE definition.
    """

    if not event_plans:
        raise ValueError("event_plans must contain at least one plan")
    if category_generator.device.type != "cpu":
        raise ValueError("Batched VaDGM category_generator must be a CPU generator")
    if diagnostic_generator is not None and diagnostic_generator.device.type != "cpu":
        raise ValueError("Batched VaDGM diagnostic_generator must be a CPU generator")
    maximum_atoms = min(base.config.max_atoms, guidance.config.max_atoms)
    if max(plan.n_atoms for plan in event_plans) > maximum_atoms:
        raise ValueError("Requested graph exceeds a model maximum")

    states = [
        _VaDGMState(
            event_plan=plan,
            graph=TokenGraph.all_mask(plan.n_atoms, vocabulary.mask_id),
            nonempty_bins=plan.nonempty_bins(),
        )
        for plan in event_plans
    ]
    if any(not capacity.invariant_holds(state.graph) for state in states):
        raise RuntimeError("An all-MASK initial graph violates the capacity invariant")

    base_before = model_state_sha256(base) if verify_model_state else None
    guidance_before = model_state_sha256(guidance) if verify_model_state else None
    base.eval()
    guidance.eval()
    schedule = LinearRevealSchedule()
    neural_batch_forwards = 0
    max_active_batch = 0

    while True:
        active = [
            state
            for state in states
            if state.next_bin < len(state.nonempty_bins)
        ]
        if not active:
            break
        max_active_batch = max(max_active_batch, len(active))
        active_events = [state.nonempty_bins[state.next_bin] for state in active]
        first_clocks = torch.tensor(
            [events[0].clock for events in active_events],
            dtype=torch.float64,
        )
        diffusion_times = schedule.inverse_hazard(first_clocks).tolist()
        inputs = _batched_model_inputs(
            [state.graph for state in active],
            vocabulary,
            diffusion_times,
            target,
            device,
        )
        base_atom, base_bond = base(
            inputs["atom_ids"],
            inputs["bond_ids"],
            inputs["atom_valid_mask"],
            inputs["bond_valid_mask"],
            inputs["time"],
            inputs["n_atoms"],
        )
        atom_u, bond_u = guidance(
            inputs["atom_ids"],
            inputs["bond_ids"],
            inputs["atom_valid_mask"],
            inputs["bond_valid_mask"],
            inputs["time"],
            inputs["target"],
            inputs["n_atoms"],
        )
        atom_scores, bond_scores = combine_dgm_log_scores(
            base_atom,
            base_bond,
            atom_u,
            bond_u,
            u_min=guidance.config.u_min,
            u_max=guidance.config.u_max,
        )
        # Tiny category sampling and dynamic gates are faster on CPU than as
        # hundreds of synchronizing batch-size-one CUDA kernels.
        atom_scores_cpu = atom_scores.cpu()
        bond_scores_cpu = bond_scores.cpu()
        neural_batch_forwards += 1

        for row, (state, events, diffusion_time) in enumerate(
            zip(active, active_events, diffusion_times)
        ):
            cached_scores: dict[Coordinate, torch.Tensor] = {}
            for event in events:
                coordinate = event.coordinate
                if coordinate.kind == "atom":
                    cached_scores[coordinate] = atom_scores_cpu[
                        row, coordinate.index
                    ].clone()
                else:
                    cached_scores[coordinate] = bond_scores_cpu[
                        row, coordinate.index
                    ].clone()
            _commit_cached_bin(
                state,
                events,
                cached_scores,
                capacity,
                category_generator=category_generator,
                diagnostic_generator=diagnostic_generator,
                diffusion_time=float(diffusion_time),
                record_trace=record_trace,
                enforce_terminal_reachability=enforce_terminal_reachability,
                relax_terminal_gate=relax_terminal_gate,
            )
            state.next_bin += 1

    if verify_model_state and base_before != model_state_sha256(base):
        raise RuntimeError("Frozen base parameters changed during batched VaDGM sampling")
    if verify_model_state and guidance_before != model_state_sha256(guidance):
        raise RuntimeError(
            "Frozen guidance parameters changed during batched VaDGM sampling"
        )

    results: list[VaDGMResult] = []
    for state in states:
        if not state.graph.is_clean(vocabulary.clean_size):
            raise RuntimeError("Batched VaDGM ended before all coordinates were revealed")
        final_safe = capacity.clean_graph_is_capacity_safe(state.graph)
        final_terminal_complete = capacity.clean_graph_is_terminal_complete(state.graph)
        if not final_safe:
            raise RuntimeError("Batched VaDGM produced a final capacity violation")
        diagnostics = VaDGMDiagnostics(
            nonempty_bins=len(state.nonempty_bins),
            base_forwards=len(state.nonempty_bins),
            guidance_forwards=len(state.nonempty_bins),
            events=len(state.event_plan.events),
            gate_deleted_categories=state.gate_deleted_total,
            terminal_reachability_enabled=enforce_terminal_reachability,
            terminal_gate_deleted_categories=state.terminal_gate_deleted_total,
            terminal_gate_relaxations=state.terminal_gate_relaxations,
            nonfinite_feasible_logits=state.nonfinite_total,
            all_nonfinite_fallbacks=state.all_nonfinite,
            atom_fallbacks=state.atom_fallbacks,
            bond_fallbacks=state.bond_fallbacks,
            forced_max_capacity_atoms=state.forced_atoms,
            path_capacity_violations=state.path_violations,
            final_capacity_safe=final_safe,
            final_terminal_complete=final_terminal_complete,
            event_trace=tuple(state.traces),
            joint_conflicts=tuple(state.conflicts),
        )
        results.append(VaDGMResult(state.graph, state.event_plan, diagnostics))
    return results, VaDGMBatchDiagnostics(
        samples=len(results),
        neural_batch_forwards=neural_batch_forwards,
        max_active_batch=max_active_batch,
    )


def result_to_dict(
    result: VaDGMResult,
    *,
    include_event_plan: bool = True,
    include_event_trace: bool = True,
) -> dict[str, object]:
    """Serialize a result, optionally omitting large per-event debug records."""

    diagnostics = asdict(result.diagnostics)
    if not include_event_trace:
        diagnostics.pop("event_trace", None)
    payload: dict[str, object] = {
        "graph": {
            "atom_ids": list(result.graph.atom_ids),
            "bond_ids": list(result.graph.bond_ids),
        },
        "diagnostics": diagnostics,
    }
    if include_event_plan:
        payload["event_plan"] = result.event_plan.to_dict()
    return payload
