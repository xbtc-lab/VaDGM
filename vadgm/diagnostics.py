"""Event plans and sampled-proposal joint-conflict diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Mapping, Sequence

import torch

from vadgm.chemistry import CapacityEngine, TokenGraph, upper_triangle_pairs


CoordinateKind = Literal["atom", "bond"]


@dataclass(frozen=True)
class Coordinate:
    kind: CoordinateKind
    index: int
    endpoints: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Coordinate index cannot be negative")
        if self.kind == "atom" and self.endpoints is not None:
            raise ValueError("Atom coordinates do not have edge endpoints")
        if self.kind == "bond" and self.endpoints is None:
            raise ValueError("Bond coordinates require endpoints")


@dataclass(frozen=True)
class Event:
    coordinate: Coordinate
    clock: float
    bin_index: int


@dataclass(frozen=True)
class EventPlan:
    n_atoms: int
    delta_s: float
    events: tuple[Event, ...]

    def __post_init__(self) -> None:
        if self.n_atoms <= 0 or self.delta_s <= 0:
            raise ValueError("Event plan requires positive n_atoms and delta_s")
        expected = set(all_coordinates(self.n_atoms))
        observed = [event.coordinate for event in self.events]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError("Event plan must contain every graph coordinate exactly once")
        clocks = [event.clock for event in self.events]
        if any(not math.isfinite(value) or value < 0 for value in clocks):
            raise ValueError("Event clocks must be finite and non-negative")
        if len(set(clocks)) != len(clocks):
            raise ValueError("Finite-precision event clocks must be unique")
        if tuple(sorted(self.events, key=lambda event: event.clock)) != self.events:
            raise ValueError("Events must be ordered by continuous clock")
        if any(event.bin_index != math.floor(event.clock / self.delta_s) for event in self.events):
            raise ValueError("Event bin does not match floor(clock / delta_s)")

    @classmethod
    def sample(
        cls,
        n_atoms: int,
        delta_s: float,
        *,
        generator: torch.Generator,
    ) -> "EventPlan":
        coordinates = all_coordinates(n_atoms)
        # Resample the whole vector on the vanishingly rare finite-precision tie.
        for _ in range(10):
            uniforms = torch.rand(
                len(coordinates),
                dtype=torch.float64,
                generator=generator,
            ).clamp_min(torch.finfo(torch.float64).tiny)
            clocks = (-torch.log(uniforms)).tolist()
            if len(set(clocks)) == len(clocks):
                break
        else:
            raise RuntimeError("Could not draw unique event clocks")
        events = tuple(
            sorted(
                (
                    Event(
                        coordinate=coordinate,
                        clock=float(clock),
                        bin_index=math.floor(float(clock) / delta_s),
                    )
                    for coordinate, clock in zip(coordinates, clocks)
                ),
                key=lambda event: event.clock,
            )
        )
        return cls(n_atoms=n_atoms, delta_s=delta_s, events=events)

    def nonempty_bins(self) -> tuple[tuple[Event, ...], ...]:
        grouped: list[list[Event]] = []
        current_bin: int | None = None
        for event in self.events:
            if event.bin_index != current_bin:
                grouped.append([])
                current_bin = event.bin_index
            grouped[-1].append(event)
        return tuple(tuple(group) for group in grouped)

    def with_terminal_atom_phase(self) -> "EventPlan":
        """将 atom 坐标移到所有 bond 之后的同一终态时间箱。

        bond 的原始指数时钟与顺序保持不变；atom 的原始随机顺序也保持不变。
        该变换是显式的终态调度实验，不再声称所有坐标时钟 iid。
        """

        bonds = [event for event in self.events if event.coordinate.kind == "bond"]
        atoms = [event for event in self.events if event.coordinate.kind == "atom"]
        maximum_clock = max((event.clock for event in bonds), default=0.0)
        terminal_bin = math.floor(maximum_clock / self.delta_s) + 1
        bin_start = terminal_bin * self.delta_s
        # 把全部 atom 放入同一新箱，并用小偏移保留原随机顺序与唯一时钟。
        atom_events = [
            Event(
                coordinate=event.coordinate,
                clock=bin_start + self.delta_s * 0.5 * (rank + 1) / (len(atoms) + 1),
                bin_index=terminal_bin,
            )
            for rank, event in enumerate(atoms)
        ]
        events = tuple(sorted((*bonds, *atom_events), key=lambda event: event.clock))
        return EventPlan(self.n_atoms, self.delta_s, events)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_atoms": self.n_atoms,
            "delta_s": self.delta_s,
            "events": [asdict(event) for event in self.events],
        }


@dataclass(frozen=True)
class Proposal:
    coordinate: Coordinate
    token_id: int


@dataclass(frozen=True)
class JointConflict:
    occurred: bool
    conflict_type: str | None
    affected_atoms: tuple[int, ...]
    excess_by_atom: tuple[tuple[int, int], ...]
    proposal_count: int

    @property
    def mean_excess_capacity(self) -> float:
        if not self.excess_by_atom:
            return 0.0
        return sum(excess for _, excess in self.excess_by_atom) / len(self.excess_by_atom)


def all_coordinates(n_atoms: int) -> tuple[Coordinate, ...]:
    atoms = tuple(Coordinate("atom", index) for index in range(n_atoms))
    bonds = tuple(
        Coordinate("bond", edge_index, pair)
        for edge_index, pair in enumerate(upper_triangle_pairs(n_atoms))
    )
    return atoms + bonds


def apply_proposals(graph: TokenGraph, proposals: Iterable[Proposal]) -> TokenGraph:
    atoms = list(graph.atom_ids)
    bonds = list(graph.bond_ids)
    seen: set[Coordinate] = set()
    for proposal in proposals:
        if proposal.coordinate in seen:
            raise ValueError("A coordinate cannot have multiple joint proposals")
        seen.add(proposal.coordinate)
        if proposal.coordinate.kind == "atom":
            atoms[proposal.coordinate.index] = proposal.token_id
        else:
            bonds[proposal.coordinate.index] = proposal.token_id
    return TokenGraph(tuple(atoms), tuple(bonds), graph.smiles)


def proposal_is_individually_feasible(
    graph: TokenGraph,
    proposal: Proposal,
    capacity: CapacityEngine,
) -> bool:
    coordinate = proposal.coordinate
    if coordinate.kind == "atom":
        gate = capacity.atom_gate(graph, coordinate.index)
    else:
        gate = capacity.bond_gate(graph, coordinate.index)
    return 0 <= proposal.token_id < gate.numel() and bool(gate[proposal.token_id])


def detect_joint_conflict(
    bin_start_graph: TokenGraph,
    proposals: Sequence[Proposal],
    capacity: CapacityEngine,
) -> JointConflict:
    """Detect coordinate-wise feasible proposals whose union breaks capacity."""

    if not proposals:
        return JointConflict(False, None, (), (), 0)
    if not all(
        proposal_is_individually_feasible(bin_start_graph, proposal, capacity)
        for proposal in proposals
    ):
        raise ValueError("Joint-conflict inputs must be individually feasible proposals")
    combined = apply_proposals(bin_start_graph, proposals)
    if capacity.invariant_holds(combined):
        return JointConflict(False, None, (), (), len(proposals))

    loads = capacity.loads(combined)
    excess: list[tuple[int, int]] = []
    affected: list[int] = []
    for atom_index, load in enumerate(loads):
        feasible = capacity.feasible_atom_ids(combined, atom_index, loads)
        if feasible:
            maximum = max(capacity.capacity(atom_id) for atom_id in feasible)
        else:
            maximum = max(
                capacity.capacity(atom_id)
                for atom_id in range(capacity.atom_vocabulary.clean_size)
            )
        if load > maximum or not feasible:
            affected.append(atom_index)
            excess.append((atom_index, max(load - maximum, 1)))

    atom_proposals = {
        proposal.coordinate.index
        for proposal in proposals
        if proposal.coordinate.kind == "atom"
    }
    bond_proposals = [
        proposal for proposal in proposals if proposal.coordinate.kind == "bond"
    ]
    atom_bond = any(
        atom in atom_proposals
        and proposal.coordinate.endpoints is not None
        and atom in proposal.coordinate.endpoints
        for atom in affected
        for proposal in bond_proposals
    )
    incident_counts = {
        atom: sum(
            proposal.coordinate.endpoints is not None
            and atom in proposal.coordinate.endpoints
            for proposal in bond_proposals
        )
        for atom in affected
    }
    bond_bond = any(count >= 2 for count in incident_counts.values())
    if atom_bond and bond_bond:
        conflict_type = "multi-way"
    elif atom_bond:
        conflict_type = "atom-bond"
    elif bond_bond:
        conflict_type = "bond-bond"
    else:
        conflict_type = "other"
    return JointConflict(
        True,
        conflict_type,
        tuple(affected),
        tuple(excess),
        len(proposals),
    )


def summarize_conflicts(conflicts: Sequence[JointConflict]) -> dict[str, float | int]:
    total = len(conflicts)
    occurred = [conflict for conflict in conflicts if conflict.occurred]
    return {
        "bins": total,
        "conflict_bins": len(occurred),
        "conflict_bin_rate": len(occurred) / total if total else 0.0,
        "mean_excess_capacity": (
            sum(conflict.mean_excess_capacity for conflict in occurred) / len(occurred)
            if occurred
            else 0.0
        ),
    }
