"""Molecular graph tokens, RDKit codec, and VaDGM capacity certificates.

The capacity engine deliberately implements only the encoded heavy-atom
upper-capacity constraint from the research specification.  It is not a
replacement for RDKit sanitization or a complete chemical-validity oracle.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import torch
from rdkit import Chem


@dataclass(frozen=True, order=True)
class AtomToken:
    atomic_number: int
    formal_charge: int
    total_hydrogens: int

    def __post_init__(self) -> None:
        if self.atomic_number <= 0:
            raise ValueError("atomic_number must be positive")
        if self.total_hydrogens < 0:
            raise ValueError("total_hydrogens cannot be negative")


class BondToken:
    """Frozen clean bond vocabulary used by every experiment."""

    NO_BOND = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    CLEAN_SIZE = 4
    MASK = 4
    PAD = 5

    ORDERS: Mapping[int, int] = {
        NO_BOND: 0,
        SINGLE: 1,
        DOUBLE: 2,
        TRIPLE: 3,
    }


_RDKIT_TO_BOND_ID = {
    Chem.BondType.SINGLE: BondToken.SINGLE,
    Chem.BondType.DOUBLE: BondToken.DOUBLE,
    Chem.BondType.TRIPLE: BondToken.TRIPLE,
}

_BOND_ID_TO_RDKIT = {
    BondToken.SINGLE: Chem.BondType.SINGLE,
    BondToken.DOUBLE: Chem.BondType.DOUBLE,
    BondToken.TRIPLE: Chem.BondType.TRIPLE,
}


def upper_triangle_pairs(n: int) -> tuple[tuple[int, int], ...]:
    if n < 0:
        raise ValueError("n must be non-negative")
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


@dataclass(frozen=True)
class TokenGraph:
    """A graph with each undirected bond stored exactly once (i < j)."""

    atom_ids: tuple[int, ...]
    bond_ids: tuple[int, ...]
    smiles: str | None = None

    def __post_init__(self) -> None:
        expected = self.n_atoms * (self.n_atoms - 1) // 2
        if len(self.bond_ids) != expected:
            raise ValueError(
                f"Expected {expected} upper-triangle bonds for n={self.n_atoms}, "
                f"received {len(self.bond_ids)}"
            )

    @property
    def n_atoms(self) -> int:
        return len(self.atom_ids)

    def is_clean(self, atom_clean_size: int) -> bool:
        return all(0 <= token < atom_clean_size for token in self.atom_ids) and all(
            token in BondToken.ORDERS for token in self.bond_ids
        )

    @classmethod
    def all_mask(cls, n_atoms: int, atom_mask_id: int) -> "TokenGraph":
        return cls(
            atom_ids=(atom_mask_id,) * n_atoms,
            bond_ids=(BondToken.MASK,) * (n_atoms * (n_atoms - 1) // 2),
        )

    def with_atom(self, index: int, token_id: int) -> "TokenGraph":
        values = list(self.atom_ids)
        values[index] = token_id
        return TokenGraph(tuple(values), self.bond_ids, self.smiles)

    def with_bond(self, edge_index: int, token_id: int) -> "TokenGraph":
        values = list(self.bond_ids)
        values[edge_index] = token_id
        return TokenGraph(self.atom_ids, tuple(values), self.smiles)


class AtomVocabulary:
    """Frozen clean atom vocabulary with distinct MASK and PAD identifiers."""

    def __init__(self, tokens: Sequence[AtomToken]) -> None:
        unique = tuple(sorted(set(tokens)))
        if not unique:
            raise ValueError("Atom vocabulary cannot be empty")
        if len(unique) != len(tokens):
            raise ValueError("Atom vocabulary tokens must be unique")
        self._tokens = unique
        self._token_to_id = {token: index for index, token in enumerate(unique)}

    @classmethod
    def build(cls, molecules: Iterable[Chem.Mol]) -> "AtomVocabulary":
        tokens: set[AtomToken] = set()
        for molecule in molecules:
            tokens.update(atom_token(atom) for atom in molecule.GetAtoms())
        return cls(sorted(tokens))

    @property
    def tokens(self) -> tuple[AtomToken, ...]:
        return self._tokens

    @property
    def clean_size(self) -> int:
        return len(self._tokens)

    @property
    def mask_id(self) -> int:
        return self.clean_size

    @property
    def pad_id(self) -> int:
        return self.clean_size + 1

    def encode(self, token: AtomToken) -> int:
        try:
            return self._token_to_id[token]
        except KeyError as exc:
            raise KeyError(f"Atom token is outside the frozen vocabulary: {token}") from exc

    def decode(self, token_id: int) -> AtomToken:
        if not 0 <= token_id < self.clean_size:
            raise ValueError(f"Expected a clean atom token ID, received {token_id}")
        return self._tokens[token_id]

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": [asdict(token) for token in self._tokens],
            "mask_id": self.mask_id,
            "pad_id": self.pad_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AtomVocabulary":
        raw_tokens = payload.get("tokens")
        if not isinstance(raw_tokens, list):
            raise ValueError("Vocabulary payload has no token list")
        tokens = [AtomToken(**item) for item in raw_tokens if isinstance(item, dict)]
        vocabulary = cls(tokens)
        if payload.get("mask_id") != vocabulary.mask_id:
            raise ValueError("Stored atom MASK ID is inconsistent with token list")
        if payload.get("pad_id") != vocabulary.pad_id:
            raise ValueError("Stored atom PAD ID is inconsistent with token list")
        return vocabulary

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AtomVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def atom_token(atom: Chem.Atom) -> AtomToken:
    return AtomToken(
        atomic_number=atom.GetAtomicNum(),
        formal_charge=atom.GetFormalCharge(),
        total_hydrogens=int(atom.GetTotalNumHs(includeNeighbors=True)),
    )


def canonicalize_molecule(molecule: Chem.Mol) -> tuple[str, Chem.Mol]:
    """Return a deterministic, heavy-atom, Kekulized molecule."""

    molecule = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
    Chem.SanitizeMol(molecule)
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )
    normalized = Chem.MolFromSmiles(canonical)
    if normalized is None:
        raise ValueError("RDKit failed to parse its canonical SMILES")
    normalized = Chem.RemoveHs(normalized, sanitize=True)
    Chem.Kekulize(normalized, clearAromaticFlags=True)
    kekule_smiles = Chem.MolToSmiles(
        normalized,
        canonical=True,
        kekuleSmiles=True,
        isomericSmiles=False,
    )
    return kekule_smiles, normalized


def parse_and_filter_smiles(
    smiles: str,
    *,
    allowed_atomic_numbers: set[int] | None = None,
    allowed_charge_states: set[tuple[int, int]] | None = None,
    max_heavy_atoms: int | None = None,
    require_single_component: bool = True,
    reject_radicals: bool = True,
) -> tuple[str, Chem.Mol]:
    stripped = smiles.strip()
    if not stripped:
        raise ValueError("SMILES is empty")
    molecule = Chem.MolFromSmiles(stripped)
    if molecule is None:
        raise ValueError(f"RDKit cannot parse SMILES: {stripped}")
    if require_single_component and len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("Only single-component molecules are supported")
    if reject_radicals and any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise ValueError("Radicals are outside the declared graph space")
    heavy_atoms = molecule.GetNumHeavyAtoms()
    if heavy_atoms == 0:
        raise ValueError("Molecule has no heavy atoms")
    if max_heavy_atoms is not None and heavy_atoms > max_heavy_atoms:
        raise ValueError(f"Molecule has {heavy_atoms} heavy atoms; maximum is {max_heavy_atoms}")
    if allowed_atomic_numbers is not None:
        observed = {atom.GetAtomicNum() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1}
        disallowed = observed - allowed_atomic_numbers
        if disallowed:
            raise ValueError(f"Disallowed atomic numbers: {sorted(disallowed)}")
    if allowed_charge_states is not None:
        observed_states = {
            (atom.GetAtomicNum(), atom.GetFormalCharge())
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() > 1
        }
        disallowed_states = observed_states - allowed_charge_states
        if disallowed_states:
            raise ValueError(
                f"Atom charge states have no declared valence protocol: {sorted(disallowed_states)}"
            )
    return canonicalize_molecule(molecule)


class GraphCodec:
    def __init__(self, atom_vocabulary: AtomVocabulary) -> None:
        self.atom_vocabulary = atom_vocabulary

    def encode_molecule(self, molecule: Chem.Mol, smiles: str | None = None) -> TokenGraph:
        canonical, normalized = canonicalize_molecule(molecule)
        atom_ids = tuple(
            self.atom_vocabulary.encode(atom_token(atom)) for atom in normalized.GetAtoms()
        )
        bond_lookup: dict[tuple[int, int], int] = {}
        for bond in normalized.GetBonds():
            key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            try:
                bond_lookup[key] = _RDKIT_TO_BOND_ID[bond.GetBondType()]
            except KeyError as exc:
                raise ValueError(f"Unsupported bond type: {bond.GetBondType()}") from exc
        bond_ids = tuple(
            bond_lookup.get(pair, BondToken.NO_BOND)
            for pair in upper_triangle_pairs(len(atom_ids))
        )
        return TokenGraph(atom_ids, bond_ids, smiles=smiles or canonical)

    def encode_smiles(self, smiles: str) -> TokenGraph:
        canonical, molecule = parse_and_filter_smiles(smiles)
        return self.encode_molecule(molecule, canonical)

    def decode_molecule(self, graph: TokenGraph, *, sanitize: bool = True) -> Chem.Mol:
        if not graph.is_clean(self.atom_vocabulary.clean_size):
            raise ValueError("Only fully clean graphs can be decoded")
        editable = Chem.RWMol()
        for token_id in graph.atom_ids:
            token = self.atom_vocabulary.decode(token_id)
            atom = Chem.Atom(token.atomic_number)
            atom.SetFormalCharge(token.formal_charge)
            atom.SetNumExplicitHs(token.total_hydrogens)
            atom.SetNoImplicit(True)
            editable.AddAtom(atom)
        for (i, j), bond_id in zip(upper_triangle_pairs(graph.n_atoms), graph.bond_ids):
            if bond_id == BondToken.NO_BOND:
                continue
            try:
                editable.AddBond(i, j, _BOND_ID_TO_RDKIT[bond_id])
            except KeyError as exc:
                raise ValueError(f"Unsupported clean bond ID: {bond_id}") from exc
        molecule = editable.GetMol()
        if sanitize:
            Chem.SanitizeMol(molecule)
        return molecule

    def decode_smiles(self, graph: TokenGraph) -> str:
        molecule = self.decode_molecule(graph, sanitize=True)
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


# Versioned declaration of allowed total valences U(Z, q).  The capacity used
# by VaDGM is max(U) - H; lower members are retained for protocol transparency.
DEFAULT_ALLOWED_VALENCES: Mapping[tuple[int, int], tuple[int, ...]] = {
    (5, -1): (4,),
    (5, 0): (3,),
    (6, -1): (3,),
    (6, 0): (4,),
    (6, 1): (3,),
    (7, -1): (2,),
    (7, 0): (3,),
    (7, 1): (4,),
    (8, -1): (1,),
    (8, 0): (2,),
    (8, 1): (3,),
    (9, 0): (1,),
    (14, 0): (4,),
    (15, -1): (4, 6),
    (15, 0): (3, 5),
    (15, 1): (4, 6),
    (16, -1): (1, 3, 5),
    (16, 0): (2, 4, 6),
    (16, 1): (3, 5),
    (17, 0): (1,),
    (35, 0): (1,),
    (53, 0): (1,),
}


class CapacityEngine:
    """Deterministic charge/H-aware upper-capacity rule engine."""

    def __init__(
        self,
        atom_vocabulary: AtomVocabulary,
        allowed_valences: Mapping[tuple[int, int], Sequence[int]] = DEFAULT_ALLOWED_VALENCES,
    ) -> None:
        self.atom_vocabulary = atom_vocabulary
        self.allowed_valences = {
            key: tuple(sorted(set(values))) for key, values in allowed_valences.items()
        }
        self._capacities = tuple(self._capacity(token) for token in atom_vocabulary.tokens)

    def _capacity(self, token: AtomToken) -> int:
        key = (token.atomic_number, token.formal_charge)
        if key not in self.allowed_valences:
            raise ValueError(f"No declared allowed-valence set for atom token {token}")
        values = self.allowed_valences[key]
        if not values or min(values) < 0:
            raise ValueError(f"Invalid allowed-valence declaration for {key}: {values}")
        capacity = max(values) - token.total_hydrogens
        if capacity < 0:
            raise ValueError(f"Hydrogen count exceeds declared maximum valence: {token}")
        return capacity

    def capacity(self, atom_id: int) -> int:
        if not 0 <= atom_id < self.atom_vocabulary.clean_size:
            raise ValueError(f"Capacity requires a clean atom ID, received {atom_id}")
        return self._capacities[atom_id]

    def allowed_bond_loads(self, atom_id: int) -> tuple[int, ...]:
        """返回 atom token 能对应的全部允许重原子键级负载。"""

        token = self.atom_vocabulary.decode(atom_id)
        total_valences = self.allowed_valences[(token.atomic_number, token.formal_charge)]
        return tuple(
            value - token.total_hydrogens
            for value in total_valences
            if value >= token.total_hydrogens
        )

    def loads(self, graph: TokenGraph) -> tuple[int, ...]:
        loads = [0] * graph.n_atoms
        for (i, j), bond_id in zip(upper_triangle_pairs(graph.n_atoms), graph.bond_ids):
            if bond_id == BondToken.MASK:
                continue
            if bond_id not in BondToken.ORDERS:
                raise ValueError(f"Unexpected bond token in an unpadded graph: {bond_id}")
            order = BondToken.ORDERS[bond_id]
            loads[i] += order
            loads[j] += order
        return tuple(loads)

    def feasible_atom_ids(
        self,
        graph: TokenGraph,
        atom_index: int,
        loads: Sequence[int] | None = None,
    ) -> tuple[int, ...]:
        current_loads = tuple(loads) if loads is not None else self.loads(graph)
        current = graph.atom_ids[atom_index]
        if current == self.atom_vocabulary.mask_id:
            return tuple(
                atom_id
                for atom_id, capacity in enumerate(self._capacities)
                if capacity >= current_loads[atom_index]
            )
        if not 0 <= current < self.atom_vocabulary.clean_size:
            raise ValueError(f"Unexpected atom token in an unpadded graph: {current}")
        return (current,)

    def max_feasible_capacity(
        self,
        graph: TokenGraph,
        atom_index: int,
        loads: Sequence[int] | None = None,
    ) -> int:
        feasible = self.feasible_atom_ids(graph, atom_index, loads)
        if not feasible:
            raise ValueError(f"Atom {atom_index} has an empty feasible token set")
        return max(self.capacity(atom_id) for atom_id in feasible)

    def atom_gate(
        self,
        graph: TokenGraph,
        atom_index: int,
        loads: Sequence[int] | None = None,
    ) -> torch.Tensor:
        load = (tuple(loads) if loads is not None else self.loads(graph))[atom_index]
        return torch.tensor(
            [capacity >= load for capacity in self._capacities],
            dtype=torch.bool,
        )

    def bond_gate(
        self,
        graph: TokenGraph,
        edge_index: int,
        loads: Sequence[int] | None = None,
    ) -> torch.Tensor:
        pairs = upper_triangle_pairs(graph.n_atoms)
        i, j = pairs[edge_index]
        current_loads = tuple(loads) if loads is not None else self.loads(graph)
        max_i = self.max_feasible_capacity(graph, i, current_loads)
        max_j = self.max_feasible_capacity(graph, j, current_loads)
        return torch.tensor(
            [
                current_loads[i] + BondToken.ORDERS[bond_id] <= max_i
                and current_loads[j] + BondToken.ORDERS[bond_id] <= max_j
                for bond_id in range(BondToken.CLEAN_SIZE)
            ],
            dtype=torch.bool,
        )

    def masked_incident_counts(self, graph: TokenGraph) -> tuple[int, ...]:
        """统计每个原子仍可用于终态补全的 MASK 键数量。"""

        counts = [0] * graph.n_atoms
        for (i, j), bond_id in zip(upper_triangle_pairs(graph.n_atoms), graph.bond_ids):
            if bond_id == BondToken.MASK:
                counts[i] += 1
                counts[j] += 1
        return tuple(counts)

    @staticmethod
    def _target_is_reachable(
        current_load: int,
        remaining_masked_bonds: int,
        target_loads: Sequence[int],
    ) -> bool:
        maximum_additional_load = 3 * remaining_masked_bonds
        return any(
            current_load <= target <= current_load + maximum_additional_load
            for target in target_loads
        )

    def terminal_atom_reachable(
        self,
        graph: TokenGraph,
        atom_index: int,
        *,
        atom_id: int | None = None,
        loads: Sequence[int] | None = None,
        masked_incident: Sequence[int] | None = None,
    ) -> bool:
        """检查单个原子是否仍可能到达某个声明的终态价态。

        这是局部必要条件：每条剩余 MASK 键最多再贡献 3 阶负载。
        它不解决不同端点共享剩余边所形成的全局 b-matching 问题。
        """

        current_loads = tuple(loads) if loads is not None else self.loads(graph)
        remaining = (
            tuple(masked_incident)
            if masked_incident is not None
            else self.masked_incident_counts(graph)
        )
        current_atom = graph.atom_ids[atom_index] if atom_id is None else atom_id
        if current_atom == self.atom_vocabulary.mask_id:
            candidates = range(self.atom_vocabulary.clean_size)
        elif 0 <= current_atom < self.atom_vocabulary.clean_size:
            candidates = (current_atom,)
        else:
            raise ValueError(f"Unexpected atom token in an unpadded graph: {current_atom}")
        return any(
            self._target_is_reachable(
                current_loads[atom_index],
                remaining[atom_index],
                self.allowed_bond_loads(candidate),
            )
            for candidate in candidates
        )

    def atom_terminal_gate(
        self,
        graph: TokenGraph,
        atom_index: int,
        *,
        loads: Sequence[int] | None = None,
        masked_incident: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """删除提交后无法由剩余 MASK 键到达允许价态的 atom 类别。"""

        current_loads = tuple(loads) if loads is not None else self.loads(graph)
        remaining = (
            tuple(masked_incident)
            if masked_incident is not None
            else self.masked_incident_counts(graph)
        )
        return torch.tensor(
            [
                self._target_is_reachable(
                    current_loads[atom_index],
                    remaining[atom_index],
                    self.allowed_bond_loads(atom_id),
                )
                for atom_id in range(self.atom_vocabulary.clean_size)
            ],
            dtype=torch.bool,
        )

    def bond_terminal_gate(
        self,
        graph: TokenGraph,
        edge_index: int,
        *,
        loads: Sequence[int] | None = None,
        masked_incident: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """检查每个 bond 类别提交后，两端是否仍局部终态可达。"""

        if graph.bond_ids[edge_index] != BondToken.MASK:
            raise ValueError("Terminal bond gate requires a MASK bond coordinate")
        current_loads = tuple(loads) if loads is not None else self.loads(graph)
        remaining = (
            tuple(masked_incident)
            if masked_incident is not None
            else self.masked_incident_counts(graph)
        )
        i, j = upper_triangle_pairs(graph.n_atoms)[edge_index]
        gate = []
        for bond_id in range(BondToken.CLEAN_SIZE):
            order = BondToken.ORDERS[bond_id]
            updated_loads = list(current_loads)
            updated_loads[i] += order
            updated_loads[j] += order
            updated_remaining = list(remaining)
            updated_remaining[i] -= 1
            updated_remaining[j] -= 1
            gate.append(
                self.terminal_atom_reachable(
                    graph,
                    i,
                    loads=updated_loads,
                    masked_incident=updated_remaining,
                )
                and self.terminal_atom_reachable(
                    graph,
                    j,
                    loads=updated_loads,
                    masked_incident=updated_remaining,
                )
            )
        return torch.tensor(gate, dtype=torch.bool)

    def terminal_reachability_holds(self, graph: TokenGraph) -> bool:
        """检查当前 partial graph 的逐原子局部终态可达不变量。"""

        loads = self.loads(graph)
        remaining = self.masked_incident_counts(graph)
        return all(
            self.terminal_atom_reachable(
                graph,
                atom_index,
                loads=loads,
                masked_incident=remaining,
            )
            for atom_index in range(graph.n_atoms)
        )

    def clean_graph_is_terminal_complete(self, graph: TokenGraph) -> bool:
        """要求每个 clean atom 的最终负载属于其允许价态集合。"""

        if not graph.is_clean(self.atom_vocabulary.clean_size):
            raise ValueError("Terminal completion evaluation requires a clean graph")
        loads = self.loads(graph)
        return all(
            load in self.allowed_bond_loads(atom_id)
            for load, atom_id in zip(loads, graph.atom_ids)
        )

    def invariant_holds(self, graph: TokenGraph) -> bool:
        loads = self.loads(graph)
        for atom_index in range(graph.n_atoms):
            feasible = self.feasible_atom_ids(graph, atom_index, loads)
            if not feasible:
                return False
            if loads[atom_index] > max(self.capacity(atom_id) for atom_id in feasible):
                return False
        return True

    def clean_graph_is_capacity_safe(self, graph: TokenGraph) -> bool:
        if not graph.is_clean(self.atom_vocabulary.clean_size):
            raise ValueError("Final capacity evaluation requires a clean graph")
        loads = self.loads(graph)
        return all(
            load <= self.capacity(atom_id)
            for load, atom_id in zip(loads, graph.atom_ids)
        )


def dense_bond_matrix(graph: TokenGraph, diagonal_value: int = BondToken.PAD) -> torch.Tensor:
    matrix = torch.full(
        (graph.n_atoms, graph.n_atoms),
        fill_value=diagonal_value,
        dtype=torch.long,
    )
    for (i, j), token_id in zip(upper_triangle_pairs(graph.n_atoms), graph.bond_ids):
        matrix[i, j] = token_id
        matrix[j, i] = token_id
    return matrix


def iter_atom_tokens(molecules: Iterable[Chem.Mol]) -> Iterator[AtomToken]:
    for molecule in molecules:
        for atom in molecule.GetAtoms():
            yield atom_token(atom)
