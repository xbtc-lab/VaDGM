from __future__ import annotations

import pytest

from vadgm.chemistry import (
    AtomToken,
    AtomVocabulary,
    BondToken,
    CapacityEngine,
    GraphCodec,
    TokenGraph,
)
from vadgm.data import vocabulary_from_smiles


def _capacity_fixture() -> tuple[AtomVocabulary, CapacityEngine]:
    vocabulary = AtomVocabulary(
        [
            AtomToken(6, 0, 0),
            AtomToken(6, 0, 4),
            AtomToken(8, 0, 0),
            AtomToken(8, 0, 2),
        ]
    )
    return vocabulary, CapacityEngine(vocabulary)


def test_atom_capacities_are_charge_and_hydrogen_aware() -> None:
    vocabulary, engine = _capacity_fixture()
    assert engine.capacity(vocabulary.encode(AtomToken(6, 0, 0))) == 4
    assert engine.capacity(vocabulary.encode(AtomToken(6, 0, 4))) == 0
    assert engine.capacity(vocabulary.encode(AtomToken(8, 0, 0))) == 2
    assert engine.capacity(vocabulary.encode(AtomToken(8, 0, 2))) == 0


def test_atom_gate_uses_current_revealed_bond_load() -> None:
    vocabulary, engine = _capacity_fixture()
    graph = TokenGraph(
        atom_ids=(vocabulary.mask_id, vocabulary.mask_id),
        bond_ids=(BondToken.DOUBLE,),
    )
    gate = engine.atom_gate(graph, 0)

    assert gate[vocabulary.encode(AtomToken(6, 0, 0))]
    assert not gate[vocabulary.encode(AtomToken(6, 0, 4))]
    assert gate[vocabulary.encode(AtomToken(8, 0, 0))]
    assert not gate[vocabulary.encode(AtomToken(8, 0, 2))]


def test_bond_gate_uses_both_endpoints_and_latest_load() -> None:
    vocabulary, engine = _capacity_fixture()
    graph = TokenGraph(
        atom_ids=(vocabulary.mask_id,) * 3,
        # pair order: (0,1), (0,2), (1,2)
        bond_ids=(BondToken.TRIPLE, BondToken.MASK, BondToken.NO_BOND),
    )
    gate = engine.bond_gate(graph, edge_index=1)

    assert gate.tolist() == [True, True, False, False]


def test_clean_capacity_check_is_separate_from_rdkit_validity() -> None:
    vocabulary = vocabulary_from_smiles(["C", "C#C"])
    codec = GraphCodec(vocabulary)
    engine = CapacityEngine(vocabulary)
    valid_graph = codec.encode_smiles("C#C")
    assert engine.clean_graph_is_capacity_safe(valid_graph)

    methane_id = vocabulary.encode(AtomToken(6, 0, 4))
    impossible = TokenGraph((methane_id, methane_id), (BondToken.SINGLE,))
    assert not engine.clean_graph_is_capacity_safe(impossible)
    with pytest.raises(Exception):
        codec.decode_molecule(impossible, sanitize=True)


def test_all_mask_initial_state_satisfies_path_invariant() -> None:
    vocabulary, engine = _capacity_fixture()
    graph = TokenGraph.all_mask(4, vocabulary.mask_id)
    assert BondToken.MASK != BondToken.NO_BOND
    assert vocabulary.mask_id not in range(vocabulary.clean_size)
    assert engine.invariant_holds(graph)


def test_terminal_completion_is_stricter_than_upper_capacity() -> None:
    vocabulary = AtomVocabulary([AtomToken(6, 0, 0), AtomToken(6, 0, 4)])
    engine = CapacityEngine(vocabulary)
    bare_carbon = TokenGraph((vocabulary.encode(AtomToken(6, 0, 0)),), ())
    methane = TokenGraph((vocabulary.encode(AtomToken(6, 0, 4)),), ())

    assert engine.clean_graph_is_capacity_safe(bare_carbon)
    assert not engine.clean_graph_is_terminal_complete(bare_carbon)
    assert engine.clean_graph_is_terminal_complete(methane)


def test_terminal_gates_preserve_a_reachable_allowed_valence() -> None:
    vocabulary = AtomVocabulary([AtomToken(6, 0, 0), AtomToken(6, 0, 3)])
    engine = CapacityEngine(vocabulary)
    carbon_h3 = vocabulary.encode(AtomToken(6, 0, 3))
    graph = TokenGraph((carbon_h3, carbon_h3), (BondToken.MASK,))

    # 两端都还差 1 阶负载，最后一条 MASK 键只能提交为单键。
    assert engine.bond_terminal_gate(graph, 0).tolist() == [False, True, False, False]
    completed = graph.with_bond(0, BondToken.SINGLE)
    assert engine.terminal_reachability_holds(completed)
    assert engine.clean_graph_is_terminal_complete(completed)
