"""独立子图的数据完整性、选择规则、统计分母与矢量导出测试。

此文件使用人工测试夹具，仅用于软件验收，不是实验结果。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
fitz = pytest.importorskip('fitz', reason='需要 figures 可选依赖进行矢量导出验收')
import yaml
from rdkit import Chem
from rdkit.Chem import Crippen

from figures.figure3 import _shared, plot_a_molecules as a, plot_b_distribution as b, plot_c_success_rate as c
from figures.figure3 import plot_figure3 as combined
from vadgm.chemistry import GraphCodec
from vadgm.data import vocabulary_from_smiles


def row(smiles, error, sample_id, condition='q50', seed=2027, **changes):
    logp = float(Crippen.MolLogP(Chem.MolFromSmiles(smiles)))
    return {'condition': condition, 'seed': seed, 'sample_id': sample_id, 'source_run': 'TEST_FIXTURE',
            'smiles': smiles, 'target': logp + error, 'logp': logp, 'absolute_error': error,
            'valid': True, 'connected': True, 'radical_free': True, 'capacity_safe': True,
            'terminal_complete': True, 'property_hit': True, 'strict_vthr': True,
            'exclusion_reason': '', **changes}


def test_screened_selection_diverse_scaffold_and_order():
    # 人工夹具：两个苯骨架与一个带哌啶环的骨架；目标误差不是实际实验结果。
    rows = [row('CCCOc1cccc(C(C)=O)c1', 0.3, '0'),
            row('CCOC(=O)CSCc1ccccc1', 0.1, '1'),
            row('Cc1ccc(C(=O)N2CCC(C)CC2)cc1', 0.2, '2')]
    selected, audit = a.select_molecules(list(reversed(rows)))
    assert [r['sample_id'] for r in selected] == ['0', '2']
    assert a.select_molecules(rows) == (selected, audit)
    assert selected[0]['scaffold'] != selected[1]['scaffold']
    assert audit[0]['screened_median_error'] == 0.2
    assert all(v == 3 for v in audit[0]['sequential_pass_counts'].values())
    assert selected[0]['source_smiles'] == rows[0]['smiles']
    assert selected[0]['logp'] == rows[0]['logp']
    picks, note = a.select_molecules(rows[:2])
    assert len(picks) == 2
    assert note[0]['selection_note'] == 'same_scaffold_fallback'
    # 重复分子不补位；未命中和过小结构不进入筛查后实例。
    picks, note = a.select_molecules([rows[0], {**rows[0], 'sample_id': 'duplicate'},
                                    {**rows[2], 'property_hit': False}, row('CC', .1, 'small')])
    assert len(picks) == 1
    assert note[0]['total'] == 4 and note[0]['eligible'] == 3
    assert note[0]['screened'] == 2 and note[0]['unique_screened'] == 1
    assert note[0]['selection_note'] == 'only_one_unique_molecule'


def test_structure_screen_exclusions_and_no_quality_fallback(monkeypatch):
    catalog = a.screening_catalog()
    for smiles, failure in [('CC', 'size'), ('Cc1ccc(C2CCC2)cc1CCO', 'rings'),
                            ('CCOP(=O)(OCC)Oc1ccccc1', 'elements_charge'),
                            ('CCCCCCOOCCCCCC', 'brenk')]:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert a.screen_structure(mol, catalog)[1] == failure
    monkeypatch.setattr(a.sascorer, 'calculateScore', lambda mol: 4.01)
    picks, note = a.select_molecules([row('CCCOc1cccc(C(C)=O)c1', .2, '0')])
    assert picks == []
    assert note[0]['sequential_pass_counts'] == {'size': 1, 'rings': 1, 'elements_charge': 1, 'brenk': 1, 'sa': 0}


def test_all_sample_denominator_includes_invalid():
    rows = [row('C', 0.1, '0'), row('CC', 0.2, '1', connected=False, strict_vthr=False),
            row('CCC', 0.8, '2', property_hit=False, strict_vthr=False),
            row('C', 0, '3', valid=False, logp=None, property_hit=False, strict_vthr=False)]
    result = c.rates(rows)[0]
    assert result['property_hit_all'] == 50
    assert result['strict_vthr'] == 25


def test_kde_integrates_to_one_without_dropping_tail_samples():
    import numpy as np
    values = np.array([-4., 0., 0., 1., 9.])
    grid = np.linspace(-6, 11, 4000)
    density = b.gaussian_density(values, grid, bandwidth=0.22)
    assert np.trapz(density, grid) == pytest.approx(1., abs=1e-5)
    expected = np.mean(np.exp(-0.5 * ((0. - values) / 0.22) ** 2)) / (0.22 * np.sqrt(2*np.pi))
    assert b.gaussian_density(values, np.array([0.]), 0.22)[0] == pytest.approx(expected)
    assert b.gaussian_density([], grid).sum() == 0
    with pytest.raises(ValueError):
        b.gaussian_density(values, grid, 0)


def make_runs(tmp_path):
    smiles = ['CC', 'CCC', 'c1ccccc1']
    vocab = vocabulary_from_smiles(smiles)
    vocab_path = tmp_path / 'vocabulary.json'
    vocab.save(vocab_path)
    codec = GraphCodec(vocab)
    seeds = [2027, 2028, 2029]
    cfg = {'experiment': 'figure3', 'settings': {'samples': 3}, 'seeds': seeds,
           'paths': {'output_dir': str(tmp_path), 'vocabulary': str(vocab_path)}, 'targets': {}}
    for index, condition in enumerate(_shared.CONDITIONS):
        cfg['targets'][condition] = {}
        if condition == 'q80':
            cfg['targets'][condition] = {'reuse_output_pattern': str(tmp_path / 'q80_seed{seed}.json')}
        for seed in seeds:
            samples = []
            for i, s in enumerate(smiles):
                graph = codec.encode_smiles(s)
                samples.append({'sample_id': str(i), 'graph': {'atom_ids': list(graph.atom_ids), 'bond_ids': list(graph.bond_ids)}, 'diagnostics': {}})
            payload = {'experiment': 'table1' if condition == 'q80' else 'figure3', 'condition': condition,
                       'method': 'VaDGM', 'seed': seed, 'target': {'target': 1 + index}, 'samples': samples}
            (tmp_path / f'{condition}_seed{seed}.json').write_text(json.dumps(payload))
    config = tmp_path / 'experiment.yaml'
    config.write_text(yaml.safe_dump(cfg))
    return argparse.Namespace(config=config, cache_dir=tmp_path / 'cache', hit_tolerance=0.5)


def test_loader_completeness_cache_and_tolerance(tmp_path, monkeypatch):
    args = make_runs(tmp_path)
    rows = _shared.load_samples(args)
    assert len(rows) == 27 and all(r['logp'] is not None for r in rows)
    original = _shared.evaluate_generation_file
    monkeypatch.setattr(_shared, 'evaluate_generation_file', lambda *a, **k: pytest.fail('Cache should be reused'))
    assert _shared.load_samples(args) == rows
    monkeypatch.setattr(_shared, 'evaluate_generation_file', original)
    args.hit_tolerance = 0.01
    changed = _shared.load_samples(args)
    assert sum(r['property_hit'] for r in changed) < sum(r['property_hit'] for r in rows)
    path = tmp_path / 'q90_seed2029.json'
    payload = json.loads(path.read_text())
    payload['samples'].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='样本数'):
        _shared.load_samples(args)
    path.unlink()
    with pytest.raises(ValueError, match='尚未完成'):
        _shared.load_samples(args)


def test_three_vector_exports_and_distributions(tmp_path):
    # 仅测试夹具：统一目标下，包含成功和不命中的分子；不输出到正式 outputs。
    rows = []
    for index, condition in enumerate(_shared.CONDITIONS):
        for seed in (2027, 2028, 2029):
            for j, smiles in enumerate(('CCCOc1cccc(C(C)=O)c1', 'Cc1ccc(C(=O)N2CCC(C)CC2)cc1', 'c1ccccc1', 'C1CCCCC1')):
                r = row(smiles, 0, str(j), condition, seed)
                r['target'] = 2.5 + index * 0.1
                r['absolute_error'] = abs(r['logp'] - r['target'])
                r['property_hit'] = r['strict_vthr'] = r['absolute_error'] <= 0.5
                rows.append(r)
    style = _shared.typography(_shared.HERE / 'typography.yaml')
    for module, name in ((a, 'a_molecules'), (b, 'b_distribution'), (c, 'c_success_rate')):
        module.draw(rows, style, tmp_path)
        svg = (tmp_path / f'{name}.svg').read_text(encoding='utf-8')
        assert '<text' in svg or ':text' in svg
        assert '<image' not in svg  # 分子结构不能退化为嵌入位图。
        if module is a:
            assert 'bond-' in svg  # 不允许所有位置都为空而仍通过矢量导出测试。
        with fitz.open(tmp_path / f'{name}.pdf') as pdf:
            assert len(pdf) == 1
            assert abs(pdf[0].rect.width * 25.4 / 72 - module.WIDTH_MM) < 0.1
            assert pdf[0].get_text().strip()
        assert (tmp_path / f'{name}.png').stat().st_size > 1000
    import csv
    bins = list(csv.DictReader((tmp_path / 'b_distribution_bins.csv').open(encoding='utf-8-sig')))
    for condition in _shared.CONDITIONS:
        data = [r for r in bins if r['condition'] == condition]
        assert sum(int(r['count']) for r in data) == 12
        assert sum(float(r['density']) * (float(r['right']) - float(r['left'])) for r in data) == pytest.approx(1)


def test_combination_preserves_fonts_sources_and_full_rate_denominator(tmp_path):
    # 人工夹具：紧误差命中、较大误差命中和失败样本共存，验证筛选只作用于 a。
    rows = []
    for condition in _shared.CONDITIONS:
        for seed in (2027, 2028, 2029):
            for index, smiles in enumerate(('CCCOc1cccc(C(C)=O)c1', 'Cc1ccc(C(=O)N2CCC(C)CC2)cc1',
                                           'CCOC(=O)CSCc1ccccc1', 'C')):
                r = row(smiles, 0, str(index), condition, seed)
                r['target'] = 2.77
                r['absolute_error'] = abs(r['logp'] - r['target'])
                r['property_hit'] = r['strict_vthr'] = r['absolute_error'] <= .5
                rows.append(r)
    original = [dict(r) for r in rows]
    style = _shared.typography(_shared.HERE / 'typography.yaml')
    combined.draw(rows, style, tmp_path)
    assert rows == original
    audit = json.loads((tmp_path / 'figure3_audit.json').read_text())
    assert audit['panel_scale'] == 1
    assert audit['total_rows'] == 36
    assert all(r['total'] == 12 for r in audit['selection']['conditions'])
    assert all(r['selected'] == 2 for r in audit['selection']['conditions'])
    import csv
    picks = list(csv.DictReader((tmp_path/'panels/a_selected_molecules.csv').open(encoding='utf-8-sig')))
    assert all(float(r['absolute_error']) <= combined.MAX_EXAMPLE_ERROR for r in picks)
    rate_rows = list(csv.DictReader((tmp_path/'panels/c_rates_per_seed.csv').open(encoding='utf-8-sig')))
    assert all(int(r['total']) == 4 for r in rate_rows)
    # 矢量字体核验：普通文字/元素 7 pt、标题 8 pt、编号 9 pt；所有实际使用字体为 TNR。
    with fitz.open(tmp_path / 'figure3.pdf') as pdf:
        assert len(pdf) == 1 and not pdf[0].get_images()
        assert pdf[0].rect.width * 25.4/72 == pytest.approx(audit['width_mm'], abs=.01)
        spans = [s for block in pdf[0].get_text('dict')['blocks'] if 'lines' in block
                 for line in block['lines'] for s in line['spans']]
        assert all('TimesNewRoman' in s['font'] for s in spans)
        assert all(round(s['size'], 2) in (7, 8, 9) for s in spans)
        for label in ('a', 'b', 'c'):
            span = next(s for s in spans if s['text'] == label)
            assert 'Bold' in span['font'] and span['size'] == pytest.approx(9)
        assert any(s['text'] == 'Generated molecules' for s in spans)
        assert all(s['size'] == pytest.approx(7) for s in spans if s['text'] in ('O', 'N'))
    import xml.etree.ElementTree as ET
    root = ET.parse(tmp_path / 'figure3.svg').getroot()
    ids = [node.get('id') for node in root.iter() if node.get('id')]
    assert len(ids) == len(set(ids))
    assert not root.findall('.//{http://www.w3.org/2000/svg}image')
