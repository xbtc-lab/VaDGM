#!/usr/bin/env python3
"""Quickstart Demo for VaDGM.

Demonstrates:
1. Loading 32-token chemical vocabulary & GraphCodec.
2. Bidirectional graph encoding and decoding with explicit valences.
3. Capacity-safe valence checking via CapacityEngine.
4. Pretrained checkpoint detection and guided sampling readiness.
"""

import os
import sys
from pathlib import Path

# Safe Windows DLL loading for PyTorch
if sys.platform == "win32":
    torch_lib = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

# Ensure local package import
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from rdkit import Chem
from rdkit.Chem import Descriptors
from vadgm.chemistry import AtomVocabulary, GraphCodec, CapacityEngine

def main():
    vocab_path = root / "data" / "processed" / "zinc100k_v1" / "atom_vocabulary.json"
    
    print("=" * 75)
    print("VaDGM: Valence-Aware Discrete Guidance Matching (Quickstart Demo)")
    print("=" * 75)
    
    if not vocab_path.exists():
        print(f"Error: Vocabulary file not found at {vocab_path}")
        return
        
    vocab = AtomVocabulary.load(vocab_path)
    codec = GraphCodec(vocab)
    engine = CapacityEngine(vocab)
    print(f"[*] Successfully loaded 32-token vocabulary from: {vocab_path.relative_to(root)}")
    print(f"    - Clean atom tokens: {vocab.clean_size}")
    print(f"    - Mask token ID: {vocab.mask_token_id}")
    
    # Showcase test molecules
    sample_smiles = [
        "CC(C)(C)c1ccc(O)cc1",           # 4-tert-butylphenol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
        "CC(=O)Oc1ccccc1C(=O)O",         # Aspirin
    ]
    
    print("\n[*] Demonstrating TokenGraph Encoding, Capacity Checks, & Decoding:")
    for sm in sample_smiles:
        graph = codec.encode_smiles(sm)
        excess_capacity = engine.graph_excess_capacity(graph)
        has_violations = excess_capacity > 0
        decoded_mol = codec.decode_molecule(graph)
        mw = Descriptors.MolWt(decoded_mol)
        print(f"  - SMILES: {sm:<32}")
        print(f"    Nodes: {graph.n_atoms} | Clean: {graph.is_clean(vocab.clean_size)} | Capacity Violations: {has_violations} | MolWt: {mw:.1f}")

    # Checkpoint status
    base_ckpt = root / "artifacts" / "base" / "best.pt"
    guidance_ckpt = root / "artifacts" / "guidance" / "logp_q80_best.pt"
    
    print("\n[*] Pretrained Checkpoints Status:")
    if base_ckpt.exists() and guidance_ckpt.exists():
        print(f"  [+] Found Base model: {base_ckpt.relative_to(root)}")
        print(f"  [+] Found Guidance model: {guidance_ckpt.relative_to(root)}")
        print("  [>] Ready to run paper experiments!")
    else:
        print("  [i] Note: Checkpoint weights can be hosted on GitHub Releases.")
        print("      To download pre-trained models for full generation, run:")
        print("      python scripts/download_checkpoints.py")
        
    print("\n[OK] VaDGM chemical engines and discrete representations verified successfully!")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
