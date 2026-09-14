#!/usr/bin/env python3
"""Download pre-trained Base and Guidance checkpoints from GitHub Releases."""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = "https://github.com/xbtc-lab/VaDGM/releases/download/v1.0.0/checkpoints.zip"

def main():
    parser = argparse.ArgumentParser(description="Download VaDGM pretrained checkpoints")
    parser.add_argument("--url", type=str, default=RELEASE_URL, help="URL to checkpoints archive")
    parser.add_argument("--dest", type=Path, default=Path("artifacts"), help="Destination directory")
    args = parser.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "checkpoints.zip"

    print(f"[*] Downloading pre-trained weights from: {args.url}")
    print(f"[*] Target destination: {dest}")
    try:
        urllib.request.urlretrieve(args.url, zip_path)
        print("[*] Extracting checkpoints archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink()
        print("[OK] Checkpoints successfully installed to:", dest)
    except Exception as exc:
        print(f"[!] Download failed: {exc}")
        print("\nManual Installation Guide:")
        print(f"1. Download checkpoints.zip from: {args.url}")
        print(f"2. Extract the archive into: {dest}")
        print("   Expected directory structure:")
        print("   artifacts/")
        print("   ├── base/")
        print("   │   └── best.pt")
        print("   └── guidance/")
        print("       ├── logp_q50_best.pt")
        print("       ├── logp_q80_best.pt")
        print("       └── logp_q90_best.pt")

if __name__ == "__main__":
    main()
