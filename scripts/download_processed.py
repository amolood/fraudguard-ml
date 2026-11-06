"""Download the exact processed training data (v4.0) from the GitHub release and verify it.

    python scripts/download_processed.py            # all three datasets (~400 MB)
    python scripts/download_processed.py ulb paysim

Files land in data/processed/ and are checked against SHA256SUMS.txt.
This is the fastest way to reproduce the published results; to rebuild the
tables from the original Kaggle sources instead, run scripts/download_data.py
followed by `python -m fraudguard.prepare`.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/amolood/fraudguard-ml/releases/download/data-v4.0"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
FILES = {"paysim": "paysim.parquet", "ulb": "ulb.parquet", "samld": "samld.parquet"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(keys: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{RELEASE}/SHA256SUMS.txt") as r:
        expected = dict(line.split()[::-1] for line in r.read().decode().splitlines() if line.strip())
    for key in keys:
        name = FILES[key]
        dest = OUT / name
        if not (dest.exists() and sha256(dest) == expected[name]):
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(f"{RELEASE}/{name}", dest)
        if sha256(dest) != expected[name]:
            raise SystemExit(f"checksum mismatch for {name}")
        print(f"ok  {name}")


if __name__ == "__main__":
    unknown = set(sys.argv[1:]) - set(FILES)
    if unknown:
        raise SystemExit(f"unknown dataset(s): {sorted(unknown)}; choose from {list(FILES)}")
    main(sys.argv[1:] or list(FILES))
