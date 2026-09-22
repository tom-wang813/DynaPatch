"""Download and extract this repo's checkpoints/data from its Zenodo record.

Two files are hosted there (see README.md "Data" / "Checkpoints"):
  - artifacts.tar.gz  -> extracts to artifacts/  (checkpoints, bug_set split manifests, risk matrices)
  - data.tar.gz       -> extracts to data/       (GTSRB / TT100K-Signs / LISA-Signs image folders)

Stdlib only (urllib/tarfile/hashlib) so this runs before `uv sync`, with the system python3.

Usage:
    python3 scripts/fetch_release_data.py                 # fetch + extract both
    python3 scripts/fetch_release_data.py --only artifacts # just the checkpoints/manifests
    python3 scripts/fetch_release_data.py --only data      # just the images
    python3 scripts/fetch_release_data.py --force          # re-download/re-extract even if present
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Filled in once the record is published; see ZENODO_RECORD.json at repo root, which this script
# reads instead of hard-coding the id/checksums here (so publishing a new version only means
# editing that file, not this script).
MANIFEST_PATH = ROOT / "ZENODO_RECORD.json"

CHUNK = 1 << 20  # 1 MiB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "dynapatch-fetch-release-data/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
        print()


def fetch_one(name: str, entry: dict, force: bool) -> Path:
    dest_dir = ROOT / entry["extracts_to"]
    archive_path = ROOT / f"{name}.tar.gz"

    if dest_dir.exists() and any(dest_dir.iterdir()) and not force:
        print(f"[{name}] {entry['extracts_to']}/ already populated, skipping (use --force to redo)")
        return archive_path

    if archive_path.exists() and not force:
        print(f"[{name}] {archive_path.name} already downloaded, verifying checksum")
    else:
        print(f"[{name}] fetching from Zenodo")
        _download(entry["url"], archive_path)

    print(f"[{name}] verifying sha256")
    got = _sha256(archive_path)
    want = entry["sha256"]
    if got != want:
        raise SystemExit(
            f"[{name}] checksum mismatch: expected {want}, got {got}. "
            f"Delete {archive_path} and re-run."
        )

    print(f"[{name}] extracting to {ROOT}/")
    with tarfile.open(archive_path) as tf:
        tf.extractall(ROOT)

    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["artifacts", "data"], help="fetch only this one")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract even if already present")
    parser.add_argument("--keep-archives", action="store_true", help="don't delete the .tar.gz after extracting")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"{MANIFEST_PATH} not found. This repo's release record hasn't been filled in yet -- "
            "see README.md for the Zenodo DOI, or download+extract artifacts.tar.gz/data.tar.gz "
            "manually from there into the repo root."
        )
    manifest = json.loads(MANIFEST_PATH.read_text())

    names = [args.only] if args.only else list(manifest["files"].keys())
    archives = []
    for name in names:
        archives.append(fetch_one(name, manifest["files"][name], args.force))

    if not args.keep_archives:
        for archive_path in archives:
            archive_path.unlink(missing_ok=True)

    print("Done.")


if __name__ == "__main__":
    main()
