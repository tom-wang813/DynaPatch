#!/usr/bin/env python3
"""Check the intended paper-reproduction environment."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

from common import ROOT, load_manifest


REQUIRED_MODULES = [
    "hydra",
    "omegaconf",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "psutil",
    "sklearn",
    "yaml",
    "PIL",
    "torch",
    "torchvision",
]


def _check_python() -> list[str]:
    major, minor = sys.version_info[:2]
    status = "OK" if (major, minor) >= (3, 10) else "FAIL"
    return [f"[{status}] python={major}.{minor}.{sys.version_info.micro}"]


def _check_uv() -> list[str]:
    uv = shutil.which("uv")
    return [f"[{'OK' if uv else 'WARN'}] uv={'present at ' + uv if uv else 'not found'}"]


def _check_modules() -> list[str]:
    lines: list[str] = []
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="dynapatch-mpl-"))
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - diagnostic surface
            lines.append(f"[FAIL] import {name}: {exc}")
            continue
        version = getattr(module, "__version__", "unknown")
        lines.append(f"[OK] import {name}: version={version}")
    return lines


def _check_checkpoints() -> list[str]:
    payload = load_manifest()
    present = 0
    missing = 0
    for entry in payload["frozen_backbones"]:
        if (ROOT / entry["checkpoint_path"]).exists():
            present += 1
        else:
            missing += 1
    return [f"[INFO] frozen_backbones_present={present} missing={missing}"]


def main() -> None:
    lines: list[str] = []
    lines.extend(_check_python())
    lines.extend(_check_uv())
    lines.extend(_check_modules())
    lines.extend(_check_checkpoints())
    lines.append(f"[INFO] repo_root={ROOT}")
    lines.append(f"[INFO] manifest={(ROOT / 'artifacts/checkpoints/manifest.json').resolve()}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
