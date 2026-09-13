#!/usr/bin/env python3
"""Validate the paper-facing asset inventory against disk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import MANIFEST_PATH, ROOT, resolve_existing_path


COLLECTION_TEMPLATE = ROOT / "artifacts" / "checkpoints" / "asset_collection_template.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _exists(rel_path: str) -> bool:
    return (ROOT / rel_path).exists()


def _resolve_path(path: str, *, is_repair: bool = False) -> str | None:
    if is_repair:
        return resolve_existing_path(path)
    return path if _exists(path) else None


def _summarize_entries(entries: list[dict], *, path_field: str) -> tuple[int, int, list[str]]:
    present = 0
    missing = 0
    missing_items: list[str] = []
    for entry in entries:
        path = str(entry[path_field])
        resolved = _resolve_path(path, is_repair="experiment_id" in entry or "stage" in entry)
        if resolved is not None:
            present += 1
        else:
            missing += 1
            label = entry.get("dataset", "unknown")
            if "backbone" in entry:
                label = f"{label}/{entry['backbone']}"
            missing_items.append(f"{label}: {path}")
    return present, missing, missing_items


def _dataset_summary(entries: list[dict]) -> tuple[int, int, list[str]]:
    present = 0
    missing = 0
    missing_items: list[str] = []
    for entry in entries:
        path = str(entry["data_root"])
        if _exists(path):
            present += 1
        else:
            missing += 1
            missing_items.append(f"{entry['dataset']}: {path}")
    return present, missing, missing_items


def _build_summary(
    *,
    backbone_present: int,
    backbone_missing: int,
    backbone_missing_items: list[str],
    repair_present: int,
    repair_missing: int,
    repair_missing_items: list[str],
    data_present: int,
    data_missing: int,
    data_missing_items: list[str],
) -> dict[str, object]:
    total_missing = backbone_missing + repair_missing + data_missing
    return {
        "frozen_backbones": {
            "present": backbone_present,
            "missing": backbone_missing,
            "missing_items": backbone_missing_items,
        },
        "repair_checkpoints": {
            "present": repair_present,
            "missing": repair_missing,
            "missing_items": repair_missing_items,
        },
        "datasets": {
            "present": data_present,
            "missing": data_missing,
            "missing_items": data_missing_items,
        },
        "total_missing": total_missing,
        "status": "ready" if total_missing == 0 else "blocked",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Checkpoint manifest JSON.")
    parser.add_argument(
        "--collection-template",
        default=str(COLLECTION_TEMPLATE),
        help="Asset collection template JSON.",
    )
    parser.add_argument("--json-out", default=None, help="Optional path to write a machine-readable summary.")
    parser.add_argument(
        "--fail-if-missing",
        action="store_true",
        help="Exit non-zero if any required asset is missing.",
    )
    args = parser.parse_args()

    manifest = _load_json(Path(args.manifest))
    collection = _load_json(Path(args.collection_template))

    backbone_present, backbone_missing, backbone_missing_items = _summarize_entries(
        manifest["frozen_backbones"],
        path_field="checkpoint_path",
    )
    repair_present, repair_missing, repair_missing_items = _summarize_entries(
        manifest["experiment_checkpoints"],
        path_field="checkpoint_file",
    )
    data_present, data_missing, data_missing_items = _dataset_summary(collection["datasets"])
    summary = _build_summary(
        backbone_present=backbone_present,
        backbone_missing=backbone_missing,
        backbone_missing_items=backbone_missing_items,
        repair_present=repair_present,
        repair_missing=repair_missing,
        repair_missing_items=repair_missing_items,
        data_present=data_present,
        data_missing=data_missing,
        data_missing_items=data_missing_items,
    )

    print(f"[INFO] frozen_backbones present={backbone_present} missing={backbone_missing}")
    for item in backbone_missing_items:
        print(f"[MISS] backbone {item}")

    print(f"[INFO] repair_checkpoints present={repair_present} missing={repair_missing}")
    for item in repair_missing_items:
        print(f"[MISS] repair {item}")

    print(f"[INFO] datasets present={data_present} missing={data_missing}")
    for item in data_missing_items:
        print(f"[MISS] dataset {item}")

    print(f"[SUMMARY] total_missing={summary['total_missing']} status={summary['status']}")

    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] wrote_json={out_path}")

    if args.fail_if_missing and int(summary["total_missing"]) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
