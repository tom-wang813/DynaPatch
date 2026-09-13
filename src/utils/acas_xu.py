from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class AcasXuPropertySpec:
    property_id: str
    path: str
    raw_text: str
    input_constraint_count: int
    output_constraint_count: int


def infer_property_id(path: Path) -> str:
    stem = path.stem
    match = re.search(r"property[_-]?(\d+)", stem, re.IGNORECASE)
    if match:
        return f"P{int(match.group(1))}"
    match = re.search(r"prop[_-]?(\d+)", stem, re.IGNORECASE)
    if match:
        return f"P{int(match.group(1))}"
    return stem


def load_property_spec(path: Path) -> AcasXuPropertySpec:
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    input_constraint_count = len(re.findall(r"\bX_[0-9]+\b", raw_text))
    output_constraint_count = len(re.findall(r"\bY_[0-9]+\b", raw_text))
    return AcasXuPropertySpec(
        property_id=infer_property_id(path),
        path=str(path),
        raw_text=raw_text,
        input_constraint_count=input_constraint_count,
        output_constraint_count=output_constraint_count,
    )
