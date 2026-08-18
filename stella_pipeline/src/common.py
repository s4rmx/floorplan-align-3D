"""Shared paths and config for the stella floorplan pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PIPELINE_ROOT / "config" / "pipeline.yaml"
OUTPUTS_ROOT = PIPELINE_ROOT / "outputs"


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["pipeline_root"] = str(PIPELINE_ROOT)
    cfg["outputs_root"] = str(OUTPUTS_ROOT)
    return cfg


def run_dir(name: str) -> Path:
    d = OUTPUTS_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def import_parent_src(parent_src: str | Path) -> None:
    p = str(Path(parent_src).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
