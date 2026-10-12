from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mdpt.schema import TaskConfig


_BENCHMARK_ALIASES = {
    "aime": "aime2025",
    "aime2025": "aime2025",
    "gsm8k": "gsm8k",
    "bfcl": "bfcl",
    "humaneval": "humaneval",
    "gpqa": "gpqamain",
    "gpqamain": "gpqamain",
    "healthbench": "healthbench",
}


def canonical_benchmark(value: str) -> str:
    key = value.strip().lower().replace("-", "").replace("_", "")
    try:
        return _BENCHMARK_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark: {value!r}") from exc


def benchmark_assets_root() -> Path:
    return Path(__file__).resolve().parent / "assets" / "benchmarks"


def load_benchmark_metadata(task: TaskConfig) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for benchmark_id in task.benchmark_ids:
        path = benchmark_assets_root() / benchmark_id / "benchmark.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing benchmark metadata: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("benchmark_id") != benchmark_id:
            raise ValueError(f"benchmark metadata ID mismatch: {path}")
        metadata[benchmark_id] = value
    return metadata
