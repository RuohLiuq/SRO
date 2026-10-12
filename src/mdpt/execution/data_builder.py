from __future__ import annotations

import hashlib
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mdpt.discovery.methods.engine import materialize_augmentation
from mdpt.discovery.schema import BenchmarkRecipe, DataMethodSpec, SourceCatalog, SourceSpec
from mdpt.discovery.sources.data import load_source


def _text_of(sample: dict[str, Any]) -> str:
    messages = sample.get("messages") or sample.get("dialogs") or []
    if isinstance(messages, list):
        return " ".join(str(message.get("content") or "") for message in messages if isinstance(message, dict))
    return " ".join(str(value) for key, value in sample.items() if not str(key).startswith("_"))


def _deduplicate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for sample in samples:
        text = re.sub(r"\s+", " ", _text_of(sample).strip().lower())
        if not text:
            continue
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            output.append(sample)
    return output


def _contamination_guard(
    samples: list[dict[str, Any]],
    probes: list[str] | None,
) -> list[dict[str, Any]]:
    if not probes:
        return samples
    normalized = [probe.strip().lower() for probe in probes if len(probe.strip()) > 20]
    return [
        sample
        for sample in samples
        if not any(probe in _text_of(sample).lower() for probe in normalized)
    ]


def _resolve_source(source: SourceSpec, workspace: Path) -> SourceSpec:
    if source.kind != "local":
        return source
    path = Path(str(source.path)).expanduser()
    resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    return source.model_copy(update={"path": str(resolved)})


@dataclass
class DomainPool:
    domain: str
    benchmark: str
    samples: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.samples)


class DomainDataBuilder:
    def __init__(
        self,
        *,
        catalog: SourceCatalog,
        methods: dict[str, DataMethodSpec],
        workspace: Path,
        run_dir: Path,
        runtime: dict[str, Any],
        per_source_limit: int = 2000,
        generation_samples: int = 500,
        seed: int = 0,
    ) -> None:
        self.catalog = catalog
        self.methods = methods
        self.workspace = workspace
        self.run_dir = run_dir
        self.runtime = runtime
        self.per_source_limit = per_source_limit
        self.generation_samples = generation_samples
        self.seed = seed

    def build(
        self,
        *,
        domain_id: str,
        benchmark_id: str,
        recipe: BenchmarkRecipe,
        benchmark_probes: list[str] | None = None,
        deadline_monotonic: float | None = None,
    ) -> DomainPool:
        source_map = self.catalog.by_id()
        selected: list[SourceSpec] = [source_map[source_id] for source_id in recipe.sources]
        generated_source: SourceSpec | None = None
        if recipe.data_method:
            method = self.methods[recipe.data_method]
            previous_rewrite = os.environ.get("MDPT_REWRITE_INPUT_EXAMPLES")
            previous_correction = os.environ.get("MDPT_CORRECTION_INPUT_EXAMPLES")
            os.environ["MDPT_REWRITE_INPUT_EXAMPLES"] = str(self.generation_samples)
            os.environ["MDPT_CORRECTION_INPUT_EXAMPLES"] = str(self.generation_samples)
            try:
                resolved_sources = {
                    source_id: _resolve_source(source, self.workspace)
                    for source_id, source in source_map.items()
                }
                generated_source = materialize_augmentation(
                    method,
                    resolved_sources,
                    self.runtime,
                    self.run_dir,
                    deadline_monotonic=deadline_monotonic,
                )
            finally:
                if previous_rewrite is None:
                    os.environ.pop("MDPT_REWRITE_INPUT_EXAMPLES", None)
                else:
                    os.environ["MDPT_REWRITE_INPUT_EXAMPLES"] = previous_rewrite
                if previous_correction is None:
                    os.environ.pop("MDPT_CORRECTION_INPUT_EXAMPLES", None)
                else:
                    os.environ["MDPT_CORRECTION_INPUT_EXAMPLES"] = previous_correction
            selected.append(generated_source)

        rows: list[dict[str, Any]] = []
        loaded_by_source: dict[str, int] = {}
        for source in selected:
            resolved = _resolve_source(source, self.workspace)
            loaded = load_source(resolved, self.per_source_limit)
            loaded_by_source[source.source_id] = len(loaded)
            for row in loaded:
                item = dict(row)
                item.setdefault("_domain_id", domain_id)
                item.setdefault("_benchmark_id", benchmark_id)
                item.setdefault("_source_id", source.source_id)
                rows.append(item)

        loaded_count = len(rows)
        rows = _deduplicate(rows)
        rows = _contamination_guard(rows, benchmark_probes)
        random.Random(f"pool:{self.seed}:{domain_id}:{benchmark_id}").shuffle(rows)
        return DomainPool(
            domain=domain_id,
            benchmark=benchmark_id,
            samples=rows,
            stats={
                "n_loaded": loaded_count,
                "n_after_guard": len(rows),
                "loaded_by_source": loaded_by_source,
                "sources": [source.source_id for source in selected],
                "data_method": recipe.data_method,
                "generated_source": (
                    generated_source.model_dump(mode="json") if generated_source is not None else None
                ),
            },
        )
