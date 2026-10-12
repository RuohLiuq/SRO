from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mdpt.discovery.adapters import get_adapter
from mdpt.io import read_json
from mdpt.schema import TaskConfig
from mdpt.discovery.schema import DataMethodSpec, Recipe, SearchSpace, SourceCatalog


def digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_methods(methods_dir: Path) -> dict[str, DataMethodSpec]:
    if not methods_dir.is_dir():
        raise FileNotFoundError(f"methods directory does not exist: {methods_dir}")
    methods: dict[str, DataMethodSpec] = {}
    for path in sorted(methods_dir.glob("*.json")):
        method = DataMethodSpec.model_validate(read_json(path))
        if method.method_id in methods:
            raise ValueError(f"duplicate method_id: {method.method_id!r}")
        methods[method.method_id] = method
    return methods


def task_benchmark_to_domain(task: TaskConfig) -> dict[str, str]:
    return {
        benchmark.benchmark_id: domain.domain_id
        for domain in task.domains
        for benchmark in domain.benchmarks
    }


def _same_ratio(left: dict[str, float], right: dict[str, float]) -> bool:
    if set(left) != set(right):
        return False
    left_total = sum(float(value) for value in left.values())
    right_total = sum(float(value) for value in right.values())
    return all(
        abs(float(left[key]) / left_total - float(right[key]) / right_total) < 1e-9
        for key in left
    )


def _validate_ratio(value: dict[str, float], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    if any(float(weight) <= 0 for weight in value.values()):
        raise ValueError(f"{label} weights must be positive")


def validate_configuration(
    task: TaskConfig,
    catalog: SourceCatalog,
    methods: dict[str, DataMethodSpec],
    space: SearchSpace,
    *,
    workspace: Path,
) -> None:
    benchmark_to_domain = task_benchmark_to_domain(task)
    task_domains = {domain.domain_id for domain in task.domains}
    task_benchmarks = set(benchmark_to_domain)

    if set(space.domains) != task_domains:
        raise ValueError("search-space domains must exactly match task domains")
    if set(space.benchmark_mix_choices) != task_domains:
        raise ValueError("benchmark_mix_choices must contain every task domain")

    sources = catalog.by_id()
    referenced_methods: set[str] = set()
    for source in catalog.sources:
        if source.benchmark_id not in task_benchmarks:
            raise ValueError(
                f"source {source.source_id!r} targets benchmark outside task: {source.benchmark_id!r}"
            )
        if source.evaluation_only:
            raise ValueError(f"evaluation-only source cannot be registered: {source.source_id!r}")
        if source.kind == "local":
            path = (workspace / str(source.path)).resolve()
            try:
                path.relative_to(workspace.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"local source must stay inside Discovery workspace: {source.source_id!r}"
                ) from exc

    for domain in task.domains:
        domain_space = space.domains[domain.domain_id]
        expected_benchmarks = {benchmark.benchmark_id for benchmark in domain.benchmarks}
        if set(domain_space.benchmarks) != expected_benchmarks:
            raise ValueError(
                f"search-space benchmarks for {domain.domain_id!r} must be {sorted(expected_benchmarks)}"
            )
        choices = space.benchmark_mix_choices[domain.domain_id]
        if not choices:
            raise ValueError(f"domain {domain.domain_id!r} requires benchmark mix choices")
        for index, ratio in enumerate(choices):
            _validate_ratio(ratio, expected_benchmarks, f"benchmark mix {domain.domain_id}[{index}]")
        equal_benchmark_mix = {benchmark_id: 1.0 for benchmark_id in expected_benchmarks}
        if not any(_same_ratio(choice, equal_benchmark_mix) for choice in choices):
            raise ValueError(
                f"domain {domain.domain_id!r} requires an equal benchmark mix for the framework seed"
            )

        for benchmark_id, benchmark_space in domain_space.benchmarks.items():
            if not benchmark_space.source_choices:
                raise ValueError(f"benchmark {benchmark_id!r} requires at least one source choice")
            for source_id in benchmark_space.source_choices:
                source = sources.get(source_id)
                if source is None:
                    raise ValueError(f"unknown source choice {source_id!r}")
                if source.benchmark_id != benchmark_id:
                    raise ValueError(
                        f"source {source_id!r} belongs to {source.benchmark_id!r}, not {benchmark_id!r}"
                    )
                if source.usage != "training":
                    raise ValueError(f"diagnostic source cannot be a training choice: {source_id!r}")
            for method_id in benchmark_space.method_choices:
                method = methods.get(method_id)
                if method is None:
                    raise ValueError(f"unknown method choice {method_id!r}")
                if method.benchmark_id != benchmark_id:
                    raise ValueError(
                        f"method {method_id!r} belongs to {method.benchmark_id!r}, not {benchmark_id!r}"
                    )
                referenced_methods.add(method_id)

    unreferenced = set(methods) - referenced_methods
    if unreferenced:
        raise ValueError(f"methods are not referenced by search space: {sorted(unreferenced)}")

    for method in methods.values():
        adapter = get_adapter(method.benchmark_id)
        supported = (
            adapter.rewrite_supported
            if method.operation == "rewrite"
            else adapter.correction_supported
        )
        if not supported:
            raise ValueError(
                f"benchmark {method.benchmark_id!r} does not support {method.operation}"
            )
        source_id = (
            method.input_source_id if method.operation == "rewrite" else method.diagnostic_source_id
        )
        source = sources.get(str(source_id))
        if source is None:
            raise ValueError(f"method {method.method_id!r} references unknown source {source_id!r}")
        if source.benchmark_id != method.benchmark_id:
            raise ValueError(
                f"method {method.method_id!r} and source {source.source_id!r} target different benchmarks"
            )
        expected_usage = "training" if method.operation == "rewrite" else "diagnostic"
        if source.usage != expected_usage:
            raise ValueError(
                f"{method.operation} method {method.method_id!r} requires a {expected_usage} source"
            )

    if not space.domain_mix_choices:
        raise ValueError("domain_mix_choices must not be empty")
    for index, ratio in enumerate(space.domain_mix_choices):
        _validate_ratio(ratio, task_domains, f"domain mix {index}")
    equal_domain_mix = {domain_id: 1.0 for domain_id in task_domains}
    if not any(_same_ratio(choice, equal_domain_mix) for choice in space.domain_mix_choices):
        raise ValueError("search space requires an equal domain mix for the framework seed")
    if "interleaved" not in space.multi_domain_modes:
        raise ValueError("search space must allow interleaved mode for the framework seed")
    if len(task_domains) == 1 and space.multi_domain_modes != ["interleaved"]:
        raise ValueError("single-domain tasks support only interleaved mode")


def validate_recipe(
    task: TaskConfig,
    catalog: SourceCatalog,
    methods: dict[str, DataMethodSpec],
    space: SearchSpace,
    recipe: Recipe,
    *,
    allow_correction: bool = False,
) -> None:
    task_domains = {domain.domain_id for domain in task.domains}
    if set(recipe.domains) != task_domains:
        raise ValueError("recipe domains must exactly match task domains")
    if set(recipe.domain_mix) != task_domains:
        raise ValueError("recipe domain_mix must exactly match task domains")
    if set(recipe.benchmark_mix) != task_domains:
        raise ValueError("recipe benchmark_mix must exactly match task domains")
    if not any(_same_ratio(recipe.domain_mix, choice) for choice in space.domain_mix_choices):
        raise ValueError("recipe domain_mix is not declared in search space")
    if recipe.multi_domain_mode not in space.multi_domain_modes:
        raise ValueError("recipe multi_domain_mode is not declared in search space")

    sources = catalog.by_id()
    for domain in task.domains:
        expected_benchmarks = {benchmark.benchmark_id for benchmark in domain.benchmarks}
        domain_recipe = recipe.domains[domain.domain_id]
        if set(domain_recipe.benchmarks) != expected_benchmarks:
            raise ValueError(
                f"recipe benchmarks for {domain.domain_id!r} must be {sorted(expected_benchmarks)}"
            )
        ratio = recipe.benchmark_mix.get(domain.domain_id)
        if ratio is None or not any(
            _same_ratio(ratio, choice) for choice in space.benchmark_mix_choices[domain.domain_id]
        ):
            raise ValueError(f"recipe benchmark mix for {domain.domain_id!r} is not declared")

        for benchmark_id, benchmark_recipe in domain_recipe.benchmarks.items():
            benchmark_space = space.domains[domain.domain_id].benchmarks[benchmark_id]
            if not benchmark_recipe.sources and not benchmark_recipe.data_method:
                raise ValueError(f"recipe selects no data for benchmark {benchmark_id!r}")
            for source_id in benchmark_recipe.sources:
                if source_id not in benchmark_space.source_choices:
                    raise ValueError(f"recipe source {source_id!r} is outside search space")
                if sources[source_id].benchmark_id != benchmark_id:
                    raise ValueError(f"recipe source {source_id!r} targets wrong benchmark")
            if benchmark_recipe.data_method:
                method_id = benchmark_recipe.data_method
                if method_id not in benchmark_space.method_choices:
                    raise ValueError(f"recipe method {method_id!r} is outside search space")
                method = methods[method_id]
                if method.operation == "correction" and not allow_correction:
                    raise ValueError("initial Recipe cannot use Correction before model diagnostics exist")
