from __future__ import annotations

from collections.abc import Mapping

from mdpt.discovery.schema import Recipe, SearchSpace
from mdpt.schema import TaskConfig


def flatten_benchmark_mix(recipe: Recipe) -> dict[str, float]:
    domain_total = sum(float(value) for value in recipe.domain_mix.values())
    if domain_total <= 0:
        raise ValueError("domain_mix total must be positive")
    result: dict[str, float] = {}
    for domain_id, domain_weight in recipe.domain_mix.items():
        inner = recipe.benchmark_mix[domain_id]
        benchmark_total = sum(float(value) for value in inner.values())
        if benchmark_total <= 0:
            raise ValueError(f"benchmark_mix total must be positive: {domain_id}")
        for benchmark_id, benchmark_weight in inner.items():
            result[benchmark_id] = (
                float(domain_weight) / domain_total
                * float(benchmark_weight) / benchmark_total
            )
    return result


class RecipeFeaturizer:
    def __init__(
        self,
        *,
        task: TaskConfig,
        space: SearchSpace,
        hp_names: list[str],
    ) -> None:
        self.task = task
        self.space = space
        self.hp_names = sorted(hp_names)
        self.benchmark_order = task.benchmark_ids
        self.mode_order = list(space.multi_domain_modes)
        self.source_axes: list[tuple[str, str]] = []
        self.method_axes: list[tuple[str, str]] = []
        for domain in task.domains:
            domain_space = space.domains[domain.domain_id]
            for benchmark in domain.benchmarks:
                benchmark_id = benchmark.benchmark_id
                benchmark_space = domain_space.benchmarks[benchmark_id]
                self.source_axes.extend(
                    (benchmark_id, source_id) for source_id in benchmark_space.source_choices
                )
                self.method_axes.extend(
                    (benchmark_id, method_id) for method_id in benchmark_space.method_choices
                )

    @property
    def feature_names(self) -> list[str]:
        return [
            *(f"mix::{benchmark_id}" for benchmark_id in self.benchmark_order),
            *(f"mode::{mode}" for mode in self.mode_order),
            "n_sources",
            "has_generated_source",
            *(f"source::{benchmark_id}::{source_id}" for benchmark_id, source_id in self.source_axes),
            *(f"method::{benchmark_id}::{method_id}" for benchmark_id, method_id in self.method_axes),
            *(f"hp::{name}" for name in self.hp_names),
        ]

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def transform(
        self,
        recipe: Recipe,
        sft_hyperparameters: Mapping[str, float | int],
    ) -> list[float]:
        flat_mix = flatten_benchmark_mix(recipe)
        selected_sources: dict[str, set[str]] = {}
        selected_methods: dict[str, str | None] = {}
        for domain_recipe in recipe.domains.values():
            for benchmark_id, benchmark_recipe in domain_recipe.benchmarks.items():
                selected_sources[benchmark_id] = set(benchmark_recipe.sources)
                selected_methods[benchmark_id] = benchmark_recipe.data_method

        values: list[float] = [float(flat_mix.get(benchmark_id, 0.0)) for benchmark_id in self.benchmark_order]
        values.extend(1.0 if recipe.multi_domain_mode == mode else 0.0 for mode in self.mode_order)
        values.extend(
            [
                float(sum(len(sources) for sources in selected_sources.values())),
                1.0 if any(method is not None for method in selected_methods.values()) else 0.0,
            ]
        )
        values.extend(
            1.0 if source_id in selected_sources.get(benchmark_id, set()) else 0.0
            for benchmark_id, source_id in self.source_axes
        )
        values.extend(
            1.0 if selected_methods.get(benchmark_id) == method_id else 0.0
            for benchmark_id, method_id in self.method_axes
        )
        for name in self.hp_names:
            value = float(sft_hyperparameters.get(name, 0.0))
            values.append(value * 1e5 if name == "learning_rate" else value)
        return values
