from __future__ import annotations

import math
import random

from mdpt.search.featurizer import RecipeFeaturizer, flatten_benchmark_mix
from mdpt.search.pareto_archive import DefaultParetoArchive
from mdpt.search.proposal import Candidate
from mdpt.search.surrogate import MultiOutputSurrogate


_RISK = {"low": 0.15, "medium": 0.4, "high": 0.7}
_CONFIDENCE = {"low": 0.3, "medium": 0.6, "high": 0.85}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _non_dominated(points: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    unique = list(dict.fromkeys(points))
    return [
        point
        for index, point in enumerate(unique)
        if not any(
            index != other_index
            and all(other[axis] >= point[axis] for axis in range(len(point)))
            and any(other[axis] > point[axis] for axis in range(len(point)))
            for other_index, other in enumerate(unique)
        )
    ]


def _hypervolume(points: list[tuple[float, ...]], reference: tuple[float, ...]) -> float:
    if not reference:
        return 0.0
    valid = [
        point
        for point in points
        if len(point) == len(reference)
        and all(point[index] > reference[index] for index in range(len(reference)))
    ]
    if not valid:
        return 0.0
    front = _non_dominated(valid)
    if len(reference) == 1:
        return max(point[0] for point in front) - reference[0]
    coordinates = sorted({reference[-1], *(point[-1] for point in front)})
    volume = 0.0
    for lower, upper in zip(coordinates, coordinates[1:]):
        active = [point[:-1] for point in front if point[-1] >= upper]
        if active:
            volume += (upper - lower) * _hypervolume(active, reference[:-1])
    return volume


class SurrogateAcquisition:
    def __init__(
        self,
        *,
        featurizer: RecipeFeaturizer,
        objective_names: list[str],
        mc_samples: int = 64,
        min_fit: int = 5,
        seed: int = 0,
    ) -> None:
        self.featurizer = featurizer
        self.objective_names = objective_names
        self.mc_samples = mc_samples
        self.min_fit = min_fit
        self.rng = random.Random(seed)
        self.surrogate: MultiOutputSurrogate | None = None
        self.front: list[tuple[float, ...]] = []

    def fit(self, archive: DefaultParetoArchive) -> None:
        entries = [
            entry
            for entry in archive.all_entries()
            if entry.result
            and entry.result.feasible
            and set(entry.result.objectives) == set(self.objective_names)
        ]
        self.front = _non_dominated(
            [
                tuple(float(entry.result.objectives[name]) for name in self.objective_names)
                for entry in entries
                if entry.result is not None
            ]
        )
        if len(entries) < self.min_fit:
            self.surrogate = None
            return
        features = [
            self.featurizer.transform(entry.recipe, entry.sft_hyperparameters)
            for entry in entries
        ]
        targets = [entry.result.objectives for entry in entries if entry.result is not None]
        self.surrogate = MultiOutputSurrogate(self.objective_names, seed=0).fit(features, targets)

    @staticmethod
    def _signature(candidate: Candidate) -> tuple[object, ...]:
        recipe = candidate.recipe
        mix = tuple(
            sorted(
                (benchmark_id, round(weight, 2))
                for benchmark_id, weight in flatten_benchmark_mix(recipe).items()
            )
        )
        selectors = tuple(
            sorted(
                (
                    benchmark_id,
                    tuple(sorted(benchmark_recipe.sources)),
                    benchmark_recipe.data_method,
                )
                for domain_recipe in recipe.domains.values()
                for benchmark_id, benchmark_recipe in domain_recipe.benchmarks.items()
            )
        )
        return (mix, "sft", "lora", recipe.multi_domain_mode, selectors)

    def _novelty(self, candidate: Candidate, archive: DefaultParetoArchive) -> float:
        signature = self._signature(candidate)
        existing = [
            self._signature(
                Candidate(
                    recipe=entry.recipe,
                    sft_hyperparameters=entry.sft_hyperparameters,
                    llm_prior={},
                )
            )
            for entry in archive.all_entries()
        ]
        if not existing:
            return 1.0
        max_similarity = max(
            sum(left == right for left, right in zip(signature, other))
            for other in existing
        )
        return round(1.0 - max_similarity / len(signature), 4)

    def score_details(
        self,
        candidate: Candidate,
        archive: DefaultParetoArchive,
    ) -> dict[str, object]:
        if self.surrogate and self.surrogate.fitted:
            features = self.featurizer.transform(candidate.recipe, candidate.sft_hyperparameters)
            samples = self.surrogate.sample(features, self.mc_samples, self.rng)
            reference = tuple(0.0 for _ in self.objective_names)
            base_hv = _hypervolume(self.front, reference)
            improvements: list[float] = []
            per_objective = {name: [] for name in self.objective_names}
            for sample in samples:
                point = tuple(float(sample.get(name, 0.0)) for name in self.objective_names)
                for name, value in zip(self.objective_names, point):
                    per_objective[name].append(value)
                improvements.append(max(0.0, _hypervolume(self.front + [point], reference) - base_hv))
            score = _mean(improvements)
            return {
                "mode": "mc_ehvi",
                "score": score,
                "ehvi": score,
                "mc_samples": len(samples),
                "predicted_mean": {name: _mean(values) for name, values in per_objective.items()},
                "predicted_std": {name: _std(values) for name, values in per_objective.items()},
                "objective_coordinates": "domain_mean_static_reference_normalized_gain",
                "objective_names": list(self.objective_names),
                "reference_point": {name: 0.0 for name in self.objective_names},
                "normalized_pareto_front": [
                    {name: point[index] for index, name in enumerate(self.objective_names)}
                    for point in self.front
                ],
                "archive_hypervolume": base_hv,
            }
        confidence = _CONFIDENCE.get(candidate.llm_prior.get("confidence", "medium"), 0.6)
        risk = _RISK.get(candidate.llm_prior.get("risk", "medium"), 0.4)
        effect = candidate.llm_prior.get("expected_effect", "").lower()
        effect_proxy = 0.02 + 0.01 * sum(
            word in effect for word in ("increase", "improve", "recover", "preserve")
        )
        novelty = self._novelty(candidate, archive)
        score = effect_proxy + 0.4 * confidence + 0.2 * novelty - 0.3 * risk
        return {
            "mode": "cold_start",
            "score": score,
            "llm_confidence": confidence,
            "effect_proxy": effect_proxy,
            "novelty": novelty,
            "risk": risk,
        }
