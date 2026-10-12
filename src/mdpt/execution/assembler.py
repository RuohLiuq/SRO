from __future__ import annotations

import random
from dataclasses import dataclass, field

from mdpt.execution.data_builder import DomainPool


@dataclass
class MixedDataset:
    samples: list[dict]
    per_benchmark_counts: dict[str, int]
    stages: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.samples)


class MultiDomainAssembler:
    VALID_MODES = {"interleaved", "sequential", "staged_warmup"}

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    @staticmethod
    def _allocate(
        pools: dict[str, DomainPool],
        ratio: dict[str, float],
        total_train_samples: int,
    ) -> dict[str, int]:
        active = {key: float(value) for key, value in ratio.items() if key in pools and float(value) > 0}
        total_ratio = sum(active.values()) or 1.0
        remaining = max(1, int(total_train_samples))
        counts: dict[str, int] = {}
        items = list(active.items())
        for index, (benchmark_id, weight) in enumerate(items):
            if index == len(items) - 1:
                wanted = remaining
            else:
                wanted = int(total_train_samples * weight / total_ratio)
                remaining -= wanted
            counts[benchmark_id] = min(max(0, wanted), pools[benchmark_id].size)
        return {key: value for key, value in counts.items() if value > 0}

    @staticmethod
    def _interleave(groups: dict[str, list[dict]], order: list[str]) -> list[dict]:
        remaining = {key: list(groups.get(key, [])) for key in order}
        weights = {key: max(1, len(value)) for key, value in remaining.items() if value}
        if not weights:
            return []
        deficits = {key: 0.0 for key in weights}
        total_weight = float(sum(weights.values())) or 1.0
        output: list[dict] = []
        while any(remaining.get(key) for key in weights):
            for key, weight in weights.items():
                if remaining.get(key):
                    deficits[key] += weight / total_weight
            candidates = [key for key in order if remaining.get(key)]
            chosen = max(candidates, key=lambda key: (deficits.get(key, 0.0), weights.get(key, 0)))
            output.append(remaining[chosen].pop(0))
            deficits[chosen] -= 1.0
        return output

    def assemble(
        self,
        *,
        pools: dict[str, DomainPool],
        ratio: dict[str, float],
        total_train_samples: int,
        mode: str,
    ) -> MixedDataset:
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown multi_domain_mode: {mode}")
        counts = self._allocate(pools, ratio, total_train_samples)
        order = [key for key in ratio if key in counts]
        rng = random.Random(f"assemble:{self.seed}")
        groups: dict[str, list[dict]] = {}
        for key in order:
            pool = list(pools[key].samples)
            rng.shuffle(pool)
            groups[key] = pool[: counts[key]]

        stages: list[dict] = []
        if mode == "sequential":
            ordered: list[dict] = []
            for key in order:
                ordered.extend(groups[key])
                stages.append({"name": f"sequential_{key}", "benchmark": key, "size": len(groups[key])})
        elif mode == "staged_warmup":
            ordered = []
            rest: dict[str, list[dict]] = {}
            for key in order:
                cut = len(groups[key]) // 2
                ordered.extend(groups[key][:cut])
                rest[key] = groups[key][cut:]
                stages.append({"name": f"warmup_{key}", "benchmark": key, "size": cut})
            mixed = self._interleave(rest, order)
            ordered.extend(mixed)
            stages.append({"name": "interleaved_mixture", "size": len(mixed)})
        else:
            ordered = self._interleave(groups, order)
            stages = [{"name": "interleaved", "size": len(ordered), "benchmarks": dict(counts)}]

        return MixedDataset(
            samples=ordered,
            per_benchmark_counts=counts,
            stages=stages,
            stats={
                "total": len(ordered),
                "mix_ratio": ratio,
                "multi_domain_mode": mode,
            },
        )
