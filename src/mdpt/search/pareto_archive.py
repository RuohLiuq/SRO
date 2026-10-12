from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mdpt.discovery.schema import Recipe


class TrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe_id: str
    metrics: dict[str, float] = Field(default_factory=dict)
    objectives: dict[str, float] = Field(default_factory=dict)
    feasible: bool = True
    checkpoint_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArchiveBucket(str, Enum):
    front = "front"
    dominated_but_informative = "dominated_but_informative"
    failed = "failed"


class ArchiveEntry(BaseModel):
    recipe: Recipe
    sft_hyperparameters: dict[str, float | int]
    result: TrialResult | None = None
    bucket: ArchiveBucket = ArchiveBucket.front
    reason: str | None = None
    hv_contribution: float | None = None


def dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    keys = set(left) | set(right)
    return bool(keys) and all(left.get(key, 0.0) >= right.get(key, 0.0) for key in keys) and any(
        left.get(key, 0.0) > right.get(key, 0.0) for key in keys
    )


class DefaultParetoArchive:
    def __init__(self, max_front: int = 12) -> None:
        self._entries: dict[str, ArchiveEntry] = {}
        self.max_front = max_front

    def update(
        self,
        recipe: Recipe,
        sft_hyperparameters: dict[str, float | int],
        result: TrialResult,
    ) -> ArchiveEntry:
        key = result.recipe_id
        entry = ArchiveEntry(
            recipe=recipe,
            sft_hyperparameters=sft_hyperparameters,
            result=result,
        )
        if not result.feasible or not result.objectives:
            entry.bucket = ArchiveBucket.failed
            entry.reason = result.error or "infeasible"
        self._entries[key] = entry
        self._recompute()
        return self._entries[key]

    def _recompute(self) -> None:
        feasible = [
            entry
            for entry in self._entries.values()
            if entry.result and entry.result.feasible and entry.result.objectives
        ]
        for entry in feasible:
            assert entry.result is not None
            dominated = any(
                other.result is not None
                and dominates(other.result.objectives, entry.result.objectives)
                for other in feasible
                if other is not entry
            )
            entry.bucket = (
                ArchiveBucket.dominated_but_informative if dominated else ArchiveBucket.front
            )
            entry.hv_contribution = round(sum(entry.result.objectives.values()), 4)
        front = [entry for entry in feasible if entry.bucket == ArchiveBucket.front]
        if len(front) > self.max_front:
            front.sort(key=lambda entry: -(entry.hv_contribution or 0.0))
            for entry in front[self.max_front :]:
                entry.bucket = ArchiveBucket.dominated_but_informative
                entry.reason = "trimmed_by_crowding"

    def front(self) -> list[ArchiveEntry]:
        return [entry for entry in self._entries.values() if entry.bucket == ArchiveBucket.front]

    def dominated(self) -> list[ArchiveEntry]:
        return [
            entry
            for entry in self._entries.values()
            if entry.bucket == ArchiveBucket.dominated_but_informative
        ]

    def failed(self) -> list[ArchiveEntry]:
        return [entry for entry in self._entries.values() if entry.bucket == ArchiveBucket.failed]

    def all_entries(self) -> list[ArchiveEntry]:
        return list(self._entries.values())

    def get(self, recipe_id: str) -> ArchiveEntry | None:
        return self._entries.get(recipe_id)
