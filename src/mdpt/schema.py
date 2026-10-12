from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkConfig(StrictModel):
    benchmark_id: str

    @field_validator("benchmark_id")
    @classmethod
    def nonempty_benchmark_id(cls, value: str) -> str:
        value = value.strip()
        if not value or not _IDENTIFIER.fullmatch(value):
            raise ValueError("benchmark_id must be a safe non-empty identifier")
        return value


class DomainConfig(StrictModel):
    domain_id: str
    benchmarks: list[BenchmarkConfig] = Field(min_length=1)

    @field_validator("domain_id")
    @classmethod
    def nonempty_domain_id(cls, value: str) -> str:
        value = value.strip()
        if not value or not _IDENTIFIER.fullmatch(value):
            raise ValueError("domain_id must be a safe non-empty identifier")
        return value

    @model_validator(mode="after")
    def unique_benchmarks_within_domain(self) -> "DomainConfig":
        benchmark_ids = [item.benchmark_id for item in self.benchmarks]
        if len(benchmark_ids) != len(set(benchmark_ids)):
            raise ValueError(f"domain {self.domain_id!r} contains duplicate benchmark_id values")
        return self


class BudgetConfig(StrictModel):
    discovery_hours: float = Field(gt=0)
    max_train_samples_per_trial: int = Field(gt=0)


class TaskConfig(StrictModel):
    schema_version: Literal[1] = 1
    base_model: str
    domains: list[DomainConfig] = Field(min_length=1)
    budget: BudgetConfig

    @field_validator("base_model")
    @classmethod
    def nonempty_base_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("base_model must be non-empty")
        return value

    @model_validator(mode="after")
    def unique_task_ids(self) -> "TaskConfig":
        domain_ids = [domain.domain_id for domain in self.domains]
        if len(domain_ids) != len(set(domain_ids)):
            raise ValueError("domain_id values must be unique")

        benchmark_ids = [
            benchmark.benchmark_id
            for domain in self.domains
            for benchmark in domain.benchmarks
        ]
        if len(benchmark_ids) != len(set(benchmark_ids)):
            raise ValueError("benchmark_id values must be unique across the task")
        return self

    @property
    def benchmark_ids(self) -> list[str]:
        return [
            benchmark.benchmark_id
            for domain in self.domains
            for benchmark in domain.benchmarks
        ]
