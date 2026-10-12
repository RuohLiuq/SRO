from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceFormat(StrictModel):
    messages_field: str | None = None
    user_fields: list[str] = Field(
        default_factory=lambda: ["user", "instruction", "prompt", "question", "problem"]
    )
    assistant_fields: list[str] = Field(
        default_factory=lambda: ["assistant", "output", "answer", "solution", "response"]
    )
    target_fields: list[str] = Field(default_factory=list)
    system_fields: list[str] = Field(default_factory=lambda: ["system"])
    formatter: Literal[
        "plain",
        "posttrainbench_math",
        "posttrainbench_humaneval",
        "posttrainbench_gpqa",
        "posttrainbench_bfcl",
        "healthbench",
    ] = "plain"
    choices_field: str = "choices"
    user_template: str = "{text}"
    assistant_template: str = "{text}"


class SourceSpec(StrictModel):
    source_id: str
    benchmark_id: str
    kind: Literal["local", "huggingface"]
    path: str | None = None
    dataset: str | None = None
    subset: str | None = None
    split: str = "train"
    revision: str | None = None
    license: str | None = None
    format: SourceFormat = Field(default_factory=SourceFormat)
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage: Literal["training", "diagnostic"] = "training"
    evaluation_only: bool = False

    @field_validator("source_id", "benchmark_id")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        value = value.strip()
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("source_id and benchmark_id must be safe non-empty identifiers")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> "SourceSpec":
        if self.kind == "local" and not self.path:
            raise ValueError(f"local source {self.source_id!r} requires path")
        if self.kind == "huggingface" and not self.dataset:
            raise ValueError(f"Hugging Face source {self.source_id!r} requires dataset")
        return self

    @property
    def adapter_id(self) -> str:
        return self.benchmark_id


class SourceCatalog(StrictModel):
    schema_version: Literal[1] = 1
    sources: list[SourceSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> "SourceCatalog":
        ids = [source.source_id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source_id values must be unique")
        return self

    def by_id(self) -> dict[str, SourceSpec]:
        return {source.source_id: source for source in self.sources}


class DataMethodSpec(StrictModel):
    method_id: str
    benchmark_id: str
    operation: Literal["rewrite", "correction"]
    input_source_id: str | None = None
    diagnostic_source_id: str | None = None
    evolution_axes: list[str] = Field(default_factory=list)
    rewrite_prompt: str | None = None
    answer_prompt: str | None = None
    correction_prompt: str | None = None
    description: str = ""

    @field_validator("method_id", "benchmark_id")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        value = value.strip()
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("method_id and benchmark_id must be safe non-empty identifiers")
        return value

    @model_validator(mode="after")
    def validate_operation(self) -> "DataMethodSpec":
        if self.operation == "rewrite":
            if not self.input_source_id or self.diagnostic_source_id:
                raise ValueError("rewrite requires input_source_id only")
            if not self.evolution_axes or any(not axis.strip() for axis in self.evolution_axes):
                raise ValueError("rewrite requires non-empty evolution_axes")
            if not str(self.rewrite_prompt or "").strip():
                raise ValueError("rewrite requires rewrite_prompt")
            if not str(self.answer_prompt or "").strip():
                raise ValueError("rewrite requires answer_prompt")
            if self.correction_prompt is not None:
                raise ValueError("rewrite does not accept correction_prompt")
        else:
            if not self.diagnostic_source_id or self.input_source_id:
                raise ValueError("correction requires diagnostic_source_id only")
            if not str(self.correction_prompt or "").strip():
                raise ValueError("correction requires correction_prompt")
            if self.evolution_axes or self.rewrite_prompt is not None or self.answer_prompt is not None:
                raise ValueError("correction does not accept Rewrite fields")
        return self


class BenchmarkSpace(StrictModel):
    source_choices: list[str] = Field(default_factory=list)
    method_choices: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_choices(self) -> "BenchmarkSpace":
        if len(self.source_choices) != len(set(self.source_choices)):
            raise ValueError("source_choices contains duplicates")
        if len(self.method_choices) != len(set(self.method_choices)):
            raise ValueError("method_choices contains duplicates")
        return self


class DomainSpace(StrictModel):
    benchmarks: dict[str, BenchmarkSpace]


class SearchSpace(StrictModel):
    schema_version: Literal[1] = 1
    domains: dict[str, DomainSpace]
    domain_mix_choices: list[dict[str, float]]
    benchmark_mix_choices: dict[str, list[dict[str, float]]]
    multi_domain_modes: list[Literal["interleaved", "sequential", "staged_warmup"]] = Field(
        default_factory=lambda: ["interleaved"]
    )

class BenchmarkRecipe(StrictModel):
    sources: list[str] = Field(default_factory=list)
    data_method: str | None = None


class DomainRecipe(StrictModel):
    benchmarks: dict[str, BenchmarkRecipe]


class Recipe(StrictModel):
    schema_version: Literal[1] = 1
    recipe_id: str
    domains: dict[str, DomainRecipe]
    domain_mix: dict[str, float]
    benchmark_mix: dict[str, dict[str, float]]
    multi_domain_mode: Literal["interleaved", "sequential", "staged_warmup"] = "interleaved"

    @field_validator("recipe_id")
    @classmethod
    def safe_recipe_id(cls, value: str) -> str:
        value = value.strip()
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("recipe_id must be a safe non-empty identifier")
        return value
