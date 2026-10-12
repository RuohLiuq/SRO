from __future__ import annotations

import json
import math
import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mdpt.discovery.schema import DataMethodSpec, Recipe, SearchSpace, SourceCatalog
from mdpt.discovery.validation import validate_recipe
from mdpt.schema import TaskConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Candidate(_StrictModel):
    recipe: Recipe
    sft_hyperparameters: dict[str, float | int]
    llm_prior: dict[str, str] = Field(default_factory=dict)


class CandidateList(_StrictModel):
    candidates: list[Candidate]


def _extract_json(text: str) -> Any:
    value = text.strip()
    if not value:
        raise ValueError("Planner response is empty")
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    starts = [index for index in (value.find("{"), value.find("[")) if index >= 0]
    if not starts:
        raise ValueError("Planner response does not contain JSON")
    start = min(starts)
    end = max(value.rfind("}"), value.rfind("]"))
    if end < start:
        raise ValueError("Planner response contains incomplete JSON")
    return json.loads(value[start : end + 1])


def _method_prompt_view(method: DataMethodSpec) -> dict[str, str]:
    return {
        "method_id": method.method_id,
        "operation": method.operation,
        "description": method.description,
    }


def _space_view(
    task: TaskConfig,
    catalog: SourceCatalog,
    methods: dict[str, DataMethodSpec],
    space: SearchSpace,
    hp_space: dict[str, list[float | int]],
) -> dict[str, Any]:
    sources = catalog.by_id()
    return {
        "domains": [
            {
                "domain_id": domain.domain_id,
                "benchmarks": [benchmark.benchmark_id for benchmark in domain.benchmarks],
            }
            for domain in task.domains
        ],
        "search_space": {
            "domains": {
                domain_id: {
                    "benchmarks": {
                        benchmark_id: {
                            "sources": [
                                {
                                    "source_id": source_id,
                                    "usage": sources[source_id].usage,
                                    "metadata": sources[source_id].metadata,
                                }
                                for source_id in benchmark_space.source_choices
                            ],
                            "methods": [
                                _method_prompt_view(methods[method_id])
                                for method_id in benchmark_space.method_choices
                            ],
                        }
                        for benchmark_id, benchmark_space in domain_space.benchmarks.items()
                    }
                }
                for domain_id, domain_space in space.domains.items()
            },
            "domain_mix_choices": space.domain_mix_choices,
            "benchmark_mix_choices": space.benchmark_mix_choices,
            "multi_domain_modes": space.multi_domain_modes,
            "sft_hyperparameters": hp_space,
        },
    }


def _weakest_skill_summary(
    metrics: dict[str, Any],
    *,
    weakest_k: int = 5,
    min_total: int = 20,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key, value in metrics.items():
        key_s = str(key)
        if ":skill_accuracy." in key_s:
            benchmark_id, skill = key_s.split(":skill_accuracy.", 1)
            total = metrics.get(f"{benchmark_id}:skill_total.{skill}")
        elif key_s.startswith("skill_accuracy."):
            skill = key_s[len("skill_accuracy.") :]
            total = metrics.get(f"skill_total.{skill}")
        else:
            continue
        row: dict[str, Any] = {"skill": skill, "accuracy": round(float(value), 6)}
        if isinstance(total, (int, float)):
            row["total"] = int(total)
        rows.append(row)
    supported = [
        row
        for row in rows
        if "total" in row and int(row["total"]) >= min_total
    ]
    ranked = sorted(
        supported,
        key=lambda row: (row["accuracy"], -int(row["total"])),
    )
    return {"weakest_skills": ranked[:weakest_k]} if ranked else {}


def _history_view(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    view: list[dict[str, Any]] = []
    for item in history[-12:]:
        metrics = item.get("metrics") or {}
        view.append(
            {
                "round": item.get("round"),
                "recipe": item.get("recipe"),
                "sft_hyperparameters": item.get("sft_hyperparameters") or {},
                "llm_prior": item.get("llm_prior") or {},
                "metrics": metrics,
                "validation_summary": _weakest_skill_summary(metrics),
                "gain_vs_baseline": item.get("gain_vs_baseline") or {},
                "domain_raw_metrics": item.get("domain_raw_metrics") or {},
                "objectives": item.get("objectives") or {},
                "raw_benchmark_mean": item.get("raw_benchmark_mean"),
                "feasible": item.get("feasible", False),
                "error": item.get("error"),
                "data_summary": (item.get("metadata") or {}).get("data") or {},
                "diagnostics": (item.get("metadata") or {}).get("diagnostics") or {},
            }
        )
    return view


class ProposalGenerator:
    def __init__(
        self,
        *,
        task: TaskConfig,
        catalog: SourceCatalog,
        methods: dict[str, DataMethodSpec],
        space: SearchSpace,
        hp_space: dict[str, list[float | int]],
        model: str,
        base_url: str,
        api_key: str,
        transport_attempts: int = 5,
        proposal_attempts: int = 3,
    ) -> None:
        self.task = task
        self.catalog = catalog
        self.methods = methods
        self.space = space
        self.hp_space = hp_space
        self.transport_attempts = transport_attempts
        self.proposal_attempts = proposal_attempts
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=1)
        self.model = model
        self.last_generation: dict[str, Any] = {}
        self._transport_retries: list[dict[str, Any]] = []

    def _default_hyperparameters(self, *, generated: bool = False) -> dict[str, float | int]:
        preferred: dict[str, float | int] = {
            "learning_rate": 2e-4,
            "epochs": 2 if generated else 3,
            "lora_rank": 64,
        }
        return {
            name: preferred[name] if preferred.get(name) in choices else choices[0]
            for name, choices in self.hp_space.items()
        }

    def _example_recipe(self, recipe_id: str) -> dict[str, Any]:
        domains: dict[str, Any] = {}
        for domain in self.task.domains:
            domain_space = self.space.domains[domain.domain_id]
            domains[domain.domain_id] = {
                "benchmarks": {
                    benchmark.benchmark_id: {
                        "sources": list(
                            domain_space.benchmarks[benchmark.benchmark_id].source_choices
                        ),
                        "data_method": None,
                    }
                    for benchmark in domain.benchmarks
                }
            }
        return {
            "schema_version": 1,
            "recipe_id": recipe_id,
            "domains": domains,
            "domain_mix": self.space.domain_mix_choices[0],
            "benchmark_mix": {
                domain.domain_id: self.space.benchmark_mix_choices[domain.domain_id][0]
                for domain in self.task.domains
            },
            "multi_domain_mode": (
                "interleaved"
                if "interleaved" in self.space.multi_domain_modes
                else self.space.multi_domain_modes[0]
            ),
        }

    def _examples(self) -> dict[str, Any]:
        existing_recipe = self._example_recipe("existing_sources_example")
        examples = [
            {
                "recipe": existing_recipe,
                "sft_hyperparameters": self._default_hyperparameters(),
                "llm_prior": {
                    "expected_effect": "baseline multi-domain SFT with accepted existing sources",
                    "confidence": "medium",
                    "risk": "low",
                },
            }
        ]
        generated_recipe = self._example_recipe("rewrite_only_example")
        selected_rewrite: str | None = None
        selected_benchmark: str | None = None
        for domain in self.task.domains:
            for benchmark in domain.benchmarks:
                benchmark_space = self.space.domains[domain.domain_id].benchmarks[
                    benchmark.benchmark_id
                ]
                selected_rewrite = next(
                    (
                        method_id
                        for method_id in benchmark_space.method_choices
                        if self.methods[method_id].operation == "rewrite"
                    ),
                    None,
                )
                if selected_rewrite:
                    selected_benchmark = benchmark.benchmark_id
                    generated_recipe["domains"][domain.domain_id]["benchmarks"][selected_benchmark] = {
                        "sources": [],
                        "data_method": selected_rewrite,
                    }
                    break
            if selected_rewrite:
                break
        if selected_rewrite and selected_benchmark:
            examples.append(
                {
                    "recipe": generated_recipe,
                    "sft_hyperparameters": self._default_hyperparameters(generated=True),
                    "llm_prior": {
                        "expected_effect": (
                            f"controlled Rewrite-only test for {selected_benchmark}; other benchmarks "
                            "retain their selected existing sources"
                        ),
                        "confidence": "medium",
                        "risk": "medium",
                    },
                }
            )
        return {"candidates": examples}

    def _schema_shape(self) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "recipe": {
                        "schema_version": 1,
                        "recipe_id": "short_unique_id",
                        "domains": {
                            "domain_id": {
                                "benchmarks": {
                                    "benchmark_id": {
                                        "sources": ["zero_or_more_declared_source_ids"],
                                        "data_method": "null_or_one_declared_method_id",
                                    }
                                }
                            }
                        },
                        "domain_mix": "one object from domain_mix_choices",
                        "benchmark_mix": {
                            "domain_id": "one object from benchmark_mix_choices[domain_id]"
                        },
                        "multi_domain_mode": "one declared mode",
                    },
                    "sft_hyperparameters": "one declared value for every SFT hyperparameter",
                    "llm_prior": {
                        "expected_effect": "evidence-based statement",
                        "confidence": "low|medium|high",
                        "risk": "low|medium|high",
                    },
                }
            ]
        }

    def _prompt(
        self,
        *,
        history: list[dict[str, Any]],
        search_context: dict[str, Any],
        count: int,
    ) -> str:
        return (
            "You are proposing SFT candidates for a multi-domain post-training search.\n"
            "Each candidate must be self-contained and use only current searchable choices.\n\n"
            "## Current task and searchable choices\n"
            f"{json.dumps(_space_view(self.task, self.catalog, self.methods, self.space, self.hp_space), ensure_ascii=False, indent=2)}\n\n"
            "## Long-term search summary\n"
            f"{json.dumps(search_context, ensure_ascii=False, indent=2)}\n\n"
            "The per-domain best entries are component evidence, not automatically the best complete Recipe. "
            "Use them and the Pareto front to propose promising cross-domain combinations. "
            "Candidates listed as proposed-but-not-evaluated have no measured outcome and must not be treated as failures. "
            "Use per-benchmark operator statistics to distinguish not-yet-evaluated strategies from evaluated-and-weak strategies.\n\n"
            "## Candidate-set requirements\n"
            "Treat all returned candidates as one candidate set and make them meaningfully diverse from one another. "
            "The set must include exploitation of observed evidence, recombination of promising components, and exploration of under-evaluated strategies. "
            "Do not return minor variants that repeatedly reuse the same per-benchmark data operator. "
            "Infer existing-only, Rewrite-only, Correction-only, or existing+generated preferences independently for each benchmark from empirical evidence. "
            "When source-mode evidence is incomplete or confounded, include controlled ablations that change one domain or benchmark decision at a time.\n\n"
            "## Historical trial feedback\n"
            f"{json.dumps(_history_view(history), ensure_ascii=False, indent=2)}\n\n"
            "## Hard rules\n"
            "- Preserve the exact domain -> benchmark nesting from the task.\n"
            "- Every domain and benchmark must appear exactly once in every Recipe.\n"
            "- Choose exactly one declared domain_mix and one declared benchmark_mix for every domain.\n"
            "- Choose sources and at most one Rewrite/Correction data_method independently per benchmark.\n"
            "- When at least one usable data method exists, every candidate set must include at least one generated-only candidate with sources=[] and a non-null data_method.\n"
            "- If data_method is selected, its generated source is always included in that benchmark's final training data.\n"
            "- sources lists only additional existing sources; sources=[] with non-null data_method means generated-only.\n"
            "- The system never adds base sources automatically; add them only to deliberately test existing+generated mixing.\n"
            "- A benchmark with data_method=null must select at least one existing source.\n"
            "- Do not invent source IDs, method IDs, paths, generated source IDs, runtime fields, or generation sizes.\n"
            "- Every candidate must have a unique recipe_id.\n"
            "- Return only fields in the required schema.\n\n"
            "## Required JSON schema\n"
            f"{json.dumps(self._schema_shape(), ensure_ascii=False, indent=2)}\n\n"
            "## Example response format\n"
            f"{json.dumps(self._examples(), ensure_ascii=False, indent=2)}\n\n"
            f"Return STRICT JSON with up to {count} candidates. No prose and no markdown fences."
        )

    @staticmethod
    def _is_retryable_transport_error(exc: BaseException) -> bool:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status in {408, 409, 425, 429} or status >= 500
        return type(exc).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
        }

    def _request(
        self,
        messages: list[dict[str, str]],
        *,
        deadline_monotonic: float | None = None,
    ) -> str:
        for attempt in range(1, self.transport_attempts + 1):
            remaining = (
                deadline_monotonic - time.monotonic()
                if deadline_monotonic is not None
                else None
            )
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Planner request exceeded the Search deadline")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=8192,
                    timeout=min(300.0, remaining) if remaining is not None else 300.0,
                )
                choice = response.choices[0]
                if getattr(choice, "finish_reason", None) == "length":
                    raise ValueError("Planner response was truncated at max_tokens")
                content = str(choice.message.content or "")
                if not content.strip():
                    raise ValueError("Planner response is empty")
                return content
            except Exception as exc:
                if attempt >= self.transport_attempts or not self._is_retryable_transport_error(exc):
                    raise
                delay = min(60.0, 5.0 * (2 ** (attempt - 1)))
                self._transport_retries.append(
                    {
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}"[:2000],
                        "sleep_seconds": delay,
                    }
                )
                if deadline_monotonic is not None:
                    delay = min(delay, max(0.0, deadline_monotonic - time.monotonic()))
                if delay <= 0:
                    raise TimeoutError("Planner retry exceeded the Search deadline") from exc
                time.sleep(delay)
        raise AssertionError("unreachable transport retry loop")

    def _valid_candidates(
        self,
        payload: CandidateList,
        *,
        correction_available: bool,
        count: int,
    ) -> tuple[list[Candidate], list[str]]:
        valid: list[Candidate] = []
        errors: list[str] = []
        hp_allowed = {key: {float(value) for value in values} for key, values in self.hp_space.items()}
        required_hp = set(hp_allowed)
        for candidate in payload.candidates:
            try:
                ratios = [candidate.recipe.domain_mix, *candidate.recipe.benchmark_mix.values()]
                if any(
                    not ratio
                    or any(
                        not math.isfinite(float(weight)) or float(weight) <= 0
                        for weight in ratio.values()
                    )
                    for ratio in ratios
                ):
                    raise ValueError("all domain_mix and benchmark_mix weights must be finite and positive")
                validate_recipe(
                    self.task,
                    self.catalog,
                    self.methods,
                    self.space,
                    candidate.recipe,
                    allow_correction=correction_available,
                )
                if set(candidate.sft_hyperparameters) != required_hp:
                    raise ValueError(f"SFT hyperparameters must be exactly {sorted(required_hp)}")
                for name, value in candidate.sft_hyperparameters.items():
                    if float(value) not in hp_allowed[name]:
                        raise ValueError(f"unsupported {name}: {value}")
                valid.append(candidate)
            except Exception as exc:
                errors.append(f"{candidate.recipe.recipe_id}: {exc}")
        return valid[:count], errors

    def generate(
        self,
        *,
        history: list[dict[str, Any]],
        search_context: dict[str, Any],
        count: int,
        correction_available: bool,
        deadline_monotonic: float | None = None,
    ) -> list[Candidate]:
        prompt = self._prompt(history=history, search_context=search_context, count=count)
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        attempts: list[dict[str, Any]] = []
        self._transport_retries = []
        self.last_generation = {
            "prompt": prompt,
            "attempts": attempts,
            "transport_retries": self._transport_retries,
        }
        for attempt in range(self.proposal_attempts):
            raw = self._request(messages, deadline_monotonic=deadline_monotonic)
            attempt_info: dict[str, Any] = {
                "attempt": attempt + 1,
                "raw_text": raw,
                "validation_error": None,
            }
            attempts.append(attempt_info)
            try:
                parsed = _extract_json(raw)
                attempt_info["parsed"] = parsed
                payload = CandidateList.model_validate(parsed)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError, ValidationError) as exc:
                error = f"{type(exc).__name__}: {exc}"[:4000]
                attempt_info["validation_error"] = error
                if attempt + 1 >= self.proposal_attempts:
                    raise
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your previous response failed required JSON parsing or schema validation.\n"
                                f"Validation error:\n{error}\n\n"
                                "Regenerate the complete candidate set from scratch. Follow the original "
                                "required schema exactly, including recipe_id for every candidate. "
                                "Return only strict JSON with no prose or markdown fences."
                            ),
                        },
                    ]
                )
                continue
            valid, errors = self._valid_candidates(
                payload,
                correction_available=correction_available,
                count=count,
            )
            attempt_info["semantic_errors"] = errors
            attempt_info["accepted"] = [candidate.model_dump(mode="json") for candidate in valid]
            self.last_generation["parsed"] = parsed
            self.last_generation["accepted"] = attempt_info["accepted"]
            return valid
        raise AssertionError("unreachable proposal retry loop")
