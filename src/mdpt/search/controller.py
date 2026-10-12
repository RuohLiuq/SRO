from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from mdpt.discovery.schema import (
    BenchmarkRecipe,
    DataMethodSpec,
    DomainRecipe,
    Recipe,
    SearchSpace,
    SourceCatalog,
)
from mdpt.discovery.validation import validate_recipe
from mdpt.io import read_json, write_json
from mdpt.schema import TaskConfig
from mdpt.search.acquisition import SurrogateAcquisition
from mdpt.search.featurizer import RecipeFeaturizer
from mdpt.search.pareto_archive import DefaultParetoArchive, TrialResult
from mdpt.search.proposal import Candidate, ProposalGenerator


SFT_SEARCH_SPACE: dict[str, list[float | int]] = {
    "learning_rate": [1e-4, 2e-4],
    "epochs": [2, 3],
    "lora_rank": [16, 64],
}
SEED_SFT_HYPERPARAMETERS: dict[str, float | int] = {
    "learning_rate": 2e-4,
    "epochs": 3,
    "lora_rank": 64,
}


def _normalized_ratio(value: dict[str, float]) -> dict[str, float]:
    total = sum(float(weight) for weight in value.values())
    if total <= 0:
        return {key: 0.0 for key in sorted(value)}
    return {key: round(float(value[key]) / total, 12) for key in sorted(value)}


def _candidate_decision(candidate: Candidate) -> dict[str, Any]:
    recipe = candidate.recipe
    domains: dict[str, Any] = {}
    for domain_id in sorted(recipe.domains):
        domains[domain_id] = {
            "benchmarks": {
                benchmark_id: {
                    "sources": sorted(set(benchmark.sources)),
                    "data_method": benchmark.data_method,
                }
                for benchmark_id, benchmark in sorted(
                    recipe.domains[domain_id].benchmarks.items()
                )
            }
        }
    return {
        "domains": domains,
        "domain_mix": _normalized_ratio(recipe.domain_mix),
        "benchmark_mix": {
            domain_id: _normalized_ratio(recipe.benchmark_mix[domain_id])
            for domain_id in sorted(recipe.benchmark_mix)
        },
        "multi_domain_mode": recipe.multi_domain_mode,
        "sft_hyperparameters": {
            name: round(float(value), 12)
            for name, value in sorted(candidate.sft_hyperparameters.items())
        },
    }


def _candidate_key(candidate: Candidate) -> str:
    return hashlib.sha256(
        json.dumps(
            _candidate_decision(candidate),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _equal_ratio_choice(
    choices: list[dict[str, float]],
    expected_ids: list[str],
    *,
    label: str,
) -> dict[str, float]:
    expected = set(expected_ids)
    target = 1.0 / len(expected_ids)
    for choice in choices:
        if set(choice) != expected:
            continue
        total = sum(float(value) for value in choice.values())
        if total > 0 and all(abs(float(choice[key]) / total - target) < 1e-9 for key in expected_ids):
            return dict(choice)
    raise ValueError(f"accepted SearchSpace does not contain an equal {label} choice")


def build_framework_seed(
    task: TaskConfig,
    catalog: SourceCatalog,
    methods: dict[str, DataMethodSpec],
    space: SearchSpace,
) -> Recipe:
    if "interleaved" not in space.multi_domain_modes:
        raise ValueError("accepted SearchSpace must allow interleaved mode for the framework seed")

    domain_ids = [domain.domain_id for domain in task.domains]
    recipe = Recipe(
        recipe_id="framework_base_source_seed",
        domains={
            domain.domain_id: DomainRecipe(
                benchmarks={
                    benchmark.benchmark_id: BenchmarkRecipe(
                        sources=list(
                            space.domains[domain.domain_id]
                            .benchmarks[benchmark.benchmark_id]
                            .source_choices
                        ),
                        data_method=None,
                    )
                    for benchmark in domain.benchmarks
                }
            )
            for domain in task.domains
        },
        domain_mix=_equal_ratio_choice(
            space.domain_mix_choices,
            domain_ids,
            label="domain mix",
        ),
        benchmark_mix={
            domain.domain_id: _equal_ratio_choice(
                space.benchmark_mix_choices[domain.domain_id],
                [benchmark.benchmark_id for benchmark in domain.benchmarks],
                label=f"benchmark mix for {domain.domain_id}",
            )
            for domain in task.domains
        },
        multi_domain_mode="interleaved",
    )
    validate_recipe(task, catalog, methods, space, recipe, allow_correction=False)
    return recipe


class Controller:
    def __init__(
        self,
        *,
        task: TaskConfig,
        space: SearchSpace,
        catalog: SourceCatalog,
        methods: dict[str, DataMethodSpec],
        proposal: ProposalGenerator,
        runner: Any,
        run_dir: Path,
        references: dict[str, Any],
        hp_space: dict[str, list[float | int]] | None = None,
        candidate_count: int = 6,
        candidate_pool_size: int = 2,
        cold_start_observations: int = 5,
        mc_samples: int = 64,
    ) -> None:
        self.task = task
        self.space = space
        self.catalog = catalog
        self.methods = methods
        self.proposal = proposal
        self.runner = runner
        self.run_dir = run_dir
        self.references = references
        self.hp_space = hp_space or SFT_SEARCH_SPACE
        if set(self.hp_space) != set(SEED_SFT_HYPERPARAMETERS):
            raise ValueError("Search policy hyperparameter keys do not match the framework seed")
        for name, value in SEED_SFT_HYPERPARAMETERS.items():
            if value not in self.hp_space[name]:
                raise ValueError(
                    f"framework seed hyperparameter {name}={value!r} is outside Search policy"
                )
        self.candidate_count = candidate_count
        self.candidate_pool_size = candidate_pool_size
        self.archive = DefaultParetoArchive()
        self.history: list[dict[str, Any]] = []
        self.candidate_logs: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []
        self.pending: dict[str, dict[str, Any]] = {}
        self.selected_keys: set[str] = set()
        self.featurizer = RecipeFeaturizer(
            task=task,
            space=space,
            hp_names=list(self.hp_space),
        )
        self.acquisition = SurrogateAcquisition(
            featurizer=self.featurizer,
            objective_names=sorted(domain.domain_id for domain in task.domains),
            min_fit=cold_start_observations,
            mc_samples=mc_samples,
        )
        self.baseline: dict[str, float] = {}

    def _model_references(self) -> tuple[dict[str, float], dict[str, float]]:
        models = self.references.get("models", {})
        model = models.get(self.task.base_model)
        if not isinstance(model, dict):
            raise ValueError(f"no benchmark references registered for {self.task.base_model!r}")
        base = {key: float(value) for key, value in (model.get("base_scores") or {}).items()}
        instruction = {key: float(value) for key, value in (model.get("scores") or {}).items()}
        missing = set(self.task.benchmark_ids) - set(base) | (set(self.task.benchmark_ids) - set(instruction))
        if missing:
            raise ValueError(f"benchmark references are incomplete: {sorted(missing)}")
        return base, instruction

    def _objectives(self, metrics: dict[str, float]) -> dict[str, float]:
        base, instruction = self._model_references()
        objectives: dict[str, float] = {}
        for domain in self.task.domains:
            gains: list[float] = []
            for benchmark in domain.benchmarks:
                benchmark_id = benchmark.benchmark_id
                denominator = instruction[benchmark_id] - base[benchmark_id]
                if denominator <= 0:
                    raise ValueError(f"invalid benchmark references for {benchmark_id!r}")
                gains.append((float(metrics[benchmark_id]) - base[benchmark_id]) / denominator)
            objectives[domain.domain_id] = sum(gains) / len(gains)
        return objectives

    def _domain_raw_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return {
            domain.domain_id: sum(
                float(metrics[benchmark.benchmark_id]) for benchmark in domain.benchmarks
            )
            / len(domain.benchmarks)
            for domain in self.task.domains
            if all(benchmark.benchmark_id in metrics for benchmark in domain.benchmarks)
        }

    def _raw_mean(self, metrics: dict[str, float]) -> float:
        return sum(float(metrics[benchmark_id]) for benchmark_id in self.task.benchmark_ids) / len(
            self.task.benchmark_ids
        )

    def _feedback(self, candidate: Candidate, result: TrialResult, round_index: int) -> dict[str, Any]:
        gain_vs_baseline = {
            benchmark_id: float(result.metrics[benchmark_id]) - float(self.baseline[benchmark_id])
            for benchmark_id in self.task.benchmark_ids
            if benchmark_id in result.metrics and benchmark_id in self.baseline
        }
        payload = {
            "round": round_index,
            "recipe": candidate.recipe.model_dump(mode="json"),
            "sft_hyperparameters": candidate.sft_hyperparameters,
            "llm_prior": candidate.llm_prior,
            "metrics": result.metrics,
            "gain_vs_baseline": gain_vs_baseline,
            "domain_raw_metrics": self._domain_raw_metrics(result.metrics),
            "objectives": result.objectives,
            "feasible": result.feasible,
            "checkpoint_path": result.checkpoint_path,
            "error": result.error,
            "metadata": result.metadata,
        }
        if result.feasible and all(benchmark_id in result.metrics for benchmark_id in self.task.benchmark_ids):
            payload["raw_benchmark_mean"] = self._raw_mean(result.metrics)
        return payload

    def _record(self, candidate: Candidate, result: TrialResult, round_index: int) -> None:
        if result.feasible:
            result.objectives = self._objectives(result.metrics)
        self.archive.update(candidate.recipe, candidate.sft_hyperparameters, result)
        self.history.append(self._feedback(candidate, result, round_index))
        self._write_progress(round_index)

    def _write_progress(self, round_index: int) -> None:
        write_json(self.run_dir / "feedback_history.json", self.history)
        write_json(self.run_dir / "candidate_plans.json", self.candidate_logs)
        write_json(
            self.run_dir / "checkpoint_pool.json",
            [
                {
                    "recipe_id": item["recipe"]["recipe_id"],
                    "checkpoint_path": item.get("checkpoint_path"),
                    "metrics": item.get("metrics"),
                }
                for item in self.history
                if item.get("feasible") and item.get("checkpoint_path")
            ],
        )
        write_json(
            self.run_dir / "controller_log.json",
            {
                "round": round_index,
                "n_trials": len(self.history),
                "events": self.log,
                "pending_candidates": [
                    {
                        "candidate_key": key,
                        "origin_round": entry["origin_round"],
                        "last_acquisition_score": entry.get("last_acquisition_score"),
                        "last_acquisition_rank": entry.get("last_acquisition_rank"),
                        "candidate": entry["candidate"].model_dump(mode="json"),
                    }
                    for key, entry in self.pending.items()
                ],
                "selected_candidate_keys": sorted(self.selected_keys),
                "pareto_front": [
                    entry.result.model_dump(mode="json")
                    for entry in self.archive.front()
                    if entry.result
                ],
            },
        )

    @staticmethod
    def _benchmark_operator(benchmark: dict[str, Any]) -> str:
        sources = benchmark.get("sources") or []
        method = benchmark.get("data_method")
        if method and sources:
            return f"existing+generated:{method}"
        if method:
            return f"generated-only:{method}"
        return "existing-only"

    def _search_context(self) -> dict[str, Any]:
        best_by_domain: dict[str, dict[str, Any]] = {}
        benchmark_to_domain = {
            benchmark.benchmark_id: domain.domain_id
            for domain in self.task.domains
            for benchmark in domain.benchmarks
        }
        operator_stats: dict[str, dict[str, dict[str, Any]]] = {}
        for domain in self.task.domains:
            for benchmark in domain.benchmarks:
                benchmark_id = benchmark.benchmark_id
                benchmark_space = self.space.domains[domain.domain_id].benchmarks[benchmark_id]
                operators: dict[str, dict[str, Any]] = {
                    "existing-only": {
                        "proposed": 0,
                        "evaluated": 0,
                        "best_domain_objective": None,
                    }
                }
                for method_id in benchmark_space.method_choices:
                    operators[f"generated-only:{method_id}"] = {
                        "proposed": 0,
                        "evaluated": 0,
                        "best_domain_objective": None,
                    }
                    operators[f"existing+generated:{method_id}"] = {
                        "proposed": 0,
                        "evaluated": 0,
                        "best_domain_objective": None,
                    }
                operator_stats[benchmark_id] = operators

        def observe_recipe(recipe: dict[str, Any], *, evaluated: bool, item: dict[str, Any]) -> None:
            for domain in recipe.get("domains", {}).values():
                for benchmark_id, benchmark in domain.get("benchmarks", {}).items():
                    operator = self._benchmark_operator(benchmark)
                    stats = operator_stats.setdefault(benchmark_id, {}).setdefault(
                        operator,
                        {"proposed": 0, "evaluated": 0, "best_domain_objective": None},
                    )
                    stats["evaluated" if evaluated else "proposed"] += 1
                    if evaluated and item.get("feasible"):
                        domain_id = benchmark_to_domain[benchmark_id]
                        value = item.get("objectives", {}).get(domain_id)
                        if value is not None:
                            previous = stats["best_domain_objective"]
                            stats["best_domain_objective"] = (
                                float(value)
                                if previous is None
                                else max(float(previous), float(value))
                            )

        for round_log in self.candidate_logs:
            for ranked in round_log.get("ranked", []):
                if ranked.get("origin_round") != round_log.get("round"):
                    continue
                candidate = ranked.get("candidate") or {}
                observe_recipe(candidate.get("recipe") or {}, evaluated=False, item={})
        for item in self.history:
            recipe = item.get("recipe") or {}
            observe_recipe(recipe, evaluated=True, item=item)
            if not item.get("feasible"):
                continue
            for domain_id, value in item.get("objectives", {}).items():
                current = best_by_domain.get(domain_id)
                if current is None or float(value) > float(current["objective"]):
                    best_by_domain[domain_id] = {
                        "objective": float(value),
                        "recipe": recipe,
                        "sft_hyperparameters": item.get("sft_hyperparameters") or {},
                        "metrics": item.get("metrics") or {},
                        "objectives": item.get("objectives") or {},
                    }
        return {
            "per_domain_best_observed": best_by_domain,
            "per_benchmark_operator_statistics": operator_stats,
            "pareto_front": [
                {
                    "recipe": entry.recipe.model_dump(mode="json"),
                    "sft_hyperparameters": entry.sft_hyperparameters,
                    "objectives": entry.result.objectives,
                    "metrics": entry.result.metrics,
                }
                for entry in self.archive.front()
                if entry.result
            ],
            "recently_proposed_but_not_evaluated": [
                {
                    "candidate": entry["candidate"].model_dump(mode="json"),
                    "origin_round": entry["origin_round"],
                    "last_acquisition_score": entry.get("last_acquisition_score"),
                    "last_acquisition_rank": entry.get("last_acquisition_rank"),
                }
                for entry in list(self.pending.values())[:8]
            ],
        }

    def _correction_known_unavailable(self, candidate: Candidate) -> bool:
        latest_checkpoint = next(
            (
                str(item["checkpoint_path"])
                for item in reversed(self.history)
                if item.get("feasible") and item.get("checkpoint_path")
            ),
            None,
        )
        if latest_checkpoint is None:
            return any(
                benchmark.data_method
                and self.methods[benchmark.data_method].operation == "correction"
                for domain in candidate.recipe.domains.values()
                for benchmark in domain.benchmarks.values()
            )
        index_path = self.run_dir / "diagnostics" / "index.json"
        entries = read_json(index_path).get("entries", []) if index_path.is_file() else []
        for domain in candidate.recipe.domains.values():
            for benchmark in domain.benchmarks.values():
                if not benchmark.data_method:
                    continue
                method = self.methods[benchmark.data_method]
                if method.operation != "correction":
                    continue
                latest = next(
                    (
                        entry
                        for entry in reversed(entries)
                        if entry.get("benchmark_id") == method.benchmark_id
                        and entry.get("diagnostic_source_id") == method.diagnostic_source_id
                        and entry.get("checkpoint_path") == latest_checkpoint
                    ),
                    None,
                )
                if latest is not None and (
                    latest.get("status") == "failed"
                    or not latest.get("usable", int(latest.get("n_wrong", 0)) > 0)
                ):
                    return True
        return False

    def _score_candidates(self, candidates: list[Candidate]) -> list[tuple[Candidate, dict[str, Any]]]:
        candidates = [candidate for candidate in candidates if not self._correction_known_unavailable(candidate)]
        self.acquisition.fit(self.archive)
        scored = [
            (candidate, self.acquisition.score_details(candidate, self.archive))
            for candidate in candidates
        ]
        return sorted(scored, key=lambda item: -float(item[1]["score"]))

    def _execute(
        self,
        candidate: Candidate,
        *,
        round_index: int,
        search_deadline_monotonic: float,
    ) -> TrialResult:
        key = _candidate_key(candidate)
        self.selected_keys.add(key)
        runtime_candidate = candidate.model_copy(
            update={
                "recipe": candidate.recipe.model_copy(
                    update={"recipe_id": f"round{round_index}_{candidate.recipe.recipe_id}"}
                )
            }
        )
        latest_checkpoint = next(
            (
                str(item["checkpoint_path"])
                for item in reversed(self.history)
                if item.get("feasible") and item.get("checkpoint_path")
            ),
            None,
        )
        result = self.runner.run(
            recipe=runtime_candidate.recipe,
            sft_hyperparameters=runtime_candidate.sft_hyperparameters,
            allow_correction=latest_checkpoint is not None,
            diagnostic_checkpoint=latest_checkpoint,
            search_deadline_monotonic=search_deadline_monotonic,
        )
        self._record(runtime_candidate, result, round_index)
        return result

    def run(self, *, max_rounds: int, max_total_hours: float) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        search_deadline = started + max_total_hours * 3600
        self.baseline = self.runner.run_baseline(deadline_monotonic=search_deadline)
        write_json(self.run_dir / "baseline.json", self.baseline)
        self.log.append({"event": "baseline_completed", "metrics": self.baseline})
        self._write_progress(0)
        if time.monotonic() >= search_deadline:
            raise TimeoutError("Search budget was exhausted by baseline evaluation")

        initial = Candidate(
            recipe=build_framework_seed(self.task, self.catalog, self.methods, self.space),
            sft_hyperparameters=dict(SEED_SFT_HYPERPARAMETERS),
            llm_prior={
                "expected_effect": "framework-owned baseline using all accepted base sources",
                "confidence": "high",
                "risk": "low",
            },
        )
        self.log.append(
            {"round": 0, "event": "framework_seed_started", "candidate": initial.model_dump(mode="json")}
        )
        self._write_progress(0)
        self._execute(
            initial,
            round_index=0,
            search_deadline_monotonic=search_deadline,
        )
        self.log.append({"round": 0, "event": "framework_seed_completed"})
        self._write_progress(0)

        for round_index in range(1, max_rounds + 1):
            if time.monotonic() >= search_deadline:
                self.log.append({"round": round_index, "event": "budget_exhausted"})
                self._write_progress(round_index)
                break
            pool_before = len(self.pending)
            proposed = self.proposal.generate(
                history=self.history,
                search_context=self._search_context(),
                count=self.candidate_count,
                correction_available=any(
                    item.get("feasible") and item.get("checkpoint_path") for item in self.history
                ),
                deadline_monotonic=search_deadline,
            )
            fresh: list[dict[str, Any]] = []
            seen_fresh: set[str] = set()
            duplicates: list[dict[str, str]] = []
            for candidate in proposed:
                key = _candidate_key(candidate)
                if key in seen_fresh:
                    duplicates.append({"recipe_id": candidate.recipe.recipe_id, "reason": "same_round"})
                    continue
                seen_fresh.add(key)
                if key in self.selected_keys:
                    duplicates.append({"recipe_id": candidate.recipe.recipe_id, "reason": "already_selected"})
                    continue
                fresh.append({"candidate": candidate, "origin_round": round_index})
            reusable = [
                entry
                for key, entry in self.pending.items()
                if key not in seen_fresh and key not in self.selected_keys
            ]
            entries = fresh + reusable
            ranked = self._score_candidates([entry["candidate"] for entry in entries])
            entry_by_key = {_candidate_key(entry["candidate"]): entry for entry in entries}
            if not ranked:
                self.candidate_logs.append(
                    {
                        "round": round_index,
                        "generation": self.proposal.last_generation,
                        "ranked": [],
                        "duplicates": duplicates,
                        "chosen": None,
                    }
                )
                self.log.append({"round": round_index, "event": "no_valid_candidates"})
                self._write_progress(round_index)
                break

            ranked_log: list[dict[str, Any]] = []
            for rank, (candidate, acquisition) in enumerate(ranked, start=1):
                entry = entry_by_key[_candidate_key(candidate)]
                entry["last_acquisition_score"] = float(acquisition["score"])
                entry["last_acquisition_rank"] = rank
                ranked_log.append(
                    {
                        "candidate": candidate.model_dump(mode="json"),
                        "origin_round": entry["origin_round"],
                        "acquisition_rank": rank,
                        "acquisition": acquisition,
                    }
                )

            chosen, chosen_details = ranked[0]
            chosen_key = _candidate_key(chosen)
            chosen_entry = entry_by_key[chosen_key]
            self.selected_keys.add(chosen_key)
            retained: dict[str, dict[str, Any]] = {}
            for candidate, _ in ranked[1:]:
                if len(retained) >= self.candidate_pool_size:
                    break
                key = _candidate_key(candidate)
                retained[key] = entry_by_key[key]
            self.pending = retained
            round_log = {
                "round": round_index,
                "generation": self.proposal.last_generation,
                "ranked": ranked_log,
                "duplicates": duplicates,
                "candidate_pool": {
                    "size_before": pool_before,
                    "new_candidates": len(fresh),
                    "reused_candidates": len(reusable),
                    "size_after": len(self.pending),
                    "retained": [
                        {
                            "recipe_id": entry["candidate"].recipe.recipe_id,
                            "origin_round": entry["origin_round"],
                            "acquisition_score": entry.get("last_acquisition_score"),
                            "acquisition_rank": entry.get("last_acquisition_rank"),
                        }
                        for entry in self.pending.values()
                    ],
                },
                "chosen": {
                    "candidate_key": chosen_key,
                    "origin_round": chosen_entry["origin_round"],
                    "acquisition_rank": 1,
                    "acquisition": chosen_details,
                    "candidate": chosen.model_dump(mode="json"),
                },
            }
            self.candidate_logs.append(round_log)
            self.log.append(
                {
                    "round": round_index,
                    "event": "candidate_selected",
                    "recipe_id": chosen.recipe.recipe_id,
                    "candidate_key": chosen_key,
                    "origin_round": chosen_entry["origin_round"],
                    "acquisition": chosen_details,
                }
            )
            self._write_progress(round_index)
            if time.monotonic() >= search_deadline:
                self.log.append({"round": round_index, "event": "budget_exhausted_before_trial"})
                self._write_progress(round_index)
                break
            self._execute(
                chosen,
                round_index=round_index,
                search_deadline_monotonic=search_deadline,
            )
            self.log.append(
                {"round": round_index, "event": "trial_recorded", "recipe_id": chosen.recipe.recipe_id}
            )
            self._write_progress(round_index)

        feasible = [
            item
            for item in self.history
            if item.get("feasible")
            and item.get("checkpoint_path")
            and all(benchmark_id in item.get("metrics", {}) for benchmark_id in self.task.benchmark_ids)
        ]
        if not feasible:
            raise RuntimeError("Search completed without a feasible checkpoint")
        winner = max(feasible, key=lambda item: float(item["raw_benchmark_mean"]))
        self.log.append(
            {
                "event": "final_selection",
                "recipe_id": winner["recipe"]["recipe_id"],
                "raw_benchmark_mean": winner["raw_benchmark_mean"],
            }
        )
        self._write_progress(max_rounds)
        output = {
            "status": "completed",
            "n_trials": len(self.history),
            "baseline": self.baseline,
            "winner": winner,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(self.run_dir / "final_selection.json", winner)
        return output
