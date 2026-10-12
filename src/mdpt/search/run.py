from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mdpt.benchmarks import load_benchmark_metadata
from mdpt.config import EnvironmentConfig
from mdpt.discovery.handoff import verify_handoff
from mdpt.discovery.schema import DataMethodSpec, Recipe, SearchSpace, SourceCatalog
from mdpt.execution.eval_runner import EvalRunner
from mdpt.execution.recipe_runner import RecipeRunner
from mdpt.execution.training_runner import TRLTrainingRunner
from mdpt.io import read_json, write_json
from mdpt.schema import TaskConfig
from mdpt.search.controller import Controller, SFT_SEARCH_SPACE
from mdpt.search.proposal import ProposalGenerator


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    generation_samples_per_method: int = Field(default=500, gt=0)
    proposal_candidates_per_round: int = Field(default=6, gt=0)
    candidate_pool_size: int = Field(default=2, ge=0)
    cold_start_observations: int = Field(default=5, gt=0)
    mc_samples: int = Field(default=64, gt=0)
    learning_rate_choices: list[float] = Field(default_factory=lambda: [1e-4, 2e-4], min_length=1)
    epoch_choices: list[int] = Field(default_factory=lambda: [2, 3], min_length=1)
    lora_rank_choices: list[int] = Field(default_factory=lambda: [16, 64], min_length=1)
    transport_attempts: int = Field(default=5, gt=0)
    proposal_attempts: int = Field(default=3, gt=0)
    max_search_rounds: int = Field(default=10, gt=0)
    max_minutes_per_trial: int = Field(default=120, gt=0)

    @property
    def hp_space(self) -> dict[str, list[float | int]]:
        return {
            "learning_rate": list(self.learning_rate_choices),
            "epochs": list(self.epoch_choices),
            "lora_rank": list(self.lora_rank_choices),
        }


def _load_handoff(work_dir: Path) -> tuple[Path, TaskConfig, SourceCatalog, dict[str, DataMethodSpec], SearchSpace, Recipe, dict[str, Any]]:
    workspace = work_dir / "discovery" / "workspace"
    verify_handoff(workspace)
    handoff = workspace / "handoff"
    task = TaskConfig.model_validate(read_json(handoff / "task.json"))
    catalog = SourceCatalog.model_validate(read_json(handoff / "sources.json"))
    methods = {
        path.stem: DataMethodSpec.model_validate(read_json(path))
        for path in sorted((handoff / "methods").glob("*.json"))
    }
    space = SearchSpace.model_validate(read_json(handoff / "search_space.json"))
    initial_recipe = Recipe.model_validate(read_json(handoff / "initial_recipe.json"))
    runtime = read_json(handoff / "runtime.json")
    return workspace, task, catalog, methods, space, initial_recipe, runtime


def _discovery_elapsed_hours(work_dir: Path) -> float:
    run_path = work_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"run state does not exist: {run_path}")
    run_state = read_json(run_path)
    discovery = (run_state.get("stages") or {}).get("discovery") or {}
    elapsed = discovery.get("agent_elapsed_seconds")
    if elapsed is None:
        raise RuntimeError("run.json does not record Discovery active elapsed time")
    return float(elapsed) / 3600.0


def _update_run_stage(work_dir: Path, stage: str, values: dict[str, Any]) -> None:
    path = work_dir / "run.json"
    if not path.is_file():
        return
    payload = read_json(path)
    payload["active_stage"] = stage
    stages = payload.setdefault("stages", {})
    if values.get("status") == "running":
        stages[stage] = dict(values)
    else:
        stages.setdefault(stage, {}).update(values)
    write_json(path, payload)


def _install_augmentation_environment(environment: EnvironmentConfig) -> None:
    os.environ.update(
        {
            "MDPT_AUGMENTATION_MODEL": environment.augmentation_model,
            "MDPT_AUGMENTATION_BASE_URL": environment.augmentation_base_url,
            "MDPT_AUGMENTATION_API_KEY": environment.augmentation_api_key.get_secret_value(),
            "MDPT_AUGMENTATION_MAX_TOKENS": str(environment.augmentation_max_tokens),
            "MDPT_AUGMENTATION_TEMPERATURE": str(environment.augmentation_temperature),
            "MDPT_AUGMENTATION_TIMEOUT_SECONDS": str(environment.augmentation_timeout_seconds),
            "CUDA_VISIBLE_DEVICES": environment.cuda_visible_devices,
            "HF_HOME": environment.hf_home,
        }
    )


def run_search(
    *,
    work_dir: Path,
    environment: EnvironmentConfig,
    search_config_path: Path,
    max_hours: float,
    max_rounds: int | None = None,
) -> dict[str, Any]:
    work_dir = work_dir.expanduser().resolve()
    environment.require_search()
    workspace, task, catalog, methods, space, _initial_recipe_witness, runtime = _load_handoff(work_dir)
    config = SearchConfig.model_validate(read_json(search_config_path.expanduser().resolve()))
    visible_devices = [value.strip() for value in environment.cuda_visible_devices.split(",") if value.strip()]
    if len(visible_devices) != 1:
        raise ValueError("the released Search runtime requires exactly one CUDA device")
    _install_augmentation_environment(environment)

    elapsed_discovery = _discovery_elapsed_hours(work_dir)
    remaining_hours = max_hours - elapsed_discovery
    if remaining_hours <= 0:
        raise RuntimeError(
            f"Discovery used {elapsed_discovery:.3f}h of the {max_hours:.3f}h total budget"
        )
    rounds = max_rounds if max_rounds is not None else config.max_search_rounds
    if rounds <= 0:
        raise ValueError("max_rounds must be positive")

    search_dir = work_dir / "search"
    if search_dir.exists():
        raise FileExistsError(
            f"Search output already exists: {search_dir}; clean it explicitly or use a new work-dir"
        )
    search_dir.mkdir(parents=True)
    write_json(search_dir / "search_config.json", config.model_dump(mode="json"))
    started_at = time.time()
    search_status = {
        "stage": "search",
        "status": "running",
        "started_at": started_at,
        "total_budget_hours": max_hours,
        "discovery_elapsed_hours": elapsed_discovery,
        "search_budget_hours": remaining_hours,
        "max_rounds": rounds,
    }
    write_json(search_dir / "status.json", search_status)
    _update_run_stage(work_dir, "search", search_status)

    try:
        benchmark_metadata = load_benchmark_metadata(task)
        eval_args = {
            benchmark_id: [str(value) for value in metadata.get("search_eval_args", [])]
            for benchmark_id, metadata in benchmark_metadata.items()
        }
        for benchmark_id, metadata in benchmark_metadata.items():
            if metadata.get("requires_judge"):
                eval_args[benchmark_id].extend(
                    ["--judge-workers", str(environment.healthbench_judge_workers)]
                )
        requires_judge = any(
            metadata.get("requires_judge") for metadata in benchmark_metadata.values()
        )
        eval_runner = EvalRunner(
            extra_args=eval_args,
            environment=environment.evaluation_environment(healthbench=requires_judge),
            timeout_seconds=config.max_minutes_per_trial * 60,
        )
        training_runner = TRLTrainingRunner(
            base_model=task.base_model,
            output_root=search_dir / "ckpts",
            templates_root=Path(__file__).resolve().parents[1] / "assets" / "templates",
        )
        recipe_runner = RecipeRunner(
            task=task,
            space=space,
            catalog=catalog,
            methods=methods,
            workspace=workspace,
            run_dir=search_dir,
            runtime=runtime,
            training_runner=training_runner,
            eval_runner=eval_runner,
            generation_samples=config.generation_samples_per_method,
            max_trial_seconds=config.max_minutes_per_trial * 60,
        )
        references = read_json(
            Path(__file__).resolve().parents[1]
            / "assets"
            / "references"
            / "benchmark_references.json"
        )
        planner_key = environment.planner_api_key
        assert planner_key is not None and environment.planner_model and environment.planner_base_url
        proposal = ProposalGenerator(
            task=task,
            catalog=catalog,
            methods=methods,
            space=space,
            hp_space=config.hp_space,
            model=environment.planner_model,
            base_url=environment.planner_base_url,
            api_key=planner_key.get_secret_value(),
            transport_attempts=config.transport_attempts,
            proposal_attempts=config.proposal_attempts,
        )
        controller = Controller(
            task=task,
            space=space,
            catalog=catalog,
            methods=methods,
            proposal=proposal,
            runner=recipe_runner,
            run_dir=search_dir,
            references=references,
            hp_space=config.hp_space,
            candidate_count=config.proposal_candidates_per_round,
            candidate_pool_size=config.candidate_pool_size,
            cold_start_observations=config.cold_start_observations,
            mc_samples=config.mc_samples,
        )
        output = controller.run(max_rounds=rounds, max_total_hours=remaining_hours)
        winner = output["winner"]
        final_model = eval_runner.export_final_model(
            str(winner["checkpoint_path"]),
            search_dir / "final_model",
        )
        output["final_model"] = final_model
        write_json(search_dir / "final_output.json", output)
        status = read_json(search_dir / "status.json")
        status.update(
            {
                "status": "completed",
                "completed_at": time.time(),
                "elapsed_seconds": output["elapsed_seconds"],
                "final_model": final_model,
            }
        )
        write_json(search_dir / "status.json", status)
        _update_run_stage(work_dir, "search", status)
        return output
    except Exception as exc:
        status = read_json(search_dir / "status.json")
        status.update(
            {
                "status": "failed",
                "completed_at": time.time(),
                "elapsed_seconds": time.time() - started_at,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(search_dir / "status.json", status)
        write_json(
            search_dir / "final_output.json",
            {
                "status": "failed",
                "error": status["error"],
                "elapsed_seconds": status["elapsed_seconds"],
            },
        )
        _update_run_stage(work_dir, "search", status)
        raise


def run_evaluate(*, work_dir: Path, environment: EnvironmentConfig) -> dict[str, Any]:
    work_dir = work_dir.expanduser().resolve()
    _, task, _, _, _, _, _ = _load_handoff(work_dir)
    model_path = work_dir / "search" / "final_model"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Search final model does not exist: {model_path}")
    benchmark_metadata = load_benchmark_metadata(task)
    final_eval_args = {
        benchmark_id: [str(value) for value in metadata.get("final_eval_args", [])]
        for benchmark_id, metadata in benchmark_metadata.items()
    }
    for benchmark_id, metadata in benchmark_metadata.items():
        if metadata.get("requires_judge"):
            final_eval_args[benchmark_id].extend(
                ["--judge-workers", str(environment.healthbench_judge_workers)]
            )
    evaluator = EvalRunner(
        extra_args=final_eval_args,
        environment=environment.evaluation_environment(
            healthbench=any(
                metadata.get("requires_judge") for metadata in benchmark_metadata.values()
            )
        ),
        timeout_seconds=None,
    )
    started = time.time()
    metrics = evaluator.evaluate(
        model_path=str(model_path),
        task=task,
        recipe_id="final_model",
    )
    output = {
        "stage": "evaluate",
        "status": "completed",
        "model_path": str(model_path),
        "metrics": metrics,
        "metadata": evaluator.last_metadata,
        "elapsed_seconds": time.time() - started,
    }
    evaluate_dir = work_dir / "evaluate"
    evaluate_dir.mkdir(parents=True, exist_ok=True)
    write_json(evaluate_dir / "metrics.json", output)
    _update_run_stage(work_dir, "evaluate", output)
    return output
