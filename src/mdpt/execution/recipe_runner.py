from __future__ import annotations

import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mdpt.benchmarks import load_benchmark_metadata
from mdpt.discovery.schema import DataMethodSpec, Recipe, SearchSpace, SourceCatalog
from mdpt.discovery.validation import validate_recipe
from mdpt.execution.assembler import MultiDomainAssembler
from mdpt.execution.data_builder import DomainDataBuilder
from mdpt.execution.diagnostics import collect_diagnostics
from mdpt.execution.eval_runner import EvalRunner
from mdpt.execution.training_runner import TrainArtifact, TRLTrainingRunner, derive_params
from mdpt.schema import TaskConfig
from mdpt.search.featurizer import flatten_benchmark_mix
from mdpt.search.pareto_archive import TrialResult


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _run_training_subprocess(
    runner: TRLTrainingRunner,
    *,
    recipe_id: str,
    dataset: list[dict[str, Any]],
    derived: Any,
    sft_hyperparameters: dict[str, float | int],
    timeout_seconds: float | None,
) -> TrainArtifact:
    with tempfile.TemporaryDirectory(prefix="mdpt_training_") as temp_dir:
        root = Path(temp_dir)
        dataset_path = root / "dataset.jsonl"
        request_path = root / "request.json"
        response_path = root / "response.json"
        with dataset_path.open("w", encoding="utf-8") as handle:
            for row in dataset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        request_path.write_text(
            json.dumps(
                {
                    "base_model": runner.base_model,
                    "output_root": str(runner.output_root),
                    "templates_root": str(runner.templates_root),
                    "bf16": runner.bf16,
                    "recipe_id": recipe_id,
                    "dataset_path": str(dataset_path),
                    "derived": asdict(derived),
                    "sft_hyperparameters": sft_hyperparameters,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        process = subprocess.Popen(
            [sys.executable, "-m", "mdpt.execution.training_worker", str(request_path), str(response_path)],
            cwd=str(Path(__file__).resolve().parents[2]),
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise TimeoutError("SFT exceeded the Trial/Search deadline") from exc
        if return_code != 0:
            raise RuntimeError(f"SFT worker exited with code {return_code}")
        if not response_path.is_file():
            raise RuntimeError("SFT worker produced no response")
        return TrainArtifact(**json.loads(response_path.read_text(encoding="utf-8")))


class RecipeRunner:
    def __init__(
        self,
        *,
        task: TaskConfig,
        space: SearchSpace,
        catalog: SourceCatalog,
        methods: dict[str, DataMethodSpec],
        workspace: Path,
        run_dir: Path,
        runtime: dict[str, Any],
        training_runner: TRLTrainingRunner,
        eval_runner: EvalRunner,
        per_source_limit: int = 2000,
        generation_samples: int = 500,
        max_trial_seconds: float | None = None,
    ) -> None:
        self.task = task
        self.space = space
        self.catalog = catalog
        self.methods = methods
        self.workspace = workspace
        self.run_dir = run_dir
        self.runtime = runtime
        self.training_runner = training_runner
        self.eval_runner = eval_runner
        self.builder = DomainDataBuilder(
            catalog=catalog,
            methods=methods,
            workspace=workspace,
            run_dir=run_dir,
            runtime=runtime,
            per_source_limit=per_source_limit,
            generation_samples=generation_samples,
        )
        self.assembler = MultiDomainAssembler(seed=0)
        self.max_trial_seconds = max_trial_seconds
        self.baseline_metrics: dict[str, float] = {}
        self.benchmark_metadata = load_benchmark_metadata(task)
        self._probe_cache: dict[str, list[str]] = {}

    def _benchmark_probes(self, benchmark_id: str) -> list[str]:
        if benchmark_id in self._probe_cache:
            return self._probe_cache[benchmark_id]
        config = self.benchmark_metadata[benchmark_id].get("contamination_probe")
        if not isinstance(config, dict):
            self._probe_cache[benchmark_id] = []
            return []
        try:
            import datasets

            dataset = datasets.load_dataset(
                str(config["dataset"]),
                config.get("subset"),
                split=str(config.get("split", "test")),
            )
            field = str(config["field"])
            probes = [str(row[field]) for row in dataset if row.get(field)]
        except Exception as exc:
            raise RuntimeError(
                f"failed to load contamination probes for {benchmark_id!r}: {exc}"
            ) from exc
        self._probe_cache[benchmark_id] = probes
        return probes

    def run_baseline(self, *, deadline_monotonic: float | None = None) -> dict[str, float]:
        self.baseline_metrics = self.eval_runner.evaluate(
            model_path=self.task.base_model,
            task=self.task,
            recipe_id="base_model",
            deadline_monotonic=deadline_monotonic,
        )
        return dict(self.baseline_metrics)

    def run(
        self,
        *,
        recipe: Recipe,
        sft_hyperparameters: dict[str, float | int],
        allow_correction: bool,
        diagnostic_checkpoint: str | None = None,
        search_deadline_monotonic: float | None = None,
    ) -> TrialResult:
        data_stats: dict[str, Any] = {}
        diagnostic_stats: dict[str, Any] = {}
        checkpoint: str | None = None
        trial_deadline = (
            time.monotonic() + self.max_trial_seconds
            if self.max_trial_seconds is not None
            else None
        )
        if search_deadline_monotonic is not None:
            trial_deadline = (
                min(trial_deadline, search_deadline_monotonic)
                if trial_deadline is not None
                else search_deadline_monotonic
            )

        def check_deadline(stage: str) -> None:
            if trial_deadline is not None and time.monotonic() >= trial_deadline:
                raise TimeoutError(f"trial time budget exhausted before {stage}")

        try:
            check_deadline("validation")
            validate_recipe(
                self.task,
                self.catalog,
                self.methods,
                self.space,
                recipe,
                allow_correction=allow_correction,
            )
            requested_diagnostics: dict[str, set[str]] = {}
            for domain_recipe in recipe.domains.values():
                for benchmark_id, benchmark_recipe in domain_recipe.benchmarks.items():
                    if not benchmark_recipe.data_method:
                        continue
                    method = self.methods[benchmark_recipe.data_method]
                    if method.operation == "correction" and method.diagnostic_source_id:
                        requested_diagnostics.setdefault(benchmark_id, set()).add(
                            method.diagnostic_source_id
                        )
            if requested_diagnostics:
                check_deadline("Correction diagnostics")
                if not diagnostic_checkpoint:
                    raise RuntimeError("Correction requires a previous feasible checkpoint")
                resolved_catalog = SourceCatalog(
                    sources=[
                        source.model_copy(
                            update={
                                "path": str(
                                    (self.workspace / str(source.path)).resolve()
                                    if source.kind == "local" and not Path(str(source.path)).is_absolute()
                                    else source.path
                                )
                            }
                        )
                        for source in self.catalog.sources
                    ]
                )
                diagnostic_stats = collect_diagnostics(
                    self.run_dir,
                    recipe.recipe_id,
                    diagnostic_checkpoint,
                    self.methods,
                    resolved_catalog,
                    self.runtime,
                    requested_sources=requested_diagnostics,
                    deadline_monotonic=trial_deadline,
                )

            pools = {}
            for domain in self.task.domains:
                domain_recipe = recipe.domains[domain.domain_id]
                for benchmark in domain.benchmarks:
                    benchmark_id = benchmark.benchmark_id
                    check_deadline(f"data build for {benchmark_id}")
                    pool = self.builder.build(
                        domain_id=domain.domain_id,
                        benchmark_id=benchmark_id,
                        recipe=domain_recipe.benchmarks[benchmark_id],
                        benchmark_probes=self._benchmark_probes(benchmark_id),
                        deadline_monotonic=trial_deadline,
                    )
                    pools[benchmark_id] = pool
                    data_stats[benchmark_id] = pool.stats

            flat_mix = flatten_benchmark_mix(recipe)
            mixed = self.assembler.assemble(
                pools=pools,
                ratio=flat_mix,
                total_train_samples=self.task.budget.max_train_samples_per_trial,
                mode=recipe.multi_domain_mode,
            )
            if not mixed.samples:
                raise RuntimeError("assembled SFT dataset is empty")
            check_deadline("SFT")
            derived = derive_params(
                base_model=self.task.base_model,
                samples=mixed.samples,
                sft_hyperparameters=sft_hyperparameters,
            )
            training_timeout = (
                trial_deadline - time.monotonic() if trial_deadline is not None else None
            )
            if training_timeout is not None and training_timeout <= 0:
                raise TimeoutError("trial time budget exhausted before SFT")
            artifact = _run_training_subprocess(
                self.training_runner,
                recipe_id=recipe.recipe_id,
                dataset=mixed.samples,
                derived=derived,
                sft_hyperparameters=sft_hyperparameters,
                timeout_seconds=training_timeout,
            )
            checkpoint = artifact.checkpoint_path
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            check_deadline("evaluation")
            metrics = self.eval_runner.evaluate(
                model_path=checkpoint,
                task=self.task,
                recipe_id=recipe.recipe_id,
                deadline_monotonic=trial_deadline,
            )
            return TrialResult(
                recipe_id=recipe.recipe_id,
                metrics=metrics,
                checkpoint_path=checkpoint,
                metadata={
                    "sft_hyperparameters": sft_hyperparameters,
                    "flat_benchmark_mix": flat_mix,
                    "data": data_stats,
                    "diagnostics": diagnostic_stats,
                    "assembled": mixed.stats,
                    "per_benchmark_counts": mixed.per_benchmark_counts,
                    "training": artifact.info,
                    "evaluation": self.eval_runner.last_metadata,
                },
            )
        except Exception as exc:
            return TrialResult(
                recipe_id=recipe.recipe_id,
                feasible=False,
                checkpoint_path=checkpoint,
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "sft_hyperparameters": sft_hyperparameters,
                    "data": data_stats,
                    "diagnostics": diagnostic_stats,
                },
            )
