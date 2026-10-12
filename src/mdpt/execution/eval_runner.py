from __future__ import annotations

import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from mdpt.schema import TaskConfig


_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_BENCHMARKS = _ASSETS / "benchmarks"
_TEMPLATES = _ASSETS / "templates"
_PRIMARY_KEYS = ("accuracy", "pass@1", "score", "mean")


def _is_lora_adapter(path: str | Path) -> bool:
    root = Path(path)
    return (root / "adapter_config.json").is_file() and not (root / "config.json").is_file()


def merge_lora_adapter(adapter_path: str | Path, output_dir: str | Path) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_root = Path(adapter_path)
    config = json.loads((adapter_root / "adapter_config.json").read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise RuntimeError(f"adapter has no base_model_name_or_path: {adapter_root}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_root))
    merged = model.merge_and_unload()
    if "lm_head" in (config.get("modules_to_save") or []):
        merged.config.tie_word_embeddings = False
    generation_config = getattr(merged, "generation_config", None)
    if generation_config is not None and getattr(generation_config, "do_sample", None) is False:
        generation_config.temperature = None
        generation_config.top_p = None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    del merged, model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(output)


@contextmanager
def _model_for_evaluation(model_path: str) -> Iterator[str]:
    if not _is_lora_adapter(model_path):
        yield model_path
        return
    temporary = tempfile.mkdtemp(prefix="mdpt_merged_")
    try:
        yield merge_lora_adapter(model_path, temporary)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


class EvalRunner:
    def __init__(
        self,
        *,
        extra_args: dict[str, list[str]] | None = None,
        environment: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.extra_args = dict(extra_args or {})
        self.environment = dict(environment or {})
        self.timeout_seconds = timeout_seconds
        self.last_metadata: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _primary(metrics: dict[str, float]) -> float:
        for key in _PRIMARY_KEYS:
            if key in metrics:
                return float(metrics[key])
        if not metrics:
            raise RuntimeError("benchmark returned no numeric metrics")
        return float(next(iter(metrics.values())))

    def _run_one(
        self,
        *,
        model_path: str,
        benchmark_id: str,
        recipe_id: str,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        benchmark_dir = _BENCHMARKS / benchmark_id
        evaluator = benchmark_dir / "evaluate.py"
        if not evaluator.is_file():
            raise FileNotFoundError(f"missing packaged evaluator: {evaluator}")
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as handle:
            output_path = handle.name
        command = [
            sys.executable,
            str(evaluator),
            "--model-path",
            model_path,
            "--json-output-file",
            output_path,
            "--templates-dir",
            str(_TEMPLATES),
        ]
        command.extend(self.extra_args.get(benchmark_id, []))
        if benchmark_id == "healthbench":
            command.extend(["--sample-seed", "42"])
        environment = os.environ.copy()
        environment.update(self.environment)
        environment["PATH"] = (
            str(Path(sys.executable).resolve().parent)
            + os.pathsep
            + environment.get("PATH", "")
        )
        process: subprocess.Popen[Any] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=benchmark_dir,
                env=environment,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise TimeoutError(
                    f"benchmark {benchmark_id!r} exceeded its remaining time budget"
                ) from exc
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            return json.loads(Path(output_path).read_text(encoding="utf-8"))
        finally:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            Path(output_path).unlink(missing_ok=True)

    def evaluate(
        self,
        *,
        model_path: str,
        task: TaskConfig,
        recipe_id: str,
        deadline_monotonic: float | None = None,
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        self.last_metadata = {}
        with _model_for_evaluation(model_path) as evaluation_model:
            for benchmark_id in task.benchmark_ids:
                remaining = (
                    deadline_monotonic - time.monotonic()
                    if deadline_monotonic is not None
                    else None
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("evaluation exceeded the trial or Search time budget")
                timeout = self.timeout_seconds
                if remaining is not None:
                    timeout = remaining if timeout is None else min(float(timeout), remaining)
                raw = self._run_one(
                    model_path=evaluation_model,
                    benchmark_id=benchmark_id,
                    recipe_id=recipe_id,
                    timeout_seconds=timeout,
                )
                if not isinstance(raw, dict):
                    raise RuntimeError(f"benchmark {benchmark_id!r} returned non-object metrics")
                numeric = {
                    key: float(value)
                    for key, value in raw.items()
                    if isinstance(value, (int, float))
                }
                self.last_metadata[benchmark_id] = {
                    key: value for key, value in raw.items() if not isinstance(value, (int, float))
                }
                output[benchmark_id] = round(self._primary(numeric), 8)
                for key, value in numeric.items():
                    output[f"{benchmark_id}:{key}"] = round(value, 8)
        return output

    def export_final_model(self, checkpoint_path: str, output_dir: Path) -> str:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if _is_lora_adapter(checkpoint_path):
            return merge_lora_adapter(checkpoint_path, output_dir)
        shutil.copytree(checkpoint_path, output_dir)
        return str(output_dir)
