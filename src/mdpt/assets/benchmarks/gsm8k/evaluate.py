#!/usr/bin/env python3
from __future__ import annotations
import os

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from inspect_ai.log._log import EvalLog, EvalMetric, EvalSample
from inspect_ai import eval as inspect_eval  # type: ignore  # noqa: E402
from inspect_ai.util._display import init_display_type  # noqa: E402

import inspect_evals.gsm8k # noqa: F401, E402  (registers task definitions)
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, prompt_template, system_message
from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE, record_to_sample, sample_to_fewshot



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inspect AI eval without banners.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="final_model",
        help="Path to the Hugging Face model (directory or model identifier).",
    )
    # Restrict the number of evaluated samples when a bounded run is required.
    parser.add_argument(
        "--limit",
        type=int,
        default=150,
        help="Optional limit for number of samples to evaluate.",
    )
    parser.add_argument(
        '--json-output-file',
        type=str,
        default=None,
        help="Optional path to output the metrics as a seperate JSON file.",
    )
    parser.add_argument(
        '--templates-dir',
        type=str,
        default="templates/",
    )
    # Controls concurrent model requests; lower values reduce backend resource pressure.
    parser.add_argument(
        "--max-connections",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional local GSM8K-format JSONL split with question/answer fields.",
    )
    parser.add_argument(
        "--skill-tags-path",
        type=str,
        default=None,
        help="Optional JSONL file with per-sample rule-based skills for skill-level accuracy.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional Inspect log directory. A temporary directory is used when skill tags require sample logs.",
    )
    return parser.parse_args()


def _load_local_gsm8k_task(dataset_path: str) -> Task:
    samples: list[Sample] = []
    with open(dataset_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample = record_to_sample(record)
            sample.id = sample.id or f"local_gsm8k_{i}"
            samples.append(sample)

    solver = [prompt_template(MATH_PROMPT_TEMPLATE), generate()]
    try:
        import datasets  # type: ignore

        ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
        fewshot_rows = ds.shuffle(seed=42).select(range(min(10, len(ds))))
        fewshots = [record_to_sample(dict(row)) for row in fewshot_rows]
        solver.insert(0, system_message("\n\n".join([sample_to_fewshot(s) for s in fewshots])))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] failed to load GSM8K fewshots for local eval: {e}")
    return Task(dataset=samples, solver=solver, scorer=match(numeric=True))


def _read_skill_tags(path: str | Path, limit: Optional[int] = None) -> list[list[str]]:
    tags: list[list[str]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            skills = obj.get("skills") or []
            tags.append([str(s) for s in skills])
            if limit is not None and limit >= 0 and len(tags) >= limit:
                break
    return tags


def _sample_to_record(sample: Any) -> dict[str, Any]:
    record = sample.model_dump(mode="json") if hasattr(sample, "model_dump") else dict(sample)
    scores = record.get("scores") or {}
    score_name = next(iter(scores), None) if isinstance(scores, dict) and scores else None
    score_obj = scores.get(score_name) if score_name else None
    record["_primary_score_name"] = score_name
    record["_score_value"] = score_obj.get("value") if isinstance(score_obj, dict) else None
    return record


def _score_is_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    text = str(value or "").strip().lower()
    return text in {"c", "correct", "true", "1", "1.0"}


def _skill_metrics_from_samples(samples: list[Any], skill_tags: list[list[str]]) -> dict[str, float]:
    totals: dict[str, int] = defaultdict(int)
    corrects: dict[str, int] = defaultdict(int)
    tagged_n = min(len(samples), len(skill_tags))
    for i in range(tagged_n):
        record = _sample_to_record(samples[i])
        correct = _score_is_correct(record.get("_score_value"))
        for skill in skill_tags[i]:
            totals[skill] += 1
            if correct:
                corrects[skill] += 1

    metrics: dict[str, float] = {
        "skill_tagged_samples": float(tagged_n),
    }
    accuracies: list[float] = []
    for skill in sorted(totals):
        total = totals[skill]
        correct = corrects.get(skill, 0)
        acc = correct / total if total else 0.0
        metrics[f"skill_total.{skill}"] = float(total)
        metrics[f"skill_correct.{skill}"] = float(correct)
        metrics[f"skill_accuracy.{skill}"] = round(acc, 6)
        accuracies.append(acc)
    if accuracies:
        metrics["skill_macro_accuracy"] = round(sum(accuracies) / len(accuracies), 6)
    return metrics


def main() -> None:
    args = parse_args()

    init_display_type("plain")

    other_kwargs = {}
    if (args.limit is not None) and (args.limit != -1):
        other_kwargs["limit"] = args.limit

    task = _load_local_gsm8k_task(args.dataset_path) if args.dataset_path else "inspect_evals/gsm8k"
    model_args = {
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }
    model_args.update(template_kwargs(args))

    eval_kwargs = {
        "score_display": False,
        "log_realtime": False,
        "log_format": 'json',
        "timeout": 18000000,
        "attempt_timeout": 18000000,
        "max_tokens": args.max_tokens,
        "max_connections": args.max_connections,
        **other_kwargs,
    }
    skill_tags = None
    if args.skill_tags_path:
        skill_tags = _read_skill_tags(args.skill_tags_path, other_kwargs.get("limit"))
        eval_kwargs["log_samples"] = True

    tmp_log_dir = None
    try:
        if args.skill_tags_path:
            if args.log_dir:
                eval_kwargs["log_dir"] = args.log_dir
            else:
                tmp_log_dir = tempfile.TemporaryDirectory(prefix="gsm8k_skill_eval_")
                eval_kwargs["log_dir"] = tmp_log_dir.name

        eval_out = inspect_eval(
            task,
            model=f"vllm/{args.model_path}",
            model_args=model_args,
            **eval_kwargs,
        )
    finally:
        if tmp_log_dir is not None:
            tmp_log_dir.cleanup()

    assert len(eval_out) == 1, eval_out
    assert len(eval_out[0].results.scores) == 1, eval_out[0].results.scores
    metrics = {}
    for k, v in eval_out[0].results.scores[0].metrics.items():
        metrics[k] = v.value
    if skill_tags is not None:
        metrics.update(_skill_metrics_from_samples(list(eval_out[0].samples or []), skill_tags))

    if args.json_output_file is not None:
        with open(args.json_output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

def model_type(args) -> str:
    if 'qwen' in args.model_path.lower():
        return 'qwen'
    if 'llama' in args.model_path.lower():
        return 'llama'
    if 'gemma' in args.model_path.lower():
        return 'gemma'
    if 'smollm' in args.model_path.lower():
        return 'smollm'

    with open(os.path.join(args.model_path, "config.json"), 'r') as f:
        config = json.load(f)
    architecture = config['architectures'][0].lower()
    if 'gemma' in architecture:
        return 'gemma'
    if 'llama' in architecture:
        return 'llama'
    if 'qwen' in architecture:
        return 'qwen'
    if 'smollm' in architecture:
        return 'smollm'
    raise ValueError(architecture)

def template_kwargs(args) -> dict:
    model_type_str = model_type(args)
    if model_type_str == 'qwen':
        template = 'qwen3.jinja'
    elif model_type_str == 'llama':
        template = 'llama3.jinja'
    elif model_type_str == 'gemma':
        template = 'gemma3.jinja'
    elif model_type_str == 'smollm':
        template = 'smollm.jinja'
    else:
        raise ValueError(model_type_str)
    return {
        'chat_template': os.path.join(args.templates_dir, template)
    }

if __name__ == "__main__":
    main()
