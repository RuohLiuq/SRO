#!/usr/bin/env python3
"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard
Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022

Based on: https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/gpqa/gpqa.py
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from inspect_ai import Task, eval as inspect_eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from inspect_ai.util._display import init_display_type  # noqa: E402

DEFAULT_EPOCHS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inspect AI GPQA eval without banners.")
    parser.add_argument("--model-path", type=str, default="final_model")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--json-output-file", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--templates-dir", type=str, default="templates/")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-connections", type=int, default=6)
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional local GPQA JSONL with question/choices/label fields.")
    parser.add_argument("--skill-tags-path", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    return parser.parse_args()


@task
def gpqa_main() -> Task:
    return Task(
        dataset=hf_dataset(
            path="Idavidrein/gpqa",
            name="gpqa_main",
            split="train",
            sample_fields=record_to_sample,
            shuffle_choices=True,
        ),
        solver=[multiple_choice(cot=True)],
        scorer=choice(),
        epochs=DEFAULT_EPOCHS,
    )


# map records to inspect samples (note that target is always "A" in the raw
# dataset; HF evaluation shuffles choices to mitigate this)
def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["Question"],
        choices=[
            str(record["Correct Answer"]),
            str(record["Incorrect Answer 1"]),
            str(record["Incorrect Answer 2"]),
            str(record["Incorrect Answer 3"]),
        ],
        target="A",
        id=record["Record ID"],
    )


def _load_local_gpqa_task(dataset_path: str) -> Task:
    samples: list[Sample] = []
    with open(dataset_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = str(record.get("question") or record.get("prompt") or record.get("problem") or "").strip()
            choices = record.get("choices") or record.get("options") or record.get("answer_choices") or []
            if not isinstance(choices, list):
                choices = []
            choices = [str(c.get("text") if isinstance(c, dict) else c) for c in choices]
            target = str(record.get("label") or record.get("gold_label") or record.get("target") or "").strip().upper()
            if not target and record.get("answer"):
                answer = str(record.get("answer") or "")
                for letter in ("A", "B", "C", "D"):
                    if f"ANSWER: {letter}" in answer.upper() or answer.strip().upper() == letter:
                        target = letter
                        break
            if not question or len(choices) != 4 or target not in {"A", "B", "C", "D"}:
                raise ValueError(f"bad local GPQA row {i}: requires question, 4 choices, and label A-D")
            samples.append(
                Sample(
                    id=str(record.get("record_id") or record.get("id") or f"local_gpqa_{i}"),
                    input=question,
                    choices=choices,
                    target=target,
                    metadata={
                        "record_id": record.get("record_id"),
                        "correct_answer": record.get("correct_answer"),
                        "label": target,
                        "skills": record.get("skills") or [],
                    },
                )
            )
    return Task(dataset=samples, solver=[multiple_choice(cot=True)], scorer=choice(), epochs=DEFAULT_EPOCHS)


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
    return str(value or "").strip().lower() in {"c", "correct", "true", "1", "1.0"}


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

    metrics: dict[str, float] = {"skill_tagged_samples": float(tagged_n)}
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

    task_obj = _load_local_gpqa_task(args.dataset_path) if args.dataset_path else gpqa_main()
    model_args = {"gpu_memory_utilization": args.gpu_memory_utilization}
    model_args.update(template_kwargs(args))

    eval_kwargs = {
        "score_display": False,
        "log_realtime": False,
        "timeout": 18000000,
        "attempt_timeout": 18000000,
        "log_format": "json",
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
                tmp_log_dir = tempfile.TemporaryDirectory(prefix="gpqa_skill_eval_")
                eval_kwargs["log_dir"] = tmp_log_dir.name

        eval_out = inspect_eval(
            task_obj,
            model=f"vllm/{args.model_path}",
            model_args=model_args,
            **eval_kwargs,
        )
    finally:
        if tmp_log_dir is not None:
            tmp_log_dir.cleanup()

    assert len(eval_out) == 1, eval_out
    assert len(eval_out[0].results.scores) == 1, eval_out[0].results.scores
    metrics = {k: v.value for k, v in eval_out[0].results.scores[0].metrics.items()}
    if skill_tags is not None:
        metrics.update(_skill_metrics_from_samples(list(eval_out[0].samples or []), skill_tags))

    if args.json_output_file is not None:
        with open(args.json_output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


def model_type(args) -> str:
    if "qwen" in args.model_path.lower():
        return "qwen"
    if "llama" in args.model_path.lower():
        return "llama"
    if "gemma" in args.model_path.lower():
        return "gemma"
    if "smollm" in args.model_path.lower():
        return "smollm"

    with open(os.path.join(args.model_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    architecture = config["architectures"][0].lower()
    if "gemma" in architecture:
        return "gemma"
    if "llama" in architecture:
        return "llama"
    if "qwen" in architecture:
        return "qwen"
    if "smollm" in architecture:
        return "smollm"
    raise ValueError(architecture)


def template_kwargs(args) -> dict:
    model_type_str = model_type(args)
    if model_type_str == "qwen":
        template = "qwen3.jinja"
    elif model_type_str == "llama":
        template = "llama3.jinja"
    elif model_type_str == "gemma":
        template = "gemma3.jinja"
    elif model_type_str == "smollm":
        template = "smollm.jinja"
    else:
        raise ValueError(model_type_str)
    return {"chat_template": os.path.join(args.templates_dir, template)}


if __name__ == "__main__":
    main()
