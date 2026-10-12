from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from mdpt.discovery.adapters import get_augmentation_adapter
from mdpt.discovery.adapters.protocol import answer as _answer
from mdpt.discovery.adapters.protocol import choices as _choices
from mdpt.discovery.adapters.protocol import question as _question
from mdpt.discovery.schema import DataMethodSpec, SourceSpec
from mdpt.discovery.sources.data import load_raw_source, with_resolved_source_target
from mdpt.discovery.formatters import (
    enforce_answer_format_gpqa,
    enforce_answer_format_math,
    wrap_gpqa_user,
    wrap_humaneval_user,
    wrap_math_user,
)
from mdpt.discovery.methods.code_execution import (
    append_jsonl,
    prepare_humaneval_sft,
    read_done_indices,
)
from mdpt.discovery.methods.generation import FrameworkGenerator


def task_format(source: SourceSpec, sample: dict[str, Any] | None = None) -> str:
    formatter = source.format.formatter
    if formatter == "posttrainbench_math":
        return "math"
    if formatter == "posttrainbench_humaneval":
        return "code"
    if formatter == "posttrainbench_gpqa":
        return "multiple_choice"
    if formatter == "posttrainbench_bfcl":
        return "bfcl"
    if formatter == "healthbench":
        return "health"
    row = sample or {}
    if any(key in row for key in ("test", "tests", "entry_point", "canonical_solution")):
        return "code"
    if isinstance(row.get("choices") or row.get("options"), list):
        return "multiple_choice"
    return "math"


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("generator response does not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("generator response must be a JSON object")
    return value


def _complete_json_object(generator: FrameworkGenerator, prompt: str) -> dict[str, Any]:
    """Retry one transient content-format failure without changing the stage prompt."""
    for attempt in range(2):
        try:
            return _json_object(generator.complete(prompt))
        except (json.JSONDecodeError, ValueError):
            if attempt:
                raise
        except RuntimeError as exc:
            transient = (
                "exhausted max_tokens" in str(exc)
                or "returned empty content" in str(exc)
            )
            if attempt or not transient:
                raise
    raise AssertionError("unreachable")


def _messages(
    fmt: str,
    question: str,
    answer: str,
    choices: list[str] | None = None,
    entry_point: str = "",
) -> list[dict[str, str]]:
    if fmt == "math":
        user = wrap_math_user(question)
        assistant = enforce_answer_format_math(answer)
    elif fmt == "code":
        code_prompt, assistant, _ = prepare_humaneval_sft(
            question=question,
            solution=answer,
            entry_point=entry_point,
        )
        user = wrap_humaneval_user(code_prompt)
    else:
        user = wrap_gpqa_user(question, choices or [])
        assistant = enforce_answer_format_gpqa(answer, choices or [])
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def _stage_prompt(
    authored_prompt: str,
    context: dict[str, Any],
    output_fields: set[str],
    *,
    stage: str,
    evolution_axes: list[str] | None = None,
    output_schema: str | None = None,
) -> str:
    axes = ""
    if evolution_axes:
        axes = "\nAgent-derived evolution axes:\n" + "\n".join(
            f"- {axis}" for axis in evolution_axes
        )
    schema = output_schema or json.dumps(
        {field: "<string>" for field in sorted(output_fields)}, ensure_ascii=False
    )
    return (
        f"You are executing the {stage} stage of an Agent-designed data method.\n"
        "The method prompt below decides the transformation; do not invent an additional strategy.\n"
        "Return exactly one JSON object, without markdown or commentary, with exactly the required fields.\n\n"
        f"Agent-authored method prompt:\n{authored_prompt.strip()}\n"
        f"{axes}\n\nInput record:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"Required output shape:\n{schema}"
    )


def generate_method_example(
    method: DataMethodSpec,
    source: SourceSpec,
    reference: dict[str, Any],
    generator: FrameworkGenerator,
) -> tuple[dict[str, Any], Any, list[str]]:
    adapter = get_augmentation_adapter(str(source.adapter_id or ""))
    prompts: list[str] = []
    if method.operation == "rewrite":
        rewrite_fields = adapter.rewrite_fields()
        rewrite_prompt = _stage_prompt(
            str(method.rewrite_prompt),
            adapter.rewrite_context(reference),
            rewrite_fields,
            stage="rewrite",
            evolution_axes=method.evolution_axes,
            output_schema=adapter.json_schema(rewrite_fields),
        )
        prompts.append(rewrite_prompt)
        transformed = _complete_json_object(generator, rewrite_prompt)
        answer_fields = adapter.answer_fields()
        answer_context = adapter.answer_context(transformed, reference)
        answer_prompt = _stage_prompt(
            str(method.answer_prompt),
            answer_context,
            answer_fields,
            stage="answer",
            output_schema=adapter.json_schema(answer_fields),
        )
        prompts.append(answer_prompt)
        supervision = _complete_json_object(generator, answer_prompt)
        verification_fields = adapter.verification_fields()
        if verification_fields:
            authored_prompt = adapter.verification_prompt()
            if not authored_prompt:
                raise RuntimeError(
                    f"adapter {adapter.benchmark!r} declares verification fields without a prompt"
                )
            verification_prompt = _stage_prompt(
                authored_prompt,
                adapter.verification_context(transformed, supervision),
                verification_fields,
                stage="independent verification artifact",
                output_schema=adapter.json_schema(verification_fields),
            )
            prompts.append(verification_prompt)
            supervision.update(_complete_json_object(generator, verification_prompt))
        generated = adapter.assemble_rewrite(transformed, supervision, reference, method)
    else:
        correction_prompt = _stage_prompt(
            str(method.correction_prompt),
            adapter.correction_context(reference),
            adapter.correction_fields(),
            stage="correction",
            output_schema=adapter.json_schema(adapter.correction_fields()),
        )
        prompts.append(correction_prompt)
        corrected = _complete_json_object(generator, correction_prompt)
        generated = adapter.assemble_correction(corrected, reference, method)
    verification = adapter.verify(generated, reference, method, generator)
    return generated, verification, prompts


def _output_row(
    fmt: str,
    generated: dict[str, Any],
    reference: dict[str, Any],
    method: DataMethodSpec,
) -> dict[str, Any]:
    del reference
    if fmt in {"bfcl", "health"}:
        result = {
            "messages": generated.get("messages") or [],
            "generation_method": method.method_id,
            "operation": method.operation,
            "label_provenance": (
                "teacher_labeled_and_verified"
                if method.operation == "rewrite"
                else "verifier_corrected"
            ),
        }
        if fmt == "bfcl":
            result["tools"] = generated.get("tools") or []
        return result
    question = str(generated.get("question") or "").strip()
    answer = str(generated.get("solution") or generated.get("answer") or "").strip()
    choices = [str(item) for item in (generated.get("choices") or [])]
    entry_point = str(generated.get("entry_point") or "")
    tests = str(generated.get("tests") or generated.get("test") or "")
    if fmt == "code":
        question, answer, _ = prepare_humaneval_sft(
            question=question,
            solution=answer,
            entry_point=entry_point,
        )
    provenance = (
        "teacher_labeled_and_verified"
        if method.operation == "rewrite"
        else "verifier_corrected"
    )
    return {
        "messages": _messages(fmt, question, answer, choices, entry_point),
        "question": question,
        "answer": answer,
        "choices": choices,
        "test": tests,
        "entry_point": entry_point,
        "generation_method": method.method_id,
        "operation": method.operation,
        "label_provenance": provenance,
    }


def _latest_diagnostic(run_dir: Path, method: DataMethodSpec) -> dict[str, Any]:
    index_path = run_dir / "diagnostics" / "index.json"
    if not index_path.is_file():
        raise RuntimeError(f"correction method {method.method_id!r} requires a completed diagnostic run")
    entries = json.loads(index_path.read_text(encoding="utf-8")).get("entries", [])
    matching = [
        entry for entry in entries
        if entry.get("benchmark_id") == method.benchmark_id
        and entry.get("diagnostic_source_id") == method.diagnostic_source_id
    ]
    if not matching:
        raise RuntimeError(f"no diagnostic result is available for correction method {method.method_id!r}")
    latest = matching[-1]
    if latest.get("status", "completed") != "completed":
        raise RuntimeError(
            f"latest diagnostic attempt failed for correction method {method.method_id!r}: "
            f"{latest.get('error', 'unknown diagnostic error')}; older results are not reused silently"
        )
    if not latest.get("usable", int(latest.get("n_wrong", 0)) > 0) or not Path(str(latest.get("wrong_examples_path", ""))).is_file():
        raise RuntimeError(
            f"latest diagnostic result has no usable wrong examples for correction method {method.method_id!r}; "
            "older failures are not reused silently"
        )
    return latest


def correction_unavailable_reason(run_dir: Path, method: DataMethodSpec) -> str | None:
    if method.operation != "correction":
        return None
    try:
        _latest_diagnostic(run_dir, method)
    except RuntimeError as exc:
        return str(exc)
    return None


def materialize_augmentation(
    method: DataMethodSpec,
    sources: dict[str, SourceSpec],
    runtime: dict[str, Any],
    run_dir: Path,
    deadline_monotonic: float | None = None,
) -> SourceSpec:
    diagnostic = _latest_diagnostic(run_dir, method) if method.operation == "correction" else None
    input_id = method.input_source_id if method.operation == "rewrite" else method.diagnostic_source_id
    source = sources.get(str(input_id))
    if source is None:
        raise ValueError(f"unknown {method.operation} input source {input_id!r}")
    identity = {"method": method.model_dump(), "diagnostic": diagnostic}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:12]
    out_dir = run_dir / "generated" / method.method_id / digest
    output_path, attempts_path, manifest_path = out_dir / "data.jsonl", out_dir / "attempts.jsonl", out_dir / "manifest.json"
    generated_id = f"generated::{method.method_id}::{digest}"
    generated_format = (
        {"messages_field": "messages", "formatter": source.format.formatter}
        if source.format.formatter in {"posttrainbench_bfcl", "healthbench"}
        else {"messages_field": "messages"}
    )
    if output_path.is_file() and manifest_path.is_file():
        return SourceSpec(
            source_id=generated_id,
            benchmark_id=method.benchmark_id,
            kind="local",
            path=str(output_path),
            format=generated_format,
            metadata={"method_id": method.method_id, "manifest": str(manifest_path)},
        )

    if method.operation == "rewrite":
        source = sources.get(str(method.input_source_id))
        if source is None:
            raise ValueError(f"unknown rewrite input source {method.input_source_id!r}")
        limit = int(os.environ.get("MDPT_REWRITE_INPUT_EXAMPLES", "500"))
        input_rows = load_raw_source(source, limit)
    else:
        source = sources.get(str(method.diagnostic_source_id))
        if source is None:
            raise ValueError(f"unknown correction diagnostic source {method.diagnostic_source_id!r}")
        limit = int(os.environ.get("MDPT_CORRECTION_INPUT_EXAMPLES", "500"))
        input_rows = []
        with Path(str(diagnostic["wrong_examples_path"])).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    input_rows.append(json.loads(line))
                    if len(input_rows) >= limit:
                        break
    if not input_rows:
        raise RuntimeError(f"augmentation method {method.method_id!r} has no input examples")

    fmt = task_format(source, input_rows[0])
    if fmt == "math":
        input_rows = [with_resolved_source_target(row, source) for row in input_rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    done_indices = read_done_indices(attempts_path, key="input_index")
    previous_timeout = os.environ.get("MDPT_AUGMENTATION_TIMEOUT_SECONDS")
    if deadline_monotonic is not None:
        remaining = max(1.0, deadline_monotonic - time.monotonic())
        configured = float(previous_timeout or "300")
        os.environ["MDPT_AUGMENTATION_TIMEOUT_SECONDS"] = str(min(configured, remaining))
    generator: FrameworkGenerator | None = None
    try:
        generator = FrameworkGenerator(runtime, deadline_monotonic=deadline_monotonic)
        for index, reference in enumerate(input_rows):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError(f"augmentation method {method.method_id!r} exceeded the Trial/Search deadline")
            if index in done_indices:
                continue
            attempt: dict[str, Any] = {"input_index": index, "passed": False}
            try:
                generated, verification, _ = generate_method_example(
                    method, source, reference, generator
                )
                attempt.update(
                    {
                        "passed": verification.passed,
                        "feedback": verification.feedback,
                        "verification_details": verification.details,
                    }
                )
                if verification.passed:
                    append_jsonl(output_path, _output_row(fmt, generated, reference, method))
            except TimeoutError:
                raise
            except Exception as exc:
                attempt["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                append_jsonl(attempts_path, attempt)
    finally:
        if generator is not None:
            generator.close()
        if previous_timeout is None:
            os.environ.pop("MDPT_AUGMENTATION_TIMEOUT_SECONDS", None)
        else:
            os.environ["MDPT_AUGMENTATION_TIMEOUT_SECONDS"] = previous_timeout

    n_attempted = sum(1 for line in attempts_path.open(encoding="utf-8") if line.strip())
    n_kept = sum(1 for line in output_path.open(encoding="utf-8") if line.strip()) if output_path.is_file() else 0
    if not n_kept:
        raise RuntimeError(
            f"augmentation method {method.method_id!r} produced no verified examples; "
            f"inspect {attempts_path}"
        )
    manifest = {
        "source_id": generated_id,
        "method_id": method.method_id,
        "operation": method.operation,
        "benchmark_id": method.benchmark_id,
        "input_source_id": method.input_source_id,
        "diagnostic_source_id": method.diagnostic_source_id,
        "diagnostic": diagnostic,
        "method_contract_hash": hashlib.sha256(
            json.dumps(method.model_dump(), sort_keys=True).encode()
        ).hexdigest(),
        "n_input": len(input_rows),
        "n_attempted": n_attempted,
        "n_kept": n_kept,
        "pass_rate": n_kept / n_attempted if n_attempted else 0.0,
        "attempts_path": str(attempts_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return SourceSpec(
        source_id=generated_id,
        benchmark_id=method.benchmark_id,
        kind="local",
        path=str(output_path),
        format=generated_format,
        metadata={
            "method_id": method.method_id,
            "manifest": str(manifest_path),
            "operation": method.operation,
        },
    )
