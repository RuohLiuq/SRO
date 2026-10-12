from __future__ import annotations

import gc
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from mdpt.discovery.adapters.bfcl import parse_tool_calls, verify_tool_calls
from mdpt.discovery.adapters.healthbench import health_messages, verify_health_answer
from mdpt.discovery.schema import DataMethodSpec, SourceCatalog, SourceSpec
from mdpt.discovery.sources.data import load_raw_source, with_resolved_source_target, write_jsonl
from mdpt.discovery.formatters import wrap_gpqa_user, wrap_humaneval_user, wrap_math_user
from mdpt.discovery.methods.code_execution import (
    append_jsonl,
    code_verifier_fields,
    extract_code,
    first_present,
    read_json_rows,
    prepare_humaneval_sft,
    verify_humaneval_sft_completion,
)
from mdpt.discovery.methods.engine import _answer, _choices, _question, task_format
from mdpt.discovery.methods.generation import FrameworkGenerator
from mdpt.discovery.methods.verifiers import MathExactMatchVerifier, MultipleChoiceVerifier
from mdpt.execution.eval_runner import _model_for_evaluation


DIAGNOSTIC_MAX_TOKENS = 1024
DIAGNOSTIC_SOURCE_LIMIT = 500
_DIAGNOSTIC_WORKER_FLAG = "--diagnostic-worker"


def diagnostic_sources(methods: dict[str, DataMethodSpec]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for method in methods.values():
        if method.operation == "correction" and method.diagnostic_source_id:
            result.setdefault(method.benchmark_id, set()).add(method.diagnostic_source_id)
    return result


def _prompt(fmt: str, row: dict[str, Any]) -> Any:
    question = _question(row)
    if fmt == "bfcl":
        return {"messages": health_messages(row, include_final_assistant=False), "tools": row.get("tools") or []}
    if fmt == "health":
        return health_messages(row, include_final_assistant=False)
    if fmt == "math":
        return wrap_math_user(question)
    if fmt == "code":
        solution = str(first_present(row, ("canonical_solution", "solution", "answer", "code")) or "")
        entry_point = code_verifier_fields(row)["entry_point"]
        code_prompt, _, _ = prepare_humaneval_sft(
            question=question,
            solution=solution,
            entry_point=entry_point,
        )
        return wrap_humaneval_user(code_prompt)
    return wrap_gpqa_user(question, _choices(row))


def _correct(fmt: str, completion: str, row: dict[str, Any], critic: Any | None = None):
    if fmt == "bfcl":
        gold_calls = row.get("tool_calls")
        if not isinstance(gold_calls, list):
            gold_calls = next(
                (
                    message.get("tool_calls")
                    for message in reversed(row.get("messages") or [])
                    if isinstance(message, dict) and message.get("role") == "assistant"
                ),
                [],
            )
        return verify_tool_calls(parse_tool_calls(completion), row.get("tools") or [], gold_calls)
    if fmt == "health":
        if critic is None:
            raise RuntimeError("Health diagnostics require the framework reference critic")
        return verify_health_answer(
            health_messages(row, include_final_assistant=False), completion, _answer(row), critic
        )
    if fmt == "code":
        reference_solution = str(first_present(row, ("canonical_solution", "solution", "answer", "code")) or "")
        fields = code_verifier_fields(row)
        code_prompt, _, _ = prepare_humaneval_sft(
            question=_question(row),
            solution=reference_solution,
            entry_point=fields["entry_point"],
        )
        return verify_humaneval_sft_completion(
            prompt=code_prompt,
            completion=completion,
            tests=fields["tests"],
            entry_point=fields["entry_point"],
        )
    if fmt == "multiple_choice":
        return MultipleChoiceVerifier()({"answer": completion, "choices": _choices(row)}, row)
    return MathExactMatchVerifier()({"answer": completion}, row)


def _wrong_row(fmt: str, completion: str, row: dict[str, Any], feedback: str, checkpoint: str) -> dict[str, Any]:
    result = dict(row)
    result.update({
        "question": _question(row),
        "model_output": completion,
        "wrong_solution": completion,
        "student_ckpt": checkpoint,
        "verifier_feedback": feedback,
    })
    if fmt == "code":
        result["wrong_code"] = extract_code(completion)
    return result


def _validate_code_diagnostic_rows(rows: list[dict[str, Any]], source_id: str) -> None:
    for index, row in enumerate(rows):
        fields = code_verifier_fields(row)
        entry_point = fields["entry_point"].strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", entry_point):
            raise ValueError(
                f"code diagnostic source {source_id!r} row {index} requires a bare Python function name in entry_point"
            )
        reference_solution = first_present(row, ("canonical_solution", "solution", "answer", "code"))
        if not reference_solution:
            raise ValueError(f"code diagnostic source {source_id!r} row {index} requires a reference solution")
        try:
            prompt, completion, _ = prepare_humaneval_sft(
                question=_question(row),
                solution=str(reference_solution),
                entry_point=entry_point,
            )
        except ValueError as exc:
            raise ValueError(
                f"code diagnostic source {source_id!r} row {index} cannot canonicalize its reference: {exc}"
            ) from exc
        result = verify_humaneval_sft_completion(
            prompt=prompt,
            completion=completion,
            tests=fields["tests"],
            entry_point=entry_point,
        )
        if not result.passed:
            raise ValueError(
                f"code diagnostic source {source_id!r} row {index} reference solution failed its tests: {result.feedback}"
            )


def _generate_transformers(
    checkpoint_path: str,
    prompts: list[Any],
    config: dict[str, Any],
    on_completion: Callable[[int, str], None] | None = None,
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = None
    try:
        with _model_for_evaluation(checkpoint_path) as model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            outputs: list[str] = []
            for prompt_index, prompt_item in enumerate(prompts):
                tools = None
                if isinstance(prompt_item, dict) and isinstance(prompt_item.get("messages"), list):
                    messages = prompt_item["messages"]
                    raw_tools = prompt_item.get("tools")
                    tools = raw_tools if isinstance(raw_tools, list) else None
                elif isinstance(prompt_item, list):
                    messages = prompt_item
                else:
                    messages = [{"role": "user", "content": str(prompt_item)}]
                try:
                    template_kwargs = {"tools": tools} if tools else {}
                    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
                except Exception:
                    rendered = prompt_item if isinstance(prompt_item, str) else json.dumps(prompt_item, ensure_ascii=False)
                encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=int(config.get("max_tokens", 1024)),
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                completion = tokenizer.decode(
                    generated[0][encoded["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
                outputs.append(completion)
                if on_completion is not None:
                    on_completion(prompt_index, completion)
            return outputs
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _generate(
    fmt: str,
    checkpoint_path: str,
    prompts: list[str],
    config: dict[str, Any],
    log_dir: Path,
) -> list[str]:
    del fmt, log_dir
    return _generate_transformers(checkpoint_path, prompts, config)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _collect_source_diagnostic(
    run_dir: Path,
    recipe_id: str,
    checkpoint_path: str,
    domain: str,
    source_id: str,
    source: SourceSpec,
    config: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    rows = load_raw_source(source, limit)
    if not rows:
        raise RuntimeError(f"diagnostic source {source_id!r} is empty")
    fmt = task_format(source, rows[0])
    if fmt == "math":
        rows = [with_resolved_source_target(row, source) for row in rows]
    if fmt == "code":
        _validate_code_diagnostic_rows(rows, source_id)
    output_dir = run_dir / "diagnostics" / _safe_name(recipe_id) / _safe_name(domain) / _safe_name(source_id)
    wrong_path = output_dir / "wrong_examples.jsonl"
    all_path = output_dir / "all_generations.jsonl"
    manifest_path = output_dir / "manifest.json"
    existing_rows = read_json_rows(all_path) if all_path.is_file() else []
    completed_indices = {
        int(record["source_index"])
        for record in existing_rows
        if isinstance(record.get("source_index"), int)
    }
    pending_indices = [index for index in range(len(rows)) if index not in completed_indices]
    critic = FrameworkGenerator({}) if fmt == "health" else None

    def persist(index: int, completion: str) -> None:
        row = rows[index]
        result = _correct(fmt, completion, row, critic)
        record = {
            "source_index": index,
            "question": _question(row),
            "model_output": completion,
            "correct": result.passed,
            "verifier_feedback": result.feedback,
        }
        append_jsonl(all_path, record)
        if not result.passed:
            wrong = _wrong_row(fmt, completion, row, result.feedback, checkpoint_path)
            wrong["source_index"] = index
            append_jsonl(wrong_path, wrong)

    try:
        prompts = [_prompt(fmt, rows[index]) for index in pending_indices]
        _generate_transformers(
            checkpoint_path,
            prompts,
            config,
            on_completion=lambda local_index, completion: persist(
                pending_indices[local_index], completion
            ),
        )
    finally:
        if critic is not None:
            critic.close()
    wrong_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_path.touch(exist_ok=True)
    all_rows = sorted(read_json_rows(all_path), key=lambda record: int(record["source_index"]))
    wrong_rows = sorted(
        read_json_rows(wrong_path) if wrong_path.is_file() else [],
        key=lambda record: int(record["source_index"]),
    )
    if len(all_rows) != len(rows):
        raise RuntimeError(
            f"diagnostic completed {len(all_rows)} of {len(rows)} source rows"
        )
    failure_modes: dict[str, int] = {}
    for wrong in wrong_rows:
        key = str(wrong.get("verifier_feedback") or "unknown")[:240]
        failure_modes[key] = failure_modes.get(key, 0) + 1
    entry = {
        "status": "completed",
        "usable": bool(wrong_rows),
        "recipe_id": recipe_id,
        "domain": domain,
        "benchmark_id": domain,
        "diagnostic_source_id": source_id,
        "checkpoint_path": checkpoint_path,
        "wrong_examples_path": str(wrong_path),
        "all_generations_path": str(all_path),
        "manifest_path": str(manifest_path),
        "n_total": len(all_rows),
        "n_wrong": len(wrong_rows),
        "wrong_rate": len(wrong_rows) / len(all_rows),
        "failure_modes": failure_modes,
        "representative_failures": [
            {
                "question": row.get("question"),
                "model_output": row.get("model_output"),
                "verifier_feedback": row.get("verifier_feedback"),
            }
            for row in wrong_rows[:3]
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entry


def _diagnostic_worker(request_path: Path, response_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    entry = _collect_source_diagnostic(
        run_dir=Path(request["run_dir"]),
        recipe_id=str(request["recipe_id"]),
        checkpoint_path=str(request["checkpoint_path"]),
        domain=str(request["domain"]),
        source_id=str(request["source_id"]),
        source=SourceSpec.model_validate(request["source"]),
        config=dict(request.get("config") or {}),
        limit=int(request["limit"]),
    )
    response_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _collect_source_diagnostic_subprocess(
    run_dir: Path,
    recipe_id: str,
    checkpoint_path: str,
    domain: str,
    source_id: str,
    source: SourceSpec,
    config: dict[str, Any],
    limit: int,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    worker_timeout = float(config.get("worker_timeout", config.get("timeout", 3600)))
    if timeout_seconds is not None:
        if timeout_seconds <= 0:
            raise TimeoutError(f"diagnostic worker has no remaining time for source {source_id!r}")
        worker_timeout = min(worker_timeout, float(timeout_seconds))
    with tempfile.TemporaryDirectory(prefix="mdpt_diagnostic_") as temp_dir:
        request_path = Path(temp_dir) / "request.json"
        response_path = Path(temp_dir) / "response.json"
        request_path.write_text(json.dumps({
            "run_dir": str(run_dir),
            "recipe_id": recipe_id,
            "checkpoint_path": checkpoint_path,
            "domain": domain,
            "source_id": source_id,
            "source": source.model_dump(mode="json"),
            "config": config,
            "limit": limit,
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-m", "mdpt.execution.diagnostics", _DIAGNOSTIC_WORKER_FLAG,
             str(request_path), str(response_path)],
            cwd=str(Path(__file__).resolve().parents[2]),
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=worker_timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise TimeoutError(
                f"diagnostic worker timed out after {worker_timeout} seconds for source {source_id!r}"
            ) from exc
        _terminate_process_group(process)
        if returncode != 0:
            raise RuntimeError(f"diagnostic worker exited with code {returncode} for source {source_id!r}")
        if not response_path.is_file():
            raise RuntimeError(f"diagnostic worker produced no response for source {source_id!r}")
        entry = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(entry, dict):
            raise RuntimeError(f"diagnostic worker returned an invalid response for source {source_id!r}")
        return entry


def collect_diagnostics(
    run_dir: Path,
    recipe_id: str,
    checkpoint_path: str,
    methods: dict[str, DataMethodSpec],
    catalog: SourceCatalog,
    runtime: dict[str, Any],
    requested_sources: dict[str, set[str]] | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    requested = requested_sources if requested_sources is not None else diagnostic_sources(methods)
    if not requested:
        return {}
    config = dict(runtime.get("diagnostics") or {})
    config["max_tokens"] = DIAGNOSTIC_MAX_TOKENS
    limit = DIAGNOSTIC_SOURCE_LIMIT
    source_map = catalog.by_id()
    stats: dict[str, Any] = {}
    index_path = run_dir / "diagnostics" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"entries": []}

    for domain, source_ids in requested.items():
        stats[domain] = {}
        for source_id in sorted(source_ids):
            matching = [
                entry
                for entry in reversed(index["entries"])
                if entry.get("domain") == domain
                and entry.get("diagnostic_source_id") == source_id
                and entry.get("checkpoint_path") == checkpoint_path
            ]
            existing = next(
                (
                    entry
                    for entry in matching
                    if entry.get("status", "completed") == "completed"
                    and Path(str(entry.get("manifest_path", ""))).is_file()
                ),
                None,
            )
            if existing is not None:
                stats[domain][source_id] = existing
                continue
            latest_matching = matching[0] if matching else None
            if latest_matching is not None and (
                latest_matching.get("status") == "failed"
                or not latest_matching.get(
                    "usable", int(latest_matching.get("n_wrong", 0)) > 0
                )
            ):
                stats[domain][source_id] = latest_matching
                continue
            source: SourceSpec = source_map[source_id]
            remaining = (
                deadline_monotonic - time.monotonic()
                if deadline_monotonic is not None
                else None
            )
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Correction diagnostics exceeded the Trial/Search deadline")
            try:
                entry = _collect_source_diagnostic_subprocess(
                    run_dir,
                    recipe_id,
                    checkpoint_path,
                    domain,
                    source_id,
                    source,
                    config,
                    limit,
                    timeout_seconds=remaining,
                )
            except TimeoutError:
                raise
            except Exception as exc:
                failed_entry = {
                    "status": "failed",
                    "usable": False,
                    "recipe_id": recipe_id,
                    "domain": domain,
                    "benchmark_id": domain,
                    "diagnostic_source_id": source_id,
                    "checkpoint_path": checkpoint_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                index["entries"].append(failed_entry)
                stats[domain][source_id] = failed_entry
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                continue
            index["entries"].append(entry)
            stats[domain][source_id] = entry
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def _main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] != _DIAGNOSTIC_WORKER_FLAG:
        raise SystemExit("diagnostics.py is an internal worker; expected --diagnostic-worker REQUEST RESPONSE")
    _diagnostic_worker(Path(sys.argv[2]), Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
