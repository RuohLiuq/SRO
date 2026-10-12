from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from mdpt.discovery.schema import DataMethodSpec, Recipe, SearchSpace, SourceCatalog, SourceSpec
from mdpt.discovery.formatters import (
    enforce_answer_format_gpqa,
    enforce_answer_format_math,
    wrap_gpqa_user,
    wrap_humaneval_user,
    wrap_math_user,
)


MAX_EXAMPLES_PER_SOURCE = 2000


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _first(row: dict[str, Any], fields: list[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return None


def resolved_source_target(row: dict[str, Any], source: SourceSpec) -> str | None:
    """Return the target explicitly mapped by the discovery Agent."""
    return _first(row, source.format.target_fields)


def with_resolved_source_target(row: dict[str, Any], source: SourceSpec) -> dict[str, Any]:
    """Expose a source-specific target under the framework verifier key."""
    target = resolved_source_target(row, source)
    if target is None or "gold_answer" in row:
        return row
    return {**row, "gold_answer": target}


def normalize_bfcl_training_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep one, structured representation of a BFCL tool-call example.

    The discovery sources follow PostTrainBench's pre-rendered representation:
    their system message already contains the ``# Tools`` block and their
    assistant content already contains a textual ``<tool_call>``. The Qwen
    chat template also renders both pieces from the top-level ``tools`` and
    assistant ``tool_calls`` fields. Passing both representations to the
    template duplicates the schema and target during SFT.

    Controller training uses the structured representation only. Unrelated
    system instructions are retained.
    """
    normalized = dict(row)
    messages: list[dict[str, Any]] = []
    for raw_message in row.get("messages") or []:
        message = dict(raw_message)
        content = str(message.get("content") or "")
        is_rendered_tools_system = (
            message.get("role") == "system"
            and content.lstrip().startswith("# Tools")
            and "<tools>" in content
            and "<tool_call>" in content
        )
        if is_rendered_tools_system:
            continue
        if message.get("role") == "assistant" and message.get("tool_calls"):
            message["content"] = ""
        messages.append(message)
    normalized["messages"] = messages
    return normalized


def canonicalize(row: dict[str, Any], source: SourceSpec) -> dict[str, Any] | None:
    fmt = source.format
    messages = row.get(fmt.messages_field) if fmt.messages_field else None
    messages = messages or row.get("messages") or row.get("dialogs")
    normalized: list[dict[str, Any]] = []
    if isinstance(messages, list) and messages:
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content") or "")
            tool_calls = message.get("tool_calls")
            if not content.strip() and not tool_calls:
                continue
            normalized_message: dict[str, Any] = {
                "role": str(message.get("role", "user")),
                "content": content,
            }
            if tool_calls:
                normalized_message["tool_calls"] = tool_calls
            if message.get("name"):
                normalized_message["name"] = str(message["name"])
            normalized.append(normalized_message)
        passthrough_formats = {"plain", "healthbench", "posttrainbench_bfcl"}
        if (
            normalized
            and normalized[-1]["role"] == "assistant"
            and fmt.formatter in passthrough_formats
        ):
            result: dict[str, Any] = {
                "messages": normalized,
                "_benchmark_id": source.benchmark_id,
                "_source_id": source.source_id,
            }
            if isinstance(row.get("tools"), list):
                result["tools"] = row["tools"]
            if fmt.formatter == "posttrainbench_bfcl":
                result = normalize_bfcl_training_row(result)
            return result

    message_user = next(
        (message["content"] for message in reversed(normalized) if message["role"] == "user"),
        None,
    )
    message_assistant = next(
        (message["content"] for message in reversed(normalized) if message["role"] == "assistant"),
        None,
    )
    message_system = next(
        (message["content"] for message in normalized if message["role"] == "system"),
        None,
    )
    user = _first(row, fmt.user_fields) or message_user
    assistant = _first(row, fmt.assistant_fields) or message_assistant
    if user is None or assistant is None:
        return None
    system = _first(row, fmt.system_fields) or message_system
    messages_out: list[dict[str, str]] = []
    if system:
        messages_out.append({"role": "system", "content": system})
    user_context = dict(row)
    user_context["text"] = user
    assistant_context = dict(row)
    assistant_context["text"] = assistant
    user_text = fmt.user_template.format_map(user_context)
    assistant_text = fmt.assistant_template.format_map(assistant_context)
    if fmt.formatter == "posttrainbench_math":
        user_text = wrap_math_user(user_text)
        assistant_text = enforce_answer_format_math(
            assistant_text,
            resolved_source_target(row, source),
        )
    elif fmt.formatter == "posttrainbench_humaneval":
        from mdpt.discovery.methods.code_execution import (
            code_verifier_fields,
            prepare_humaneval_sft,
        )

        fields = code_verifier_fields(row)
        code_prompt, assistant_text, _ = prepare_humaneval_sft(
            question=user_text,
            solution=assistant_text,
            entry_point=fields["entry_point"],
        )
        user_text = wrap_humaneval_user(code_prompt)
    elif fmt.formatter == "posttrainbench_gpqa":
        choices = row.get(fmt.choices_field)
        user_text = wrap_gpqa_user(user_text, choices)
        assistant_text = enforce_answer_format_gpqa(
            assistant_text,
            choices if isinstance(choices, list) else [],
        )
    messages_out.extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ])
    return {
        "messages": messages_out,
        "_benchmark_id": source.benchmark_id,
        "_source_id": source.source_id,
    }


def load_raw_source(source: SourceSpec, limit: int) -> list[dict[str, Any]]:
    if source.kind == "local":
        path = Path(_expand(source.path or "")).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"source file does not exist: {path}")
        if path.suffix == ".jsonl":
            return read_jsonl(path, limit)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        return [dict(row) for row in rows[:limit]]

    from datasets import load_dataset

    dataset = load_dataset(
        source.dataset,
        source.subset,
        split=source.split,
        revision=source.revision,
    )
    return [dict(dataset[index]) for index in range(min(limit, len(dataset)))]


def load_source(source: SourceSpec, limit: int) -> list[dict[str, Any]]:
    rows = [canonicalize(row, source) for row in load_raw_source(source, limit)]
    return [row for row in rows if row is not None]
