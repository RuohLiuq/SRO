from __future__ import annotations

import json
import re
from typing import Any

from mdpt.discovery.adapters.protocol import AugmentationAdapter, strict_keys
from mdpt.discovery.schema import DataMethodSpec
from mdpt.discovery.methods.verifiers import VerificationResult


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("messages") or row.get("dialogs") or []
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _tools(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("tools") or []
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _gold_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    direct = row.get("tool_calls")
    if isinstance(direct, list):
        return [dict(item) for item in direct if isinstance(item, dict)]
    for message in reversed(_messages(row)):
        calls = message.get("tool_calls")
        if message.get("role") == "assistant" and isinstance(calls, list):
            return [dict(item) for item in calls if isinstance(item, dict)]
    return []


def _prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = _messages(row)
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return messages


def _function_payload(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str(payload.get("name") or "").strip()
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("tool-call arguments must be a JSON object")
    return name, arguments


def normalize_tool_calls(calls: Any) -> list[tuple[str, str]]:
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise ValueError("BFCL exec_simple requires exactly one tool call")
    name, arguments = _function_payload(calls[0])
    if not name:
        raise ValueError("tool call has no function name")
    return [(name, json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")))]


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    cleaned = str(text or "").strip()
    tagged = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", cleaned, flags=re.DOTALL)
    candidate = tagged.group(1) if tagged else cleaned
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        return []
    try:
        value = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return []
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    elif isinstance(value, dict):
        value = [value]
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _schema_functions(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    functions: dict[str, dict[str, Any]] = {}
    for tool in tools:
        payload = tool.get("function") if tool.get("type") == "function" else tool
        if isinstance(payload, dict) and str(payload.get("name") or "").strip():
            functions[str(payload["name"])] = payload
    return functions


def verify_tool_calls(calls: Any, tools: list[dict[str, Any]], gold_calls: Any | None = None) -> VerificationResult:
    try:
        normalized = normalize_tool_calls(calls)
        name, arguments = _function_payload(calls[0])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return VerificationResult(False, str(exc))
    functions = _schema_functions(tools)
    if name not in functions:
        return VerificationResult(False, f"function {name!r} is absent from tools")
    schema = functions[name].get("parameters") or {}
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    required = schema.get("required") if isinstance(schema, dict) else []
    if not isinstance(properties, dict) or not isinstance(required, list):
        return VerificationResult(False, "function parameters schema is invalid")
    missing = [key for key in required if key not in arguments]
    unknown = [key for key in arguments if key not in properties]
    if missing or unknown:
        return VerificationResult(False, f"arguments schema mismatch missing={missing}, unknown={unknown}")
    if gold_calls is not None:
        try:
            gold = normalize_tool_calls(gold_calls)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return VerificationResult(False, f"invalid hidden reference tool call: {exc}")
        if normalized != gold:
            return VerificationResult(False, "tool call does not match the hidden verified reference", normalized, gold)
    else:
        gold = normalized
    return VerificationResult(True, "", normalized, gold)


class BFCLAugmentationAdapter(AugmentationAdapter):
    benchmark = "bfcl"

    @staticmethod
    def json_schema(fields: set[str]) -> str:
        shape: dict[str, Any] = {field: "<value>" for field in sorted(fields)}
        if "messages" in shape:
            shape["messages"] = [{"role": "user", "content": "<semantically equivalent request>"}]
        if "tool_calls" in shape:
            shape["tool_calls"] = [{"type": "function", "function": {"name": "<name>", "arguments": {}}}]
        return json.dumps(shape, ensure_ascii=False)

    def rewrite_context(self, row: dict[str, Any]) -> dict[str, Any]:
        return {"messages": _prompt_messages(row), "tools": _tools(row)}

    def rewrite_fields(self) -> set[str]:
        return {"messages"}

    def answer_fields(self) -> set[str]:
        return {"tool_calls"}

    def answer_context(
        self,
        transformed: dict[str, Any],
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        return {"messages": transformed["messages"], "tools": _tools(reference)}

    def correction_context(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": _prompt_messages(row),
            "tools": _tools(row),
            "previous_incorrect_attempt": str(row.get("model_output") or row.get("previous_attempt") or ""),
            "verifier_feedback": str(row.get("verifier_feedback") or row.get("failure_mode") or ""),
        }

    def correction_fields(self) -> set[str]:
        return {"tool_calls"}

    @staticmethod
    def _require_prompt_messages(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value or not isinstance(value[-1], dict) or value[-1].get("role") != "user":
            raise ValueError("BFCL messages must be a non-empty conversation ending in a user request")
        if any(not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"} for item in value):
            raise ValueError("BFCL messages contain an invalid turn")
        return [dict(item) for item in value]

    def assemble_rewrite(self, transformed: dict[str, Any], supervision: dict[str, Any] | None,
                         reference: dict[str, Any], method: DataMethodSpec) -> dict[str, Any]:
        del method
        strict_keys(transformed, {"messages"}, stage="rewrite")
        strict_keys(supervision or {}, {"tool_calls"}, stage="answer")
        messages = self._require_prompt_messages(transformed["messages"])
        calls = (supervision or {})["tool_calls"]
        return {"messages": messages + [{"role": "assistant", "content": "", "tool_calls": calls}], "tools": _tools(reference), "tool_calls": calls}

    def assemble_correction(self, corrected: dict[str, Any], reference: dict[str, Any],
                            method: DataMethodSpec) -> dict[str, Any]:
        del method
        strict_keys(corrected, {"tool_calls"}, stage="correction")
        calls = corrected["tool_calls"]
        return {"messages": _prompt_messages(reference) + [{"role": "assistant", "content": "", "tool_calls": calls}], "tools": _tools(reference), "tool_calls": calls}

    def verify(self, generated: dict[str, Any], reference: dict[str, Any], method: DataMethodSpec,
               generator: Any | None = None) -> VerificationResult:
        del generator
        hidden_gold = _gold_calls(reference) if method.operation == "correction" else None
        structural = verify_tool_calls(generated.get("tool_calls"), _tools(generated), hidden_gold)
        if structural.passed:
            structural.details["verifier"] = (
                "hidden_reference_exact_match"
                if method.operation == "correction"
                else "deterministic_tool_schema"
            )
        return structural
