from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from mdpt.discovery.schema import DataMethodSpec
from mdpt.discovery.methods.code_execution import first_present
from mdpt.discovery.methods.verifiers import VerificationResult


def question(row: dict[str, Any]) -> str:
    value = first_present(row, ("question", "prompt", "problem", "instruction", "text"))
    if value not in (None, ""):
        return str(value).strip()
    messages = row.get("messages") or row.get("dialogs") or []
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def answer(row: dict[str, Any]) -> str:
    value = first_present(
        row,
        (
            "answer",
            "solution",
            "canonical_solution",
            "gold_solution",
            "gold_answer",
            "correct_answer",
            "response",
            "output",
        ),
    )
    if value not in (None, ""):
        return str(value).strip()
    messages = row.get("messages") or row.get("dialogs") or []
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def choices(row: dict[str, Any]) -> list[str]:
    raw = row.get("choices") or row.get("options") or row.get("answer_choices")
    if not isinstance(raw, list):
        return []
    return [
        str(item.get("text") or item.get("value") or "") if isinstance(item, dict) else str(item)
        for item in raw
    ]


def strict_keys(value: dict[str, Any], required: set[str], *, stage: str) -> None:
    actual = set(value)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        raise ValueError(
            f"{stage} JSON fields mismatch: required={sorted(required)}, "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    empty = [key for key in required if value.get(key) in (None, "", [])]
    if empty:
        raise ValueError(f"{stage} JSON has empty required fields: {sorted(empty)}")


class AugmentationAdapter(ABC):
    benchmark: str

    @abstractmethod
    def rewrite_context(self, row: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def rewrite_fields(self) -> set[str]: ...

    @abstractmethod
    def answer_fields(self) -> set[str]: ...

    def answer_context(
        self,
        transformed: dict[str, Any],
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        del reference
        return dict(transformed)

    def verification_prompt(self) -> str | None:
        return None

    def verification_context(
        self,
        transformed: dict[str, Any],
        supervision: dict[str, Any],
    ) -> dict[str, Any]:
        del transformed, supervision
        return {}

    def verification_fields(self) -> set[str]:
        return set()

    @abstractmethod
    def correction_context(self, row: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def correction_fields(self) -> set[str]: ...

    @abstractmethod
    def assemble_rewrite(
        self,
        transformed: dict[str, Any],
        supervision: dict[str, Any] | None,
        reference: dict[str, Any],
        method: DataMethodSpec,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def assemble_correction(
        self,
        corrected: dict[str, Any],
        reference: dict[str, Any],
        method: DataMethodSpec,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def verify(
        self,
        generated: dict[str, Any],
        reference: dict[str, Any],
        method: DataMethodSpec,
        generator: Any | None = None,
    ) -> VerificationResult: ...

    def validate_method(self, method: DataMethodSpec) -> list[str]:
        del method
        return []

    @staticmethod
    def json_schema(fields: set[str]) -> str:
        return json.dumps({field: "<string>" for field in sorted(fields)}, ensure_ascii=False)
