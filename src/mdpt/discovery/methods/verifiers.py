"""Reusable verifiers for source-generation data augmentation strategies.

Verification is implemented as small callables so domains can share an augmentation
pipeline while selecting benchmark-appropriate checks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from mdpt.discovery.methods.code_execution import first_present, infer_entry_point, verify_python_solution
from mdpt.discovery.formatters import extract_final_number as _extract_final_number


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    feedback: str = ""
    predicted: Optional[str] = None
    gold: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Support tuple-style unpacking as ``passed, feedback``."""
        yield self.passed
        yield self.feedback


class Verifier(Protocol):
    def __call__(self, generated: dict[str, Any], reference: dict[str, Any]) -> VerificationResult:
        ...


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_number(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    extracted = _extract_final_number(raw, allow_plain="\n" not in raw)
    return str(extracted).strip() if extracted is not None else None


class MathExactMatchVerifier:
    """Final numeric-answer verifier used by GSM8K/AIME-style data."""

    def __call__(self, generated: dict[str, Any], reference: dict[str, Any]) -> VerificationResult:
        pred_text = first_present(
            generated,
            (
                "answer",
                "solution",
                "teacher_raw_answer",
                "corrected_solution",
                "response",
                "output",
            ),
        )
        explicit_gold = first_present(reference, ("gold_answer", "gt", "correct_answer"))
        gold_text = explicit_gold or first_present(
            reference, ("answer", "solution", "gold_solution", "answer_detail")
        )
        pred = _normalize_number(pred_text)
        gold = (
            _extract_final_number(str(explicit_gold), allow_plain=True)
            if explicit_gold is not None
            else _normalize_number(gold_text)
        )
        passed = bool(pred is not None and gold is not None and pred == gold)
        feedback = "" if passed else f"final answer mismatch: predicted={pred!r}, gold={gold!r}"
        return VerificationResult(passed=passed, feedback=feedback, predicted=pred, gold=gold)


class CodeExecutionVerifier:
    """HumanEval/MBPP-style Python verifier using generated or reference tests."""

    def __init__(self, timeout_s: float = 5.0) -> None:
        self.timeout_s = float(timeout_s)

    def __call__(self, generated: dict[str, Any], reference: dict[str, Any]) -> VerificationResult:
        prompt = str(
            first_present(generated, ("question", "prompt", "problem", "instruction"))
            or first_present(reference, ("question", "prompt", "problem", "instruction"))
            or ""
        ).strip()
        solution = str(
            first_present(generated, ("solution", "answer", "code", "corrected_code", "response", "output"))
            or ""
        ).strip()
        tests = str(
            first_present(generated, ("test", "tests", "unit_tests"))
            or first_present(reference, ("test", "tests", "unit_tests"))
            or ""
        ).strip()
        entry_point = str(
            first_present(generated, ("entry_point",))
            or first_present(reference, ("entry_point",))
            or infer_entry_point(prompt, solution)
            or ""
        ).strip()
        verification = verify_python_solution(
            prompt=prompt,
            solution=solution,
            tests=tests,
            entry_point=entry_point,
            timeout_s=self.timeout_s,
            require_explicit_entry_point=True,
        )
        return VerificationResult(
            passed=verification.passed,
            feedback=verification.feedback,
            predicted="pass" if verification.passed else "fail",
            gold="pass",
            details={
                "stdout": verification.stdout,
                "stderr": verification.stderr,
                "error": verification.error,
                "entry_point": entry_point,
            },
        )


_CHOICE_RE = re.compile(
    r"(?i)(?:ANSWER|FINAL\s*ANSWER|CORRECT\s*ANSWER)\s*[:：]\s*\(?\s*([A-D])\s*\)?\b"
)
_STANDALONE_CHOICE_RE = re.compile(r"(?i)(?:^|\n|\b)([A-D])(?:\.|\)|:|\s*$)")


class MultipleChoiceVerifier:
    """Verifier for GPQA/BFCL-like multiple-choice rows.

    It supports either letter labels (A-D) or answer-text matching against choices.
    """

    letters = ("A", "B", "C", "D")

    @staticmethod
    def normalize_choice_text(value: Any) -> str:
        text = _normalize_text(value).lower()
        text = re.sub(r"^[a-d][\.)]\s*", "", text, flags=re.IGNORECASE)
        return text

    @classmethod
    def _choices(cls, row: dict[str, Any]) -> list[str]:
        raw = row.get("choices") or row.get("options") or row.get("answer_choices")
        if isinstance(raw, list):
            values = []
            for item in raw:
                if isinstance(item, dict):
                    values.append(str(item.get("text") or item.get("value") or item.get("answer") or ""))
                else:
                    values.append(str(item))
            return values
        out = []
        for letter in cls.letters:
            value = first_present(row, (letter, letter.lower(), f"choice_{letter}", f"option_{letter}"))
            if value not in (None, ""):
                out.append(str(value))
        return out

    @classmethod
    def _label_from_text(cls, text: Any, choices: list[str]) -> Optional[str]:
        norm = cls.normalize_choice_text(text)
        if not norm:
            return None
        if len(norm) == 1 and norm.upper() in cls.letters:
            return norm.upper()
        for idx, choice in enumerate(choices[: len(cls.letters)]):
            if norm == cls.normalize_choice_text(choice):
                return cls.letters[idx]
        return None

    @classmethod
    def _gold_label(cls, reference: dict[str, Any], choices: list[str]) -> Optional[str]:
        raw = first_present(
            reference,
            (
                "gold_label",
                "label",
                "target",
                "answer_label",
                "correct_label",
                "correct_choice",
            ),
        )
        label = cls._label_from_text(raw, choices)
        if label:
            return label
        correct_text = first_present(reference, ("correct_answer", "gold_answer", "answer"))
        return cls._label_from_text(correct_text, choices)

    @classmethod
    def extract_predicted_label(cls, text: Any, choices: list[str]) -> Optional[str]:
        raw = str(text or "").strip()
        if not raw:
            return None
        match = _CHOICE_RE.search(raw)
        if match:
            return match.group(1).upper()
        matches = _STANDALONE_CHOICE_RE.findall(raw)
        if matches:
            return matches[-1].upper()
        return cls._label_from_text(raw, choices)

    def __call__(self, generated: dict[str, Any], reference: dict[str, Any]) -> VerificationResult:
        choices = self._choices(generated) or self._choices(reference)
        answer_text = first_present(
            generated,
            ("answer", "solution", "response", "teacher_raw_answer", "corrected_solution", "output"),
        )
        pred = self.extract_predicted_label(answer_text, choices)
        gold = self._gold_label(reference, choices)
        passed = bool(pred is not None and gold is not None and pred == gold)
        feedback = "" if passed else f"choice mismatch: predicted={pred!r}, gold={gold!r}"
        return VerificationResult(
            passed=passed,
            feedback=feedback,
            predicted=pred,
            gold=gold,
            details={"choices": choices},
        )
