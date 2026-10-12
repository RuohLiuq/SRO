from __future__ import annotations

import re
from typing import Any

from mdpt.discovery.adapters.protocol import AugmentationAdapter, answer, question, strict_keys
from mdpt.discovery.schema import DataMethodSpec
from mdpt.discovery.formatters import extract_final_number
from mdpt.discovery.methods.verifiers import MathExactMatchVerifier, VerificationResult


class MathAugmentationAdapter(AugmentationAdapter):
    def __init__(self, benchmark: str = "gsm8k") -> None:
        self.benchmark = benchmark

    def rewrite_context(self, row: dict[str, Any]) -> dict[str, Any]:
        return {"question": question(row)}

    def rewrite_fields(self) -> set[str]:
        return {"question"}

    def answer_fields(self) -> set[str]:
        return {"answer"}

    def correction_context(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "question": question(row),
            "previous_incorrect_attempt": str(
                row.get("wrong_solution") or row.get("model_output") or row.get("previous_attempt") or ""
            ),
            "verifier_feedback": str(row.get("verifier_feedback") or row.get("failure_mode") or ""),
        }

    def correction_fields(self) -> set[str]:
        return {"answer"}

    def assemble_rewrite(
        self,
        transformed: dict[str, Any],
        supervision: dict[str, Any] | None,
        reference: dict[str, Any],
        method: DataMethodSpec,
    ) -> dict[str, Any]:
        del reference, method
        strict_keys(transformed, self.rewrite_fields(), stage="rewrite")
        strict_keys(supervision or {}, self.answer_fields(), stage="answer")
        return {
            "question": str(transformed["question"]),
            "answer": str((supervision or {})["answer"]),
        }

    def assemble_correction(
        self,
        corrected: dict[str, Any],
        reference: dict[str, Any],
        method: DataMethodSpec,
    ) -> dict[str, Any]:
        del method
        strict_keys(corrected, self.correction_fields(), stage="correction")
        return {"question": question(reference), "answer": str(corrected["answer"])}

    def verify(
        self,
        generated: dict[str, Any],
        reference: dict[str, Any],
        method: DataMethodSpec,
        generator: Any | None = None,
    ) -> VerificationResult:
        if method.operation == "correction":
            return MathExactMatchVerifier()(generated, reference)

        del generator

        answer_text = str(generated.get("answer") or "").strip()
        target = extract_final_number(answer_text, allow_plain=False)
        if self.benchmark == "gsm8k":
            answer_match = re.search(
                r"(?i)(?<![A-Za-z])ANSWER\s*:\s*\$?[-+]?\d[\d,]*(?:\.\d+)?\s*[.]?\s*\Z",
                answer_text,
            )
        else:
            answer_match = re.search(r"(?im)^\s*ANSWER\s*:\s*[^\n]+\s*\Z", answer_text)
        if target is None or answer_match is None:
            return VerificationResult(False, "math Rewrite requires complete reasoning ending with ANSWER: <number>")
        reasoning = answer_text[: answer_match.start()].strip()
        if len(reasoning.split()) < 5:
            return VerificationResult(False, "math Rewrite answer has no substantive reasoning before ANSWER:")
        if self.benchmark == "aime2025":
            try:
                numeric = int(str(target))
            except ValueError:
                return VerificationResult(False, "AIME final answer must be an integer")
            if not 0 <= numeric <= 999:
                return VerificationResult(False, "AIME final answer must be in the range 0..999")
        return VerificationResult(
            True,
            "",
            target,
            None,
            {"verifier": "deterministic_output_contract", "label_provenance": "teacher_labeled"},
        )
