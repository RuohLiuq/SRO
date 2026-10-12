from __future__ import annotations

import json
from typing import Any

from mdpt.discovery.adapters.protocol import AugmentationAdapter, answer, choices, question, strict_keys
from mdpt.discovery.schema import DataMethodSpec
from mdpt.discovery.methods.verifiers import MultipleChoiceVerifier, VerificationResult


class GPQAAugmentationAdapter(AugmentationAdapter):
    benchmark = "gpqamain"

    @staticmethod
    def json_schema(fields: set[str]) -> str:
        shape: dict[str, Any] = {field: "<string>" for field in sorted(fields)}
        if "choices" in shape:
            shape["choices"] = ["<choice A>", "<choice B>", "<choice C>", "<choice D>"]
        return json.dumps(shape, ensure_ascii=False)

    def rewrite_context(self, row: dict[str, Any]) -> dict[str, Any]:
        return {"question": question(row), "choices": choices(row)}

    def rewrite_fields(self) -> set[str]:
        return {"question", "choices"}

    def answer_fields(self) -> set[str]:
        return {"answer"}

    def correction_context(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "question": question(row),
            "choices": choices(row),
            "previous_incorrect_attempt": str(
                row.get("wrong_solution") or row.get("model_output") or row.get("previous_attempt") or ""
            ),
            "verifier_feedback": str(row.get("verifier_feedback") or row.get("failure_mode") or ""),
        }

    def correction_fields(self) -> set[str]:
        return {"answer"}

    @staticmethod
    def _require_choices(values: Any) -> list[str]:
        if not isinstance(values, list) or len(values) != 4 or any(not str(item).strip() for item in values):
            raise ValueError("GPQA transformed task requires exactly four non-empty choices")
        return [str(item) for item in values]

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
            "choices": self._require_choices(transformed["choices"]),
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
        return {
            "question": question(reference),
            "choices": self._require_choices(choices(reference)),
            "answer": str(corrected["answer"]),
        }

    def verify(
        self,
        generated: dict[str, Any],
        reference: dict[str, Any],
        method: DataMethodSpec,
        generator: Any | None = None,
    ) -> VerificationResult:
        if method.operation == "correction":
            return MultipleChoiceVerifier()(generated, reference)

        del generator

        answer_text = str(generated.get("answer") or "").strip()
        reasoning = answer_text.upper().split("ANSWER:", 1)[0].strip()
        if "ANSWER:" not in answer_text.upper() or len(reasoning) < 40:
            return VerificationResult(False, "GPQA Rewrite requires substantive reasoning ending with ANSWER: A/B/C/D")
        verifier = MultipleChoiceVerifier()
        predicted = verifier.extract_predicted_label(
            answer_text, [str(item) for item in generated.get("choices") or []]
        )
        if predicted is None:
            return VerificationResult(False, "GPQA Rewrite requires a parseable ANSWER: A/B/C/D label")
        return VerificationResult(
            True,
            "",
            predicted,
            None,
            {"verifier": "deterministic_output_contract", "label_provenance": "teacher_labeled"},
        )
