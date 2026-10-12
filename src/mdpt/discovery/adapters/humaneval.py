from __future__ import annotations

from typing import Any

from mdpt.discovery.adapters.protocol import AugmentationAdapter, answer, question, strict_keys
from mdpt.discovery.schema import DataMethodSpec
from mdpt.discovery.methods.code_execution import (
    build_code_task_prompt,
    build_solution_program,
    code_verifier_fields,
    extract_code,
    verify_python_solution,
)
from mdpt.discovery.methods.verifiers import VerificationResult


class HumanEvalAugmentationAdapter(AugmentationAdapter):
    benchmark = "humaneval"

    def rewrite_context(self, row: dict[str, Any]) -> dict[str, Any]:
        fields = code_verifier_fields(row)
        return {
            "question": question(row),
            "entry_point": fields["entry_point"],
        }

    def rewrite_fields(self) -> set[str]:
        return {"question", "entry_point"}

    def answer_fields(self) -> set[str]:
        return {"solution", "tests"}

    def correction_context(self, row: dict[str, Any]) -> dict[str, Any]:
        fields = code_verifier_fields(row)
        return {
            "question": question(row),
            "entry_point": fields["entry_point"],
            "previous_incorrect_attempt": str(
                row.get("wrong_solution") or row.get("wrong_code") or row.get("model_output") or row.get("previous_attempt") or ""
            ),
            "verifier_feedback": str(row.get("verifier_feedback") or row.get("failure_mode") or ""),
        }

    def correction_fields(self) -> set[str]:
        return {"solution"}

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
        entry_point = str(transformed["entry_point"])
        solution = str((supervision or {})["solution"])
        tests = str((supervision or {})["tests"])
        program = build_solution_program(
            extract_code(solution),
            prompt=str(transformed["question"]),
            entry_point=entry_point,
        )
        prompt = build_code_task_prompt(str(transformed["question"]), program, entry_point)
        return {
            "question": prompt,
            "entry_point": entry_point,
            "solution": program,
            "tests": extract_code(tests),
        }

    def assemble_correction(
        self,
        corrected: dict[str, Any],
        reference: dict[str, Any],
        method: DataMethodSpec,
    ) -> dict[str, Any]:
        del method
        strict_keys(corrected, self.correction_fields(), stage="correction")
        fields = code_verifier_fields(reference)
        entry_point = fields["entry_point"]
        program = build_solution_program(
            extract_code(str(corrected["solution"])),
            prompt=question(reference),
            entry_point=entry_point,
        )
        prompt = build_code_task_prompt(question(reference), program, entry_point)
        return {
            "question": prompt,
            "entry_point": entry_point,
            "solution": program,
            "tests": fields["tests"],
        }

    def verify(
        self,
        generated: dict[str, Any],
        reference: dict[str, Any],
        method: DataMethodSpec,
        generator: Any | None = None,
    ) -> VerificationResult:
        del reference, generator
        result = verify_python_solution(
            prompt=str(generated.get("question") or ""),
            solution=str(generated.get("solution") or ""),
            tests=str(generated.get("tests") or ""),
            entry_point=str(generated.get("entry_point") or ""),
            require_explicit_entry_point=True,
        )
        details = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
        }
        if method.operation == "rewrite":
            details["label_provenance"] = "teacher_labeled_and_execution_verified"
        return VerificationResult(
            passed=result.passed,
            feedback=result.feedback,
            predicted="pass" if result.passed else "fail",
            gold="pass",
            details=details,
        )
