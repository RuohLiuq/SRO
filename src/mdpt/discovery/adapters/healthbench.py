from __future__ import annotations

import json
from typing import Any

from mdpt.discovery.adapters.protocol import AugmentationAdapter, strict_keys
from mdpt.discovery.schema import DataMethodSpec
from mdpt.discovery.methods.verifiers import VerificationResult


def health_messages(row: dict[str, Any], *, include_final_assistant: bool = True) -> list[dict[str, str]]:
    raw = row.get("messages") or row.get("dialogs") or row.get("prompt") or []
    messages = [
        {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
        for item in raw
        if isinstance(item, dict)
    ] if isinstance(raw, list) else []
    if not include_final_assistant and messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    return messages


def _parse_critic(text: str) -> tuple[bool, str]:
    cleaned = str(text or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        return False, "health critic returned no JSON verdict"
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        return False, f"health critic returned invalid JSON: {exc}"
    verdict = str(payload.get("verdict") or "").strip().upper()
    feedback = str(payload.get("feedback") or "").strip()
    return verdict == "PASS", feedback or ("reference-based health critic rejected answer" if verdict != "PASS" else "")


def verify_health_answer(
    messages: list[dict[str, str]],
    candidate: str,
    reference_answer: str | None,
    generator: Any,
) -> VerificationResult:
    if not messages or messages[-1].get("role") != "user":
        return VerificationResult(False, "Health prompt must end in a user turn")
    if not candidate.strip():
        return VerificationResult(False, "Health candidate answer must be non-empty")
    reference_guidance = (
        f"Hidden curated reference answer for the same conversation:\n{reference_answer}\n\n"
        if str(reference_answer or "").strip()
        else "No reference answer is available; judge the candidate directly against the conversation.\n\n"
    )
    prompt = (
        "You are an independent medical answer verifier for post-training data. "
        "This is not the HealthBench evaluator and you must not use or infer HealthBench evaluation rubrics. "
        "PASS only if the candidate directly answers this conversation, is medically sound, contains no "
        "material unsafe or unsupported claim, communicates uncertainty appropriately, and preserves essential "
        "actionable safety guidance. Return exactly JSON with fields verdict (PASS or FAIL) and feedback.\n\n"
        f"Conversation:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
        f"{reference_guidance}Candidate answer:\n{candidate}"
    )
    passed, feedback = _parse_critic(generator.complete(prompt))
    return VerificationResult(
        passed,
        feedback,
        "pass" if passed else "fail",
        "pass",
        {"critic": "non_official_reference_based"},
    )


class HealthBenchAugmentationAdapter(AugmentationAdapter):
    benchmark = "healthbench"

    @staticmethod
    def json_schema(fields: set[str]) -> str:
        shape: dict[str, Any] = {field: "<string>" for field in sorted(fields)}
        if "messages" in shape:
            shape["messages"] = [{"role": "user", "content": "<semantically equivalent health request>"}]
        return json.dumps(shape, ensure_ascii=False)

    def rewrite_context(self, row: dict[str, Any]) -> dict[str, Any]:
        return {"messages": health_messages(row, include_final_assistant=False)}

    def rewrite_fields(self) -> set[str]:
        return {"messages"}

    def answer_fields(self) -> set[str]:
        return {"answer"}

    def correction_context(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": health_messages(row, include_final_assistant=False),
            "previous_incorrect_attempt": str(row.get("model_output") or row.get("previous_attempt") or ""),
            "verifier_feedback": str(row.get("verifier_feedback") or row.get("failure_mode") or ""),
        }

    def correction_fields(self) -> set[str]:
        return {"answer"}

    @staticmethod
    def _require_prompt(value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            raise ValueError("Health rewrite messages must be a list")
        messages = [
            {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")}
            for item in value
            if isinstance(item, dict)
        ]
        if len(messages) != len(value) or not messages or messages[-1]["role"] != "user":
            raise ValueError("Health rewrite conversation must end in a user turn")
        if any(
            item["role"] not in {"system", "user", "assistant"} or not item["content"].strip()
            for item in messages
        ):
            raise ValueError("Health rewrite conversation has an invalid or empty turn")
        return messages

    def assemble_rewrite(
        self,
        transformed: dict[str, Any],
        supervision: dict[str, Any] | None,
        reference: dict[str, Any],
        method: DataMethodSpec,
    ) -> dict[str, Any]:
        del reference, method
        strict_keys(transformed, {"messages"}, stage="rewrite")
        strict_keys(supervision or {}, {"answer"}, stage="answer")
        messages = self._require_prompt(transformed["messages"])
        result_answer = str((supervision or {})["answer"]).strip()
        return {
            "messages": messages + [{"role": "assistant", "content": result_answer}],
            "answer": result_answer,
        }

    def assemble_correction(
        self,
        corrected: dict[str, Any],
        reference: dict[str, Any],
        method: DataMethodSpec,
    ) -> dict[str, Any]:
        del method
        strict_keys(corrected, {"answer"}, stage="correction")
        result_answer = str(corrected["answer"]).strip()
        messages = health_messages(reference, include_final_assistant=False)
        return {
            "messages": messages + [{"role": "assistant", "content": result_answer}],
            "answer": result_answer,
        }

    def verify(
        self,
        generated: dict[str, Any],
        reference: dict[str, Any],
        method: DataMethodSpec,
        generator: Any | None = None,
    ) -> VerificationResult:
        del reference, method, generator
        prompt_messages = health_messages(generated, include_final_assistant=False)
        candidate = str(generated.get("answer") or "").strip()
        if not prompt_messages or prompt_messages[-1].get("role") != "user":
            return VerificationResult(False, "Health prompt must end in a user turn")
        if not candidate:
            return VerificationResult(False, "Health candidate answer must be non-empty")
        return VerificationResult(
            True,
            "",
            "structurally_valid",
            None,
            {"verifier": "deterministic_conversation_structure"},
        )
