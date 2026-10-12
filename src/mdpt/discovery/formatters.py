from __future__ import annotations

import re
from typing import Any


MATH_USER_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your response should be of the form '
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    '{prompt}\n\nRemember to put your answer on its own line at the end in the form '
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need '
    'to use a \\boxed command.\n\nReasoning:'
)
HUMANEVAL_INSTRUCTION = (
    "Read the following function signature and docstring, and fully implement\n"
    "the function described. Your response should only contain the code for\n"
    "this function.\n"
)
GPQA_USER_TEMPLATE = (
    'Answer the following multiple choice question. Reason step by step. The last line of your response '
    'should be of the form "ANSWER: $LETTER" (without quotes), where $LETTER is one of A, B, C, or D.\n\n'
    'Question:\n{question}\n\nChoices:\n{choices}\n\nReasoning:'
)
ANSWER_LINE = re.compile(r"(?i)ANSWER\s*:\s*([^\n]+)\s*\Z")
GSM8K_TAIL = re.compile(r"####\s*([\-+]?\d[\d,\.]*)\s*$")
PYTHON_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
CHOICE_LINE = re.compile(r"(?i)(?:ANSWER|FINAL\s*ANSWER|CORRECT\s*ANSWER)\s*[:：]\s*\(?\s*([A-D])\s*\)?\b")


def _clean_math_target(value: str) -> str:
    target = value.strip().rstrip(".,")
    if re.fullmatch(r"\$[\-+]?\d[\d,\.]*", target):
        target = target[1:]
    return target.replace(",", "").strip()


def extract_final_number(text: str, *, allow_plain: bool = False) -> str | None:
    """Extract only an explicit final target.

    Free-form reasoning is never searched for a last number and boxed notation is
    intentionally not interpreted here. External-source preprocessing owns that
    conversion and must expose an explicit target to the adapter.
    """
    value = str(text or "").strip()
    for pattern in (ANSWER_LINE, GSM8K_TAIL):
        match = pattern.search(value)
        if match:
            return _clean_math_target(match.group(1))
    if allow_plain and value and "\n" not in value:
        return _clean_math_target(value)
    return None


def enforce_answer_format_math(text: str, target: str | None = None) -> str:
    answer = (
        extract_final_number(target, allow_plain=True)
        if target is not None
        else extract_final_number(text)
    )
    if answer is None:
        return str(text or "").strip()
    body = GSM8K_TAIL.sub("", str(text or "").rstrip()).rstrip()
    body = re.sub(r"(?i)ANSWER\s*:\s*[^\n]+\s*\Z", "", body).rstrip()
    return f"{body}\n\nANSWER: {answer}" if body else f"ANSWER: {answer}"


def enforce_python_fence(text: str) -> str:
    value = str(text or "").strip()
    if PYTHON_FENCE.search(value):
        return value
    if any(token in value for token in ("def ", "import ", "class ")):
        return f"```python\n{value}\n```"
    return value


def _choice_label(value: Any, choices: list[str]) -> str | None:
    text = str(value or "").strip()
    match = CHOICE_LINE.search(text)
    if match:
        return match.group(1).upper()
    if len(text) == 1 and text.upper() in {"A", "B", "C", "D"}:
        return text.upper()
    normalized = re.sub(r"^[a-d][\.)]\s*", "", " ".join(text.lower().split()))
    for index, choice in enumerate(choices[:4]):
        candidate = re.sub(r"^[a-d][\.)]\s*", "", " ".join(str(choice).lower().split()))
        if normalized == candidate:
            return chr(ord("A") + index)
    return None


def enforce_answer_format_gpqa(text: str, choices: list[str] | None = None) -> str:
    label = _choice_label(text, choices or [])
    if not label:
        return str(text or "").strip()
    body = re.sub(r"(?i)ANSWER\s*[:：]\s*[A-D]\s*\Z", "", str(text or "").strip()).rstrip()
    return f"{body}\n\nANSWER: {label}" if body else f"ANSWER: {label}"


def wrap_math_user(question: str) -> str:
    return MATH_USER_TEMPLATE.format(prompt=question.strip())


def wrap_humaneval_user(question: str) -> str:
    return HUMANEVAL_INSTRUCTION + question


def wrap_gpqa_user(question: str, choices: Any = None) -> str:
    values = [str(item.get("text") if isinstance(item, dict) else item).strip() for item in choices] if isinstance(choices, list) else []
    choice_text = "\n".join(f"{chr(ord('A') + index)}. {value}" for index, value in enumerate(values[:4]) if value)
    return GPQA_USER_TEMPLATE.format(question=question.strip(), choices=choice_text)
