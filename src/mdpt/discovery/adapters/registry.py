from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any

from mdpt.discovery.schema import SourceSpec
from mdpt.discovery.sources.data import canonicalize, load_raw_source, resolved_source_target
from mdpt.discovery.formatters import extract_final_number
from mdpt.benchmarks import canonical_benchmark
from mdpt.discovery.methods.code_execution import (
    code_verifier_fields,
    defines_entry_point,
    extract_code,
    first_present,
    function_signature,
    prepare_humaneval_sft,
    verify_humaneval_sft_completion,
)
from mdpt.discovery.methods.verifiers import MathExactMatchVerifier, MultipleChoiceVerifier
from mdpt.discovery.adapters.bfcl import BFCLAugmentationAdapter
from mdpt.discovery.adapters.gpqa import GPQAAugmentationAdapter
from mdpt.discovery.adapters.gsm8k import MathAugmentationAdapter
from mdpt.discovery.adapters.healthbench import HealthBenchAugmentationAdapter
from mdpt.discovery.adapters.humaneval import HumanEvalAugmentationAdapter
from mdpt.discovery.adapters.protocol import AugmentationAdapter


@dataclass(frozen=True)
class BenchmarkAdapter:
    benchmark: str
    required_formatter: str | None
    rewrite_supported: bool
    correction_supported: bool


@dataclass
class AuditReport:
    source_id: str
    adapter_id: str
    sampled: int = 0
    canonicalized: int = 0
    verified: int = 0
    errors: list[str] = field(default_factory=list)
    adaptations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.sampled > 0 and not self.errors and self.canonicalized == self.sampled

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "adapter_id": self.adapter_id,
            "sampled": self.sampled,
            "canonicalized": self.canonicalized,
            "verified": self.verified,
            "ok": self.ok,
            "errors": self.errors,
            "adaptations": self.adaptations,
        }


_ADAPTERS = {
    "gsm8k": BenchmarkAdapter("gsm8k", "posttrainbench_math", True, True),
    "aime2025": BenchmarkAdapter("aime2025", "posttrainbench_math", True, True),
    "humaneval": BenchmarkAdapter("humaneval", "posttrainbench_humaneval", True, True),
    "gpqamain": BenchmarkAdapter("gpqamain", "posttrainbench_gpqa", True, True),
    "arenahardwriting": BenchmarkAdapter("arenahardwriting", None, False, False),
    "healthbench": BenchmarkAdapter("healthbench", "healthbench", True, True),
    "bfcl": BenchmarkAdapter("bfcl", "posttrainbench_bfcl", True, True),
}


def get_adapter(value: str) -> BenchmarkAdapter:
    return _ADAPTERS[canonical_benchmark(value)]


_AUGMENTATION_ADAPTERS: dict[str, AugmentationAdapter] = {
    "gsm8k": MathAugmentationAdapter(),
    "aime2025": MathAugmentationAdapter("aime2025"),
    "humaneval": HumanEvalAugmentationAdapter(),
    "gpqamain": GPQAAugmentationAdapter(),
    "healthbench": HealthBenchAugmentationAdapter(),
    "bfcl": BFCLAugmentationAdapter(),
}


def get_augmentation_adapter(value: str) -> AugmentationAdapter:
    benchmark = canonical_benchmark(value)
    try:
        return _AUGMENTATION_ADAPTERS[benchmark]
    except KeyError as exc:
        raise ValueError(f"benchmark {benchmark!r} has no augmentation adapter") from exc


def _role_content(canonical: dict[str, Any], role: str) -> str:
    messages = canonical.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def _audit_python_training_row(
    canonical: dict[str, Any],
    report: AuditReport,
    *,
    scorer_facing: bool,
    entry_point: str = "",
) -> None:
    assistant = _role_content(canonical, "assistant")
    if not assistant.strip():
        report.errors.append(f"row {report.sampled - 1}: empty canonical Python assistant")
        return
    if scorer_facing:
        if assistant.lstrip().startswith("```"):
            report.errors.append(
                f"row {report.sampled - 1}: canonical HumanEval assistant must be unfenced complete Python"
            )
        if "\r" in assistant or "\t" in assistant:
            report.errors.append(
                f"row {report.sampled - 1}: canonical HumanEval assistant must use LF and spaces"
            )
        if entry_point and not defines_entry_point(assistant, entry_point):
            report.errors.append(
                f"row {report.sampled - 1}: canonical HumanEval assistant must define "
                f"the required top-level function {entry_point!r}"
            )
        return

    code = extract_code(assistant)
    try:
        ast.parse(code)
    except SyntaxError as exc:
        report.errors.append(f"row {report.sampled - 1}: canonical Python SyntaxError: {exc}")


def _audit_humaneval_row(
    row: dict[str, Any],
    source: SourceSpec,
    report: AuditReport,
    canonical: dict[str, Any],
    *,
    execute: bool,
) -> None:
    prompt = first_present(row, ("prompt", "question", "problem", "instruction", "text"))
    solution = _role_content(canonical, "assistant")
    fields = code_verifier_fields(row)
    tests = fields["tests"]
    entry_point = fields["entry_point"]
    missing = [
        name
        for name, value in (
            ("prompt/question", prompt),
            ("solution/assistant", solution),
            ("tests", tests),
            ("entry_point", entry_point),
        )
        if not str(value or "").strip()
    ]
    if missing:
        report.errors.append(f"row {report.sampled - 1}: missing {', '.join(missing)}")
        return
    if any(token in str(entry_point) for token in ("(", ")", " ", "\n", "\r")):
        report.errors.append(f"row {report.sampled - 1}: entry_point must be a bare function name")
        return
    if "def check" not in tests:
        report.errors.append(f"row {report.sampled - 1}: tests must define check(candidate)")
        return
    if execute:
        raw_solution = first_present(
            row,
            ("canonical_solution", "solution", "answer", "code", "response", "output"),
        )
        try:
            code_prompt, solution, _ = prepare_humaneval_sft(
                question=str(prompt),
                solution=str(raw_solution or ""),
                entry_point=entry_point,
            )
        except ValueError as exc:
            report.errors.append(
                f"row {report.sampled - 1}: cannot build HumanEval scorer prompt: {exc}"
            )
            return
        adaptation = (
            "preserved a complete implementation as complete canonical Python"
            if defines_entry_point(str(raw_solution or ""), entry_point)
            else "reconstructed a body-only implementation from the prompt signature"
            if function_signature(str(prompt), entry_point)
            else "reconstructed the canonical implementation"
        )
        if adaptation not in report.adaptations:
            report.adaptations.append(adaptation)
        result = verify_humaneval_sft_completion(
            prompt=code_prompt,
            completion=solution,
            tests=extract_code(tests),
            entry_point=entry_point,
        )
        if not result.passed:
            detail = result.error or result.stderr or result.stdout or "verification failed"
            report.errors.append(f"row {report.sampled - 1}: gold solution failed tests: {detail[:240]}")
            return
        report.verified += 1


def _audit_math_row(
    row: dict[str, Any],
    source: SourceSpec,
    report: AuditReport,
    canonical: dict[str, Any],
) -> None:
    index = report.sampled - 1
    if not source.format.target_fields:
        report.errors.append(
            f"row {index}: verifier-backed math source must declare format.target_fields; "
            "the framework will not infer a target from free-form reasoning"
        )
        return
    raw_target = resolved_source_target(row, source)
    if raw_target is None:
        report.errors.append(
            f"row {index}: none of target_fields {source.format.target_fields!r} contains a value"
        )
        return
    gold = extract_final_number(raw_target, allow_plain=True)
    if gold is None:
        report.errors.append(f"row {index}: explicit math target is empty")
        return
    assistant = _role_content(canonical, "assistant")
    result = MathExactMatchVerifier()(
        {"answer": assistant},
        {"gold_answer": gold},
    )
    if not result.passed:
        report.errors.append(
            f"row {index}: canonical final answer does not match explicit target: {result.feedback}"
        )
        return
    report.verified += 1


def _audit_gpqa_row(
    row: dict[str, Any],
    source: SourceSpec,
    report: AuditReport,
    canonical: dict[str, Any],
) -> None:
    index = report.sampled - 1
    choices = row.get(source.format.choices_field)
    if (
        not isinstance(choices, list)
        or len(choices) != 4
        or any(
            not str(choice.get("text") if isinstance(choice, dict) else choice).strip()
            for choice in choices
        )
    ):
        report.errors.append(f"row {index}: GPQA row requires exactly four non-empty choices")
        return
    assistant = _role_content(canonical, "assistant")
    result = MultipleChoiceVerifier()(
        {"answer": assistant, "choices": choices},
        row,
    )
    if not result.passed:
        report.errors.append(
            f"row {index}: canonical GPQA answer must preserve one explicit A-D gold label: "
            f"{result.feedback}"
        )
        return
    report.verified += 1


def _audit_health_row(
    canonical: dict[str, Any], report: AuditReport, raw: dict[str, Any] | None = None
) -> None:
    index = report.sampled - 1
    messages = canonical.get("messages") or []
    raw = raw or {}
    forbidden = sorted(set(raw) & {"rubrics", "ideal_completions_data", "canary"})
    if forbidden:
        report.errors.append(
            f"row {index}: official HealthBench evaluation fields are forbidden in "
            f"training/diagnostic sources: {forbidden}"
        )
        return
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        report.errors.append(f"row {index}: Health training conversation must end with assistant supervision")
        return
    if not any(message.get("role") == "user" for message in messages[:-1]):
        report.errors.append(f"row {index}: Health conversation requires at least one user turn")
        return
    if any(
        message.get("role") not in {"system", "user", "assistant"}
        or not str(message.get("content") or "").strip()
        for message in messages
    ):
        report.errors.append(f"row {index}: Health conversation has an invalid or empty turn")
        return
    report.verified += 1


def _audit_bfcl_row(
    row: dict[str, Any], canonical: dict[str, Any], report: AuditReport
) -> None:
    index = report.sampled - 1
    tools = canonical.get("tools") or row.get("tools")
    if not isinstance(tools, list) or not tools:
        report.errors.append(f"row {index}: BFCL row requires a non-empty tools list")
        return
    messages = canonical.get("messages") or []
    assistant = messages[-1] if messages else {}
    calls = assistant.get("tool_calls") or row.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        report.errors.append(f"row {index}: BFCL exec_simple row requires exactly one gold tool call")
        return
    functions: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if tool.get("type") == "function" else tool
        if isinstance(function, dict) and str(function.get("name") or "").strip():
            functions[str(function["name"])] = function
    call = calls[0] if isinstance(calls[0], dict) else {}
    function_call = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str(function_call.get("name") or "")
    if name not in functions:
        report.errors.append(f"row {index}: BFCL target function {name!r} is absent from tools")
        return
    arguments = function_call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            report.errors.append(f"row {index}: BFCL arguments are not valid JSON")
            return
    if not isinstance(arguments, dict):
        report.errors.append(f"row {index}: BFCL arguments must be an object")
        return
    schema = functions[name].get("parameters") or {}
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    required = schema.get("required") if isinstance(schema, dict) else []
    if not isinstance(properties, dict) or not isinstance(required, list):
        report.errors.append(f"row {index}: BFCL function parameters schema is invalid")
        return
    missing = [key for key in required if key not in arguments]
    unknown = [key for key in arguments if key not in properties]
    if missing or unknown:
        report.errors.append(
            f"row {index}: BFCL arguments schema mismatch missing={missing}, unknown={unknown}"
        )
        return
    report.verified += 1


def audit_source(
    source: SourceSpec,
    expected_benchmark: str,
    *,
    sample_size: int = 16,
    canonicalize_size: int | None = None,
    execute_code: bool = True,
    require_verifier: bool = True,
) -> AuditReport:
    expected = canonical_benchmark(expected_benchmark)
    if not source.adapter_id:
        raise ValueError(f"source {source.source_id!r} must declare adapter_id")
    declared = canonical_benchmark(source.adapter_id)
    if declared != expected:
        raise ValueError(
            f"source {source.source_id!r} adapter {declared!r} does not match domain benchmark {expected!r}"
        )
    adapter = get_adapter(expected)
    report = AuditReport(source_id=source.source_id, adapter_id=expected)
    if require_verifier and adapter.required_formatter and source.format.formatter != adapter.required_formatter:
        report.errors.append(
            f"formatter must be {adapter.required_formatter!r}, got {source.format.formatter!r}"
        )
        return report
    try:
        audit_limit = max(sample_size, canonicalize_size or sample_size)
        rows = load_raw_source(source, audit_limit)
    except Exception as exc:
        report.errors.append(f"source load failed: {type(exc).__name__}: {exc}")
        return report
    if not rows:
        report.errors.append("source is empty")
        return report
    for row in rows:
        report.sampled += 1
        try:
            canonical = canonicalize(row, source)
        except (TypeError, ValueError) as exc:
            report.errors.append(
                f"row {report.sampled - 1}: cannot canonicalize: {type(exc).__name__}: {exc}"
            )
            continue
        if canonical is None:
            report.errors.append(f"row {report.sampled - 1}: cannot canonicalize user/assistant messages")
            continue
        messages = canonical.get("messages") or []
        if not messages or messages[-1].get("role") != "assistant":
            report.errors.append(f"row {report.sampled - 1}: canonical form must end with assistant")
            continue
        report.canonicalized += 1
        if expected == "humaneval":
            entry_point = code_verifier_fields(row)["entry_point"]
            _audit_python_training_row(
                canonical,
                report,
                scorer_facing=source.format.formatter == "posttrainbench_humaneval",
                entry_point=entry_point,
            )
            if require_verifier:
                _audit_humaneval_row(row, source, report, canonical, execute=execute_code)
        elif expected in {"gsm8k", "aime2025"} and require_verifier:
            _audit_math_row(row, source, report, canonical)
        elif expected == "gpqamain" and require_verifier:
            _audit_gpqa_row(row, source, report, canonical)
        elif expected == "healthbench":
            _audit_health_row(canonical, report, row)
        elif expected == "bfcl":
            _audit_bfcl_row(row, canonical, report)
    return report
