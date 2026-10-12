from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from mdpt.discovery.adapters import get_augmentation_adapter
from mdpt.discovery.schema import DataMethodSpec, SourceSpec
from mdpt.discovery.sources.data import load_raw_source, with_resolved_source_target
from mdpt.discovery.methods.engine import _output_row, generate_method_example, task_format
from mdpt.discovery.methods.generation import FrameworkGenerator


def method_contract_digest(method: DataMethodSpec) -> str:
    encoded = json.dumps(method.model_dump(), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class MethodAuditReport:
    method_id: str
    method_contract_digest: str
    benchmark: str
    requested: int = 0
    attempted: int = 0
    accepted: int = 0
    rows_considered: int = 0
    infrastructure_failures: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def required_accepted(self) -> int:
        target = self.requested or self.attempted
        return max(1, math.ceil(target * 0.6))

    @property
    def completed(self) -> bool:
        target = self.requested or self.attempted
        return target > 0 and self.attempted >= target

    @property
    def ok(self) -> bool:
        return self.completed and self.accepted >= self.required_accepted

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "method_contract_digest": self.method_contract_digest,
            "benchmark": self.benchmark,
            "requested": self.requested,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rows_considered": self.rows_considered,
            "infrastructure_failures": self.infrastructure_failures,
            "required_accepted": self.required_accepted,
            "pass_rate": self.accepted / self.attempted if self.attempted else 0.0,
            "completed": self.completed,
            "ok": self.ok,
            "errors": self.errors,
            "examples": self.examples,
        }


def _is_infrastructure_failure(exc: Exception) -> bool:
    """Separate transient generator/service failures from method semantics."""
    if isinstance(exc, (json.JSONDecodeError, TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__
    if name in {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"}:
        return True
    message = str(exc).lower()
    if isinstance(exc, ValueError) and ("generator response" in message or "json object" in message):
        return True
    markers = (
        "exhausted max_tokens",
        "returned empty content",
        "connection reset",
        "connection error",
        "rate limit",
        "timed out",
        "timeout",
        "http 429",
        "status code 429",
        "status code 500",
        "status code 502",
        "status code 503",
        "status code 504",
    )
    return any(marker in message for marker in markers)


def audit_method(
    method: DataMethodSpec,
    source: SourceSpec,
    runtime: dict[str, Any],
    *,
    sample_size: int = 5,
    generator_factory: Callable[[dict[str, Any]], FrameworkGenerator] = FrameworkGenerator,
) -> MethodAuditReport:
    if sample_size <= 0:
        raise ValueError("method audit sample_size must be positive")
    if method.benchmark_id != source.benchmark_id:
        raise ValueError(
            f"method {method.method_id!r} targets {method.benchmark_id!r}, "
            f"but source {source.source_id!r} targets {source.benchmark_id!r}"
        )
    adapter = get_augmentation_adapter(source.benchmark_id)
    report = MethodAuditReport(
        method_id=method.method_id,
        method_contract_digest=method_contract_digest(method),
        benchmark=adapter.benchmark,
        requested=sample_size,
    )
    contract_errors = adapter.validate_method(method)
    report.errors.extend(contract_errors)
    if contract_errors:
        return report
    # Load bounded replacement rows. Transient service failures must not become
    # evidence that the Agent-authored method is semantically bad.
    rows = load_raw_source(source, sample_size * 3)
    if not rows:
        report.errors.append("method input source is empty")
        return report
    fmt = task_format(source, rows[0])
    if fmt == "math":
        rows = [with_resolved_source_target(row, source) for row in rows]
    if method.operation == "correction":
        rows = [
            row
            | {
                "previous_attempt": "<no valid output>",
                "verifier_feedback": "missing or invalid benchmark output",
            }
            for row in rows
        ]
    generator = generator_factory(runtime)
    try:
        for index, row in enumerate(rows):
            if report.attempted >= sample_size:
                break
            report.rows_considered += 1
            try:
                generated, verification, prompts = generate_method_example(
                    method, source, row, generator
                )
                report.attempted += 1
                if verification.passed:
                    _output_row(fmt, generated, row, method)
                    report.accepted += 1
                else:
                    report.errors.append(f"row {index}: {verification.feedback or 'adapter rejected output'}")
                report.examples.append(
                    {
                        "row": index,
                        "passed": verification.passed,
                        "feedback": verification.feedback,
                        "generated": generated,
                        "stages": len(prompts),
                    }
                )
            except Exception as exc:
                detail = f"row {index}: {type(exc).__name__}: {exc}"
                if _is_infrastructure_failure(exc):
                    report.infrastructure_failures.append(detail)
                    continue
                report.attempted += 1
                report.errors.append(detail)
    finally:
        generator.close()
    if report.attempted < sample_size:
        report.errors.append(
            "method audit could not obtain enough completed semantic samples: "
            f"completed {report.attempted}/{sample_size} after considering "
            f"{report.rows_considered} source rows; rerun the unchanged method after "
            "the generation service recovers"
        )
    return report
