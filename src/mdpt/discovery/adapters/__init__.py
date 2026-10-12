"""Benchmark-aware Discovery contracts."""

from mdpt.discovery.adapters.registry import (
    AuditReport,
    BenchmarkAdapter,
    audit_source,
    get_adapter,
    get_augmentation_adapter,
)

__all__ = [
    "AuditReport",
    "BenchmarkAdapter",
    "audit_source",
    "get_adapter",
    "get_augmentation_adapter",
]
