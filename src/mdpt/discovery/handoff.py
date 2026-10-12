from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from mdpt.discovery.adapters import audit_source
from mdpt.discovery.methods.audit import method_contract_digest
from mdpt.discovery.schema import Recipe
from mdpt.discovery.state import (
    assert_initialized_configuration_current,
    load_initialized_artifacts,
    utc_now,
)
from mdpt.discovery.validation import digest, validate_recipe
from mdpt.io import read_json, write_json


def _required_source_ids(space: Any, methods: dict[str, Any]) -> set[str]:
    source_ids = {
        source_id
        for domain in space.domains.values()
        for benchmark in domain.benchmarks.values()
        for source_id in benchmark.source_choices
    }
    for method in methods.values():
        source_id = (
            method.input_source_id if method.operation == "rewrite" else method.diagnostic_source_id
        )
        if source_id:
            source_ids.add(source_id)
    return source_ids


def _run_root(workspace: Path) -> Path:
    if workspace.parent.name == "discovery":
        return workspace.parent.parent
    return workspace


def create_handoff(
    workspace: Path,
    initial_recipe_path: Path,
    *,
    source_sample_size: int = 16,
    execute_code: bool = True,
) -> dict[str, Any]:
    state, task, catalog, methods, space = load_initialized_artifacts(workspace)
    assert_initialized_configuration_current(state, task, catalog, methods, space)

    recipe = Recipe.model_validate(read_json(initial_recipe_path))
    validate_recipe(task, catalog, methods, space, recipe, allow_correction=False)

    sources = catalog.by_id()
    source_audits: list[dict[str, Any]] = []
    failures: list[str] = []
    per_source_limit = max(
        source_sample_size,
        int(os.environ.get("MDPT_PER_SOURCE_LIMIT", "2000")),
    )
    for source_id in sorted(_required_source_ids(space, methods)):
        source = sources[source_id]
        report = audit_source(
            source,
            source.benchmark_id,
            sample_size=source_sample_size,
            canonicalize_size=per_source_limit,
            execute_code=execute_code,
            require_verifier=True,
        )
        payload = report.as_dict()
        source_audits.append(payload)
        write_json(workspace / "audits" / "sources" / f"{source_id}.json", payload)
        if not report.ok:
            failures.append(f"{source_id}: " + "; ".join(report.errors[:4]))

    method_audits: list[dict[str, Any]] = []
    for method_id, method in sorted(methods.items()):
        contract_digest = method_contract_digest(method)
        path = workspace / "audits" / "methods" / method_id / f"{contract_digest}.json"
        if not path.is_file():
            failures.append(
                f"{method_id}: no passing audit exists for the current method contract"
            )
            continue
        report = read_json(path)
        method_audits.append(report)
        if (
            report.get("method_contract_digest") != contract_digest
            or report.get("benchmark") != method.benchmark_id
            or not report.get("ok")
        ):
            failures.append(f"{method_id}: current method audit is rejected or mismatched")

    if failures:
        raise RuntimeError("handoff validation failed:\n- " + "\n- ".join(failures))

    runtime_path = _run_root(workspace) / "runtime.json"
    runtime = read_json(runtime_path) if runtime_path.is_file() else {}
    methods_payload = {
        method_id: method.model_dump(mode="json")
        for method_id, method in sorted(methods.items())
    }
    configuration = {
        "task": task.model_dump(mode="json"),
        "sources": catalog.model_dump(mode="json"),
        "methods": methods_payload,
        "search_space": space.model_dump(mode="json"),
        "initial_recipe": recipe.model_dump(mode="json"),
        "runtime": runtime,
    }
    configuration_digest = digest(configuration)

    handoff_dir = workspace / "handoff"
    if handoff_dir.exists():
        shutil.rmtree(handoff_dir)
    (handoff_dir / "methods").mkdir(parents=True)
    write_json(handoff_dir / "task.json", configuration["task"])
    write_json(handoff_dir / "sources.json", configuration["sources"])
    write_json(handoff_dir / "search_space.json", configuration["search_space"])
    write_json(handoff_dir / "initial_recipe.json", configuration["initial_recipe"])
    write_json(handoff_dir / "runtime.json", runtime)
    for method_id, payload in methods_payload.items():
        write_json(handoff_dir / "methods" / f"{method_id}.json", payload)

    manifest = {
        "schema_version": 1,
        "status": "accepted",
        "created_at": utc_now(),
        "ownership": {
            "completed_by": "discovery_agent",
            "next_owner": "automatic_search",
        },
        "configuration_digest": configuration_digest,
        "source_audits": source_audits,
        "method_audits": method_audits,
        "initial_recipe_id": recipe.recipe_id,
    }
    write_json(handoff_dir / "manifest.json", manifest)
    write_json(
        handoff_dir / "state.json",
        {
            "schema_version": 1,
            "manifest_digest": digest(manifest),
            "configuration_digest": configuration_digest,
        },
    )

    state["status"] = "accepted"
    state["accepted_at"] = manifest["created_at"]
    state["handoff_manifest_digest"] = digest(manifest)
    write_json(workspace / "discovery_state.json", state)

    run_root = _run_root(workspace)
    stage_status_path = run_root / "discovery" / "status.json"
    if stage_status_path.is_file():
        stage_status = read_json(stage_status_path)
        stage_status["status"] = "accepted"
        stage_status["accepted_at"] = manifest["created_at"]
        write_json(stage_status_path, stage_status)
    run_status_path = run_root / "run.json"
    if run_status_path.is_file():
        run_status = read_json(run_status_path)
        run_status.setdefault("stages", {}).setdefault("discovery", {})["status"] = "accepted"
        run_status["stages"]["discovery"]["accepted_at"] = manifest["created_at"]
        write_json(run_status_path, run_status)
    return manifest


def verify_handoff(workspace: Path) -> dict[str, Any]:
    handoff_dir = workspace / "handoff"
    manifest = read_json(handoff_dir / "manifest.json")
    handoff_state = read_json(handoff_dir / "state.json")
    if manifest.get("status") != "accepted":
        raise RuntimeError("handoff status is not accepted")
    if digest(manifest) != handoff_state.get("manifest_digest"):
        raise RuntimeError("handoff manifest changed after acceptance")

    methods = {
        path.stem: read_json(path)
        for path in sorted((handoff_dir / "methods").glob("*.json"))
    }
    configuration = {
        "task": read_json(handoff_dir / "task.json"),
        "sources": read_json(handoff_dir / "sources.json"),
        "methods": methods,
        "search_space": read_json(handoff_dir / "search_space.json"),
        "initial_recipe": read_json(handoff_dir / "initial_recipe.json"),
        "runtime": read_json(handoff_dir / "runtime.json"),
    }
    if digest(configuration) != manifest.get("configuration_digest"):
        raise RuntimeError("handoff configuration changed after acceptance")
    return manifest
