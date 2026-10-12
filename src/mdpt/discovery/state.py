from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mdpt.io import read_json, write_json
from mdpt.schema import TaskConfig
from mdpt.discovery.schema import DataMethodSpec, SearchSpace, SourceCatalog
from mdpt.discovery.validation import digest, load_methods, validate_configuration


STATE_FILE = "discovery_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_workspace(path: str | Path | None = None) -> Path:
    workspace = Path(path or ".").expanduser().resolve()
    if not (workspace / "task.json").is_file():
        raise FileNotFoundError(f"Discovery workspace has no task.json: {workspace}")
    return workspace


def artifact_paths(
    workspace: Path,
    *,
    sources: str | Path = "sources.json",
    space: str | Path = "search_space.json",
    methods_dir: str | Path = "methods",
) -> tuple[Path, Path, Path]:
    def inside(value: str | Path) -> Path:
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (workspace / path).resolve()
        try:
            path.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"Discovery artifact must stay inside workspace: {path}") from exc
        return path

    return inside(sources), inside(space), inside(methods_dir)


def load_artifacts(
    workspace: Path,
    *,
    sources_path: Path,
    space_path: Path,
    methods_dir: Path,
) -> tuple[TaskConfig, SourceCatalog, dict[str, DataMethodSpec], SearchSpace]:
    task = TaskConfig.model_validate(read_json(workspace / "task.json"))
    run_task_path = workspace.parent.parent / "task.json"
    if run_task_path.is_file():
        run_task = TaskConfig.model_validate(read_json(run_task_path))
        if task != run_task:
            raise RuntimeError("workspace task.json differs from the immutable run task")
    catalog = SourceCatalog.model_validate(read_json(sources_path))
    methods = load_methods(methods_dir)
    space = SearchSpace.model_validate(read_json(space_path))
    validate_configuration(task, catalog, methods, space, workspace=workspace)
    return task, catalog, methods, space


def configuration_digests(
    task: TaskConfig,
    catalog: SourceCatalog,
    methods: dict[str, DataMethodSpec],
    space: SearchSpace,
) -> dict[str, Any]:
    return {
        "task": digest(task.model_dump(mode="json")),
        "sources": digest(catalog.model_dump(mode="json")),
        "search_space": digest(space.model_dump(mode="json")),
        "methods": {
            method_id: digest(method.model_dump(mode="json"))
            for method_id, method in sorted(methods.items())
        },
    }


def initialize_state(
    workspace: Path,
    *,
    sources_path: Path,
    space_path: Path,
    methods_dir: Path,
) -> dict[str, Any]:
    task, catalog, methods, space = load_artifacts(
        workspace,
        sources_path=sources_path,
        space_path=space_path,
        methods_dir=methods_dir,
    )
    state = {
        "schema_version": 1,
        "status": "initialized",
        "initialized_at": utc_now(),
        "artifacts": {
            "sources": str(sources_path.relative_to(workspace)),
            "search_space": str(space_path.relative_to(workspace)),
            "methods_dir": str(methods_dir.relative_to(workspace)),
        },
        "digests": configuration_digests(task, catalog, methods, space),
    }
    write_json(workspace / STATE_FILE, state)
    return state


def load_state(workspace: Path) -> dict[str, Any]:
    path = workspace / STATE_FILE
    if not path.is_file():
        raise RuntimeError("Discovery is not initialized; run `mdpt discovery init` first")
    return read_json(path)


def load_initialized_artifacts(
    workspace: Path,
) -> tuple[dict[str, Any], TaskConfig, SourceCatalog, dict[str, DataMethodSpec], SearchSpace]:
    state = load_state(workspace)
    artifacts = state["artifacts"]
    sources_path, space_path, methods_dir = artifact_paths(
        workspace,
        sources=artifacts["sources"],
        space=artifacts["search_space"],
        methods_dir=artifacts["methods_dir"],
    )
    task, catalog, methods, space = load_artifacts(
        workspace,
        sources_path=sources_path,
        space_path=space_path,
        methods_dir=methods_dir,
    )
    return state, task, catalog, methods, space


def assert_initialized_configuration_current(
    state: dict[str, Any],
    task: TaskConfig,
    catalog: SourceCatalog,
    methods: dict[str, DataMethodSpec],
    space: SearchSpace,
) -> None:
    current = configuration_digests(task, catalog, methods, space)
    if current != state.get("digests"):
        raise RuntimeError(
            "Discovery artifacts changed after initialization; run `mdpt discovery init` again, "
            "then repeat audits for any changed method"
        )
