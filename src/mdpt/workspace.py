from __future__ import annotations

import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mdpt.benchmarks import load_benchmark_metadata
from mdpt.config import EnvironmentConfig
from mdpt.io import content_digest, write_json
from mdpt.prompt import build_discovery_prompt
from mdpt.schema import TaskConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _asset_root() -> Path:
    return Path(__file__).resolve().parent / "assets"


def _validate_assets(task: TaskConfig) -> None:
    assets = _asset_root()
    required = [
        assets / "discovery" / "AGENTS.md",
        assets / "discovery" / "skills",
        assets / "templates",
    ]
    required.extend(assets / "benchmarks" / benchmark_id for benchmark_id in task.benchmark_ids)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing packaged Discovery assets:\n- " + "\n- ".join(missing))


def _make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def prepare_discovery_workspace(
    *,
    task: TaskConfig,
    environment: EnvironmentConfig,
    work_dir: Path,
) -> dict[str, object]:
    work_dir = work_dir.resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"work directory is not empty: {work_dir}")

    _validate_assets(task)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    run_id = uuid.uuid4().hex

    discovery_dir = work_dir / "discovery"
    agent_workspace = discovery_dir / "workspace"
    agent_workspace.mkdir(parents=True, exist_ok=True)
    (agent_workspace / "sources").mkdir()
    (agent_workspace / "methods").mkdir()

    task_payload = task.model_dump(mode="json")
    run_task = work_dir / "task.json"
    write_json(run_task, task_payload)
    run_task.chmod(0o444)
    workspace_task = agent_workspace / "task.json"
    write_json(workspace_task, task_payload)
    workspace_task.chmod(0o444)

    assets = _asset_root()
    shutil.copy2(assets / "discovery" / "AGENTS.md", agent_workspace / "AGENTS.md")
    shutil.copytree(assets / "discovery" / "skills", agent_workspace / "skills")
    opencode_dir = agent_workspace / ".opencode"
    opencode_dir.mkdir()
    (opencode_dir / "skills").symlink_to(Path("../skills"), target_is_directory=True)

    benchmark_root = agent_workspace / "benchmarks"
    benchmark_root.mkdir()
    for benchmark_id in task.benchmark_ids:
        shutil.copytree(
            assets / "benchmarks" / benchmark_id,
            benchmark_root / benchmark_id,
        )
    templates_root = agent_workspace / "templates"
    shutil.copytree(assets / "templates", templates_root)
    _make_read_only(benchmark_root)
    _make_read_only(templates_root)

    benchmark_metadata = load_benchmark_metadata(task)
    prompt = build_discovery_prompt(task, benchmark_metadata)
    (discovery_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    runtime_summary = environment.redacted_summary()
    write_json(work_dir / "runtime.json", runtime_summary)

    completed_at = _utc_now()
    elapsed_seconds = time.monotonic() - started_monotonic
    stage_status = {
        "stage": "discovery",
        "status": "prepared",
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "budget_hours": task.budget.discovery_hours,
        "agent_started": False,
    }
    write_json(discovery_dir / "status.json", stage_status)

    run_status = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": started_at,
        "task_digest": content_digest(task_payload),
        "active_stage": "discovery",
        "stages": {"discovery": stage_status},
    }
    write_json(work_dir / "run.json", run_status)
    return {
        "run_id": run_id,
        "stage": "discovery",
        "status": "prepared",
        "work_dir": str(work_dir),
        "prompt": str(discovery_dir / "prompt.md"),
    }
