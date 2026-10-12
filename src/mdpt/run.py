from __future__ import annotations

from pathlib import Path
from typing import Literal

from mdpt.agent import run_discovery_agent
from mdpt.config import load_environment
from mdpt.io import load_task
from mdpt.search.run import run_evaluate, run_search
from mdpt.workspace import prepare_discovery_workspace


Stage = Literal["discovery", "search", "evaluate", "full"]


def run(
    *,
    stage: Stage,
    task_path: Path | None,
    env_path: Path,
    work_dir: Path,
    search_config_path: Path | None = None,
    max_hours: float | None = None,
    max_rounds: int | None = None,
) -> dict[str, object]:
    """Run the selected MDPT stage through the single public orchestrator."""
    environment = load_environment(env_path.expanduser().resolve())
    resolved_work_dir = work_dir.expanduser().resolve()

    if stage == "discovery":
        if task_path is None:
            raise ValueError("Discovery requires --task")
        task = load_task(task_path.expanduser().resolve())
        prepare_discovery_workspace(
            task=task,
            environment=environment,
            work_dir=resolved_work_dir,
        )
        return run_discovery_agent(
            task=task,
            environment=environment,
            work_dir=resolved_work_dir,
        )

    if stage == "search":
        if search_config_path is None or max_hours is None:
            raise ValueError("Search requires --search-config and --max-hours")
        return run_search(
            work_dir=resolved_work_dir,
            environment=environment,
            search_config_path=search_config_path,
            max_hours=max_hours,
            max_rounds=max_rounds,
        )

    if stage == "evaluate":
        return run_evaluate(work_dir=resolved_work_dir, environment=environment)

    if task_path is None or search_config_path is None or max_hours is None:
        raise ValueError("Full mode requires --task, --search-config, and --max-hours")
    task = load_task(task_path.expanduser().resolve())
    prepare_discovery_workspace(
        task=task,
        environment=environment,
        work_dir=resolved_work_dir,
    )
    discovery_result = run_discovery_agent(
        task=task,
        environment=environment,
        work_dir=resolved_work_dir,
    )
    search_result = run_search(
        work_dir=resolved_work_dir,
        environment=environment,
        search_config_path=search_config_path,
        max_hours=max_hours,
        max_rounds=max_rounds,
    )
    evaluate_result = run_evaluate(
        work_dir=resolved_work_dir,
        environment=environment,
    )
    return {
        "stage": "full",
        "status": "completed",
        "discovery": discovery_result,
        "search": search_result,
        "evaluate": evaluate_result,
    }
