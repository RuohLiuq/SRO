from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from mdpt.config import EnvironmentConfig
from mdpt.discovery.handoff import verify_handoff
from mdpt.io import read_json, write_json
from mdpt.schema import TaskConfig


def _model_parts(model: str) -> tuple[str, str, str]:
    if "/" in model:
        provider, model_id = model.split("/", 1)
        return provider, model_id, model
    return "mdpt", model, f"mdpt/{model}"


def _write_opencode_config(workspace: Path, environment: EnvironmentConfig) -> Path:
    provider, model_id, _ = _model_parts(environment.agent_model)
    path = workspace / "opencode.json"
    write_json(
        path,
        {
            "$schema": "https://opencode.ai/config.json",
            "permission": "allow",
            "provider": {
                provider: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "MDPT Agent Provider",
                    "options": {
                        "baseURL": "{env:MDPT_AGENT_BASE_URL}",
                        "apiKey": "{env:MDPT_AGENT_API_KEY}",
                    },
                    "models": {model_id: {"name": "MDPT Discovery Model"}},
                }
            },
        },
    )
    return path


def _pump(stream: TextIO, log: TextIO, console: TextIO) -> None:
    for line in stream:
        log.write(line)
        log.flush()
        console.write(line)
        console.flush()


def _update_status(work_dir: Path, values: dict[str, Any]) -> None:
    status_path = work_dir / "discovery" / "status.json"
    status = read_json(status_path)
    status.update(values)
    write_json(status_path, status)

    run_path = work_dir / "run.json"
    run_status = read_json(run_path)
    run_status.setdefault("stages", {}).setdefault("discovery", {}).update(values)
    write_json(run_path, run_status)


def run_discovery_agent(
    *,
    task: TaskConfig,
    environment: EnvironmentConfig,
    work_dir: Path,
) -> dict[str, Any]:
    discovery_dir = work_dir / "discovery"
    workspace = discovery_dir / "workspace"
    prompt_path = discovery_dir / "prompt.md"
    if not workspace.is_dir() or not prompt_path.is_file():
        raise RuntimeError("Discovery workspace is not prepared")

    prompt = prompt_path.read_text(encoding="utf-8")
    command = shlex.split(environment.agent_command)
    if not command:
        raise ValueError("MDPT_AGENT_COMMAND is empty")
    executable = Path(command[0]).expanduser()
    if executable.is_absolute() and not executable.is_file():
        raise FileNotFoundError(f"Agent command does not exist: {executable}")

    _, _, model = _model_parts(environment.agent_model)
    config_path = _write_opencode_config(workspace, environment)
    logs_dir = discovery_dir / "logs"
    runtime_dir = discovery_dir / "agent_runtime"
    logs_dir.mkdir(exist_ok=True)
    runtime_dir.mkdir(exist_ok=True)

    child_environment = environment.child_environment()
    child_environment.update(
        {
            "XDG_DATA_HOME": str(runtime_dir / "data"),
            "XDG_CONFIG_HOME": str(runtime_dir / "config"),
            "XDG_CACHE_HOME": str(runtime_dir / "cache"),
            "MDPT_DISCOVERY_WORKSPACE": str(workspace),
        }
    )
    argv = command + [
        "run",
        "--model",
        model,
        "--format",
        "json",
        "--auto",
        "--dir",
        str(workspace),
        prompt,
    ]

    started_at = time.time()
    deadline_seconds = max(1, int(task.budget.discovery_hours * 3600))
    _update_status(
        work_dir,
        {
            "status": "running",
            "agent_started": True,
            "agent_started_at": started_at,
            "agent_timeout_seconds": deadline_seconds,
        },
    )

    stdout_path = logs_dir / "agent.jsonl"
    stderr_path = logs_dir / "agent.stderr.log"
    timed_out = False
    return_code: int | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_log, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_log:
            process = subprocess.Popen(
                argv,
                cwd=workspace,
                env=child_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("Agent process pipes were not created")
            stdout_thread = threading.Thread(
                target=_pump,
                args=(process.stdout, stdout_log, sys.stdout),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_pump,
                args=(process.stderr, stderr_log, sys.stderr),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            try:
                return_code = process.wait(timeout=deadline_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGINT)
                try:
                    return_code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    return_code = process.wait()
            finally:
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
    except Exception:
        _update_status(
            work_dir,
            {
                "status": "failed",
                "agent_completed_at": time.time(),
                "agent_return_code": return_code,
            },
        )
        raise
    finally:
        config_path.unlink(missing_ok=True)

    completed_at = time.time()
    manifest_path = workspace / "handoff" / "manifest.json"
    if return_code == 0 and manifest_path.is_file():
        manifest = verify_handoff(workspace)
        status = "accepted"
    else:
        manifest = None
        status = "timed_out" if timed_out else "failed"

    _update_status(
        work_dir,
        {
            "status": status,
            "agent_completed_at": completed_at,
            "agent_elapsed_seconds": completed_at - started_at,
            "agent_return_code": return_code,
            "agent_timed_out": timed_out,
            "agent_stdout": "discovery/logs/agent.jsonl",
            "agent_stderr": "discovery/logs/agent.stderr.log",
        },
    )
    if manifest is None:
        reason = "timed out" if timed_out else f"exited with code {return_code}"
        raise RuntimeError(
            f"Discovery Agent {reason} without an accepted handoff; inspect {logs_dir}"
        )
    return {
        "stage": "discovery",
        "status": "accepted",
        "work_dir": str(work_dir),
        "handoff": str(manifest_path),
        "elapsed_seconds": completed_at - started_at,
    }
