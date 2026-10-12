from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdpt.discovery.cli import add_discovery_parser
from mdpt.run import run


def _run(args: argparse.Namespace) -> None:
    stage = next(
        name
        for name in ("discovery", "search", "evaluate", "full")
        if getattr(args, name)
    )
    if stage in {"discovery", "full"} and not args.task:
        raise ValueError("--task is required for Discovery and Full modes")
    if stage in {"search", "full"} and (args.max_hours is None or args.max_hours <= 0):
        raise ValueError("a positive --max-hours is required for Search and Full modes")
    if stage in {"search", "full"} and not args.search_config:
        raise ValueError("--search-config is required for Search and Full modes")
    result = run(
        stage=stage,
        task_path=Path(args.task) if args.task else None,
        env_path=Path(args.env),
        work_dir=Path(args.work_dir),
        search_config_path=Path(args.search_config) if args.search_config else None,
        max_hours=args.max_hours,
        max_rounds=args.max_rounds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdpt")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run an MDPT stage")
    run.add_argument("--task", help="Task JSON; required only for Discovery and Full modes")
    run.add_argument("--env", required=True, help="Path to the local .env file")
    run.add_argument("--work-dir", required=True, help="Directory for durable run state")
    run.add_argument(
        "--search-config",
        help="Search parameter JSON; required for Search and Full modes",
    )
    run.add_argument("--max-hours", type=float, help="Total Discovery + Search wall-clock budget")
    run.add_argument("--max-rounds", type=int, help="Optional override for Search rounds")
    stages = run.add_mutually_exclusive_group(required=True)
    stages.add_argument("--discovery", action="store_true")
    stages.add_argument("--search", action="store_true")
    stages.add_argument("--evaluate", action="store_true")
    stages.add_argument("--full", action="store_true")
    run.set_defaults(handler=_run)
    add_discovery_parser(commands)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
