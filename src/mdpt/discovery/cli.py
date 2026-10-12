from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

from mdpt.discovery.handoff import create_handoff, verify_handoff
from mdpt.discovery.methods.audit import audit_method, method_contract_digest
from mdpt.discovery.schema import (
    DataMethodSpec,
    Recipe,
    SearchSpace,
    SourceCatalog,
)
from mdpt.discovery.state import (
    artifact_paths,
    assert_initialized_configuration_current,
    initialize_state,
    load_initialized_artifacts,
    resolve_workspace,
)
from mdpt.io import read_json, write_json


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@contextmanager
def _inside(workspace: Path):
    previous = Path.cwd()
    os.chdir(workspace)
    try:
        yield
    finally:
        os.chdir(previous)


def cmd_schema(args: argparse.Namespace) -> None:
    models = {
        "sources": SourceCatalog,
        "method": DataMethodSpec,
        "search-space": SearchSpace,
        "recipe": Recipe,
    }
    _print(models[args.name].model_json_schema())


def cmd_init(args: argparse.Namespace) -> None:
    workspace = resolve_workspace(args.workspace)
    sources_path, space_path, methods_dir = artifact_paths(
        workspace,
        sources=args.sources,
        space=args.space,
        methods_dir=args.methods_dir,
    )
    state = initialize_state(
        workspace,
        sources_path=sources_path,
        space_path=space_path,
        methods_dir=methods_dir,
    )
    (workspace / "audits" / "sources").mkdir(parents=True, exist_ok=True)
    (workspace / "audits" / "methods").mkdir(parents=True, exist_ok=True)
    _print({"ok": True, "status": state["status"], "workspace": str(workspace)})


def cmd_validate(args: argparse.Namespace) -> None:
    workspace = resolve_workspace(args.workspace)
    state, task, catalog, methods, space = load_initialized_artifacts(workspace)
    assert_initialized_configuration_current(state, task, catalog, methods, space)
    _print(
        {
            "ok": True,
            "status": state["status"],
            "domains": len(task.domains),
            "benchmarks": len(task.benchmark_ids),
            "sources": len(catalog.sources),
            "methods": len(methods),
        }
    )


def cmd_audit_method(args: argparse.Namespace) -> None:
    workspace = resolve_workspace(args.workspace)
    state, task, catalog, methods, space = load_initialized_artifacts(workspace)
    assert_initialized_configuration_current(state, task, catalog, methods, space)

    method_path = Path(args.method)
    method_path = method_path.resolve() if method_path.is_absolute() else (workspace / method_path).resolve()
    try:
        method_path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("method file must stay inside Discovery workspace") from exc
    method = DataMethodSpec.model_validate(read_json(method_path))
    registered = methods.get(method.method_id)
    if registered is None:
        raise ValueError(f"method is not registered in methods directory: {method.method_id!r}")
    if registered != method:
        raise ValueError(
            f"method file differs from registered method {method.method_id!r}; run discovery init again"
        )

    source_id = (
        method.input_source_id if method.operation == "rewrite" else method.diagnostic_source_id
    )
    source = catalog.by_id()[str(source_id)]
    with _inside(workspace):
        report = audit_method(method, source, {}, sample_size=int(args.samples))
    payload = report.as_dict()
    audit_dir = workspace / "audits" / "methods" / method.method_id
    digest = method_contract_digest(method)
    write_json(audit_dir / f"{digest}.json", payload)
    write_json(audit_dir / "latest.json", payload)
    _print(payload)
    if not report.ok:
        raise RuntimeError(
            f"method audit rejected {method.method_id!r}: accepted "
            f"{report.accepted}/{report.attempted}"
        )


def cmd_handoff(args: argparse.Namespace) -> None:
    workspace = resolve_workspace(args.workspace)
    recipe_path = Path(args.initial_recipe)
    recipe_path = recipe_path.resolve() if recipe_path.is_absolute() else (workspace / recipe_path).resolve()
    try:
        recipe_path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("initial Recipe must stay inside Discovery workspace") from exc
    with _inside(workspace):
        manifest = create_handoff(
            workspace,
            recipe_path,
            source_sample_size=int(args.source_samples),
            execute_code=not args.skip_code_execution,
        )
    _print({"ok": True, "handoff": manifest})


def cmd_verify_handoff(args: argparse.Namespace) -> None:
    workspace = resolve_workspace(args.workspace)
    _print({"ok": True, "handoff": verify_handoff(workspace)})


def add_discovery_parser(commands: argparse._SubParsersAction) -> None:
    discovery = commands.add_parser("discovery", help="Build and validate Discovery artifacts")
    actions = discovery.add_subparsers(dest="discovery_command", required=True)

    schema = actions.add_parser("schema", help="Print an Agent-authored artifact JSON Schema")
    schema.add_argument("--name", choices=("sources", "method", "search-space", "recipe"), required=True)
    schema.set_defaults(handler=cmd_schema)

    init = actions.add_parser("init", help="Validate Agent-authored artifacts and initialize state")
    init.add_argument("--workspace", default=".")
    init.add_argument("--sources", default="sources.json")
    init.add_argument("--space", default="search_space.json")
    init.add_argument("--methods-dir", default="methods")
    init.set_defaults(handler=cmd_init)

    validate = actions.add_parser("validate", help="Validate the initialized Discovery state")
    validate.add_argument("--workspace", default=".")
    validate.set_defaults(handler=cmd_validate)

    audit = actions.add_parser("audit-method", help="Run a real augmentation method audit")
    audit.add_argument("--workspace", default=".")
    audit.add_argument("--method", required=True)
    audit.add_argument("--samples", type=int, default=5)
    audit.set_defaults(handler=cmd_audit_method)

    handoff = actions.add_parser("handoff", help="Audit sources and submit the handoff")
    handoff.add_argument("--workspace", default=".")
    handoff.add_argument("--initial-recipe", default="initial_recipe.json")
    handoff.add_argument("--source-samples", type=int, default=16)
    handoff.add_argument("--skip-code-execution", action="store_true")
    handoff.set_defaults(handler=cmd_handoff)

    verify = actions.add_parser("verify-handoff", help="Verify an accepted handoff")
    verify.add_argument("--workspace", default=".")
    verify.set_defaults(handler=cmd_verify_handoff)
