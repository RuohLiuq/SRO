from __future__ import annotations

from typing import Any

from mdpt.schema import TaskConfig


def _mean_expression(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return f"({' + '.join(names)}) / {len(names)}"


def build_discovery_prompt(
    task: TaskConfig,
    benchmark_metadata: dict[str, dict[str, Any]],
) -> str:
    domain_sections: list[str] = []
    domain_scores: list[str] = []
    primary_scores: list[str] = []

    for index, domain in enumerate(task.domains, start=1):
        benchmark_lines: list[str] = []
        domain_primary_scores: list[str] = []
        for benchmark in domain.benchmarks:
            metadata = benchmark_metadata[benchmark.benchmark_id]
            primary_score = str(metadata["primary_score"])
            primary_scores.append(primary_score)
            domain_primary_scores.append(primary_score)
            benchmark_lines.append(
                f"   - `{benchmark.benchmark_id}` — {metadata['display_name']}"
            )
        domain_sections.append(
            f"{index}. `{domain.domain_id}`\n" + "\n".join(benchmark_lines)
        )
        domain_scores.append(
            f"- `{domain.domain_id}_score = {_mean_expression(domain_primary_scores)}`"
        )

    primary_score_lines = "\n".join(f"- `{name}`" for name in primary_scores)
    domains = "\n".join(domain_sections)
    diagnostics = "\n".join(domain_scores)
    final_score = _mean_expression(primary_scores)

    return f"""# MDPT Discovery Task

Your objective is to design, validate, and submit a high-quality executable data search space that enables the later automatic Search stage to improve the specified base model broadly across all configured benchmarks.

You are responsible only for Discovery. Do not train the target model, run Search, perform RL, evaluate final checkpoints, or create `final_model`.

## Task

- Base model: `{task.base_model}`
- Discovery time limit: {task.budget.discovery_hours:g} hours
- Maximum training examples per later Search trial: {task.budget.max_train_samples_per_trial:,}

## Domains and benchmarks

{domains}

The task in `task.json` is authoritative. Do not add, remove, rename, or regroup its domains or benchmarks. The search space should support broad improvement across all configured benchmarks rather than specialize in one benchmark or domain.

## Evaluation interfaces and scoring

The later Evaluate stage reports one primary score for every configured benchmark:

{primary_score_lines}

The final task score is the arithmetic mean of all configured benchmark primary scores:

`mean_score = {final_score}`

Diagnostic domain scores are:

{diagnostics}

Domain scores are diagnostic summaries and do not add another weighting level.

Read the read-only public context under `benchmarks/<benchmark_id>/` for every configured benchmark. Inspect its evaluator interface, scorer, templates, task format, and representative non-evaluation documentation before selecting data. The benchmarks may use different model-facing input and output contracts; preserve these differences when designing sources and augmentation methods.

Do not modify the packaged benchmark context, evaluators, scorers, or templates. Evaluation questions, answers, solutions, tool-call targets, grader rubrics, canaries, and transformations of them are evaluation-only and must not enter training sources, Rewrite inputs, diagnostic sources, generated-data prompts, examples, or audit fixtures.

## Discovery autonomy and environment

- Operate autonomously; no user interaction will be available.
- Public internet access may be used to research documentation and non-evaluation datasets.
- Work only inside the current Discovery workspace and explicitly assigned external caches.
- Use only the CUDA devices assigned through `CUDA_VISIBLE_DEVICES`. Do not modify that variable or access other GPUs.
- Do not modify host services or system configuration.
- Do not stop or interfere with unrelated processes. Manage only processes started by this run.
- Do not inspect sibling runs, previous experiments, historical handoffs, checkpoints, controller logs, recipes, or generated training data.
- Use only credentials explicitly assigned to Discovery for their configured purpose. Never expose credentials, private endpoints, host paths, or machine information in Discovery artifacts.
- Keep dataset identities, revisions, splits, licenses, field mappings, deterministic preprocessing, prompts, and audit results reproducible.

## Discovery deliverable

For every configured benchmark:

- research suitable non-evaluation data sources;
- prepare useful and meaningfully different base-training alternatives;
- design the augmentation operations supported by its benchmark adapter;
- validate sources and methods;
- include substantive alternatives in a finite SFT data search space.

Produce:

- a source catalog;
- augmentation method declarations;
- a finite SFT data search space;
- one valid initial Recipe;
- required source and method audit reports;
- an accepted handoff.

The framework owns model execution, optimization, Search, RL, checkpoint selection, and final evaluation.

Read `AGENTS.md`, follow the Skills in the required order, and stop immediately after handoff acceptance.
"""
