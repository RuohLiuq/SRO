# Agent-designed augmentation method contract

## Rewrite

A Rewrite method declares:

```json
{
  "method_id": "<stable-versioned-id>",
  "benchmark_id": "<configured benchmark>",
  "operation": "rewrite",
  "input_source_id": "<audited training source>",
  "evolution_axes": ["<Agent-derived benchmark dimension>"],
  "rewrite_prompt": "<Agent-authored prompt creating a new benchmark input>",
  "answer_prompt": "<Agent-authored prompt independently creating complete supervision>",
  "description": "<short search-facing summary>"
}
```

The Rewrite worker creates only the new benchmark input. A separate Answer call receives that new input, never the old target, and produces complete assistant supervision. Never copy or retain the source target. All reasoning, entities, units, constraints, code, tests, tool calls, and safety guidance must match the new input.

The target benchmark context defines exact stage fields and output conventions.

## Correction

A Correction method declares:

```json
{
  "method_id": "<stable-versioned-id>",
  "benchmark_id": "<configured benchmark>",
  "operation": "correction",
  "diagnostic_source_id": "<audited disjoint diagnostic source>",
  "correction_prompt": "<Agent-authored prompt using task, failed attempt, and verifier feedback>",
  "description": "<short search-facing summary>"
}
```

During Search, the framework supplies real failed attempts and verifier feedback. The Correction worker must produce complete training supervision, not only a final label unless that is the benchmark's complete assistant contract.

## Adapter feedback loop

Audit a method before handoff. The audit runs real data-worker calls, checks exact stage fields, canonicalizes the result, and invokes the same benchmark adapter and verifier used during Search materialization.

Inspect every accepted row as a self-contained input-supervision pair. A matching final label alone is insufficient when reasoning, names, units, constraints, choices, code, tool arguments, or safety guidance disagree. Revise prompts from row-level feedback and audit the complete revised contract again.

Deterministic checks cannot prove all generated supervision is semantically correct. Do not optimize only for audit pass rate or simplify the task distribution in response to stochastic infrastructure failures.
