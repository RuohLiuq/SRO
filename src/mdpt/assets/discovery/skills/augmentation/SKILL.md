---
name: augmentation
description: Derive benchmark-aware Rewrite and Correction methods and improve them through adapter feedback.
---

Study each configured benchmark's capability, model-facing contract, scorer, difficulty, representative non-evaluation examples, and adapter capabilities. Design only augmentation operations explicitly supported by its adapter.

Read [references/evol_instruct.md](references/evol_instruct.md) when exploring evolution dimensions. It describes depth and breadth as design abstractions, not fixed transformations.

Read [references/method_contract.md](references/method_contract.md) when implementing methods:

- Rewrite evolves an audited source task into a new task within the target benchmark's capability neighborhood.
- Correction turns a model failure and verifier feedback into corrected training supervision.

Author methods that communicate benchmark-derived transformations clearly to the data worker. Methods must offer substantive capability coverage rather than surface paraphrases.

## Rewrite

Every Rewrite uses two stages:

1. `rewrite_prompt` creates a new benchmark input by evolving the provided source task.
2. `answer_prompt` receives only the new input and independently creates every field of supervision required by the benchmark contract.

The Answer stage must not see or copy the old target. The new input, reasoning, entities, units, constraints, final answer, code, tests, tool arguments, and safety guidance must agree.

## Correction

Correction uses a separate audited diagnostic source. During Search, the framework supplies the original task, a failed model attempt, and verifier feedback. `correction_prompt` must produce complete corrected training supervision that can be checked against the diagnostic source's existing oracle or tests.

## Benchmark-specific contracts

Bind every method to one explicit benchmark. Read that benchmark's packaged context for its required stage fields, final-answer syntax, structured tool schema, executable-test contract, conversation format, and verifier behavior. Do not propagate a workaround from one benchmark to another.

## Audit loop

After registering the final method files and initializing Discovery state, probe each method through the framework generator and target benchmark adapter:

```bash
mdpt discovery audit-method \
  --method methods/<method_id>.json \
  --samples 5
```

The audit checks exact stage fields, canonicalizes the generated example, and invokes the benchmark verifier.

Use row-level feedback to repair only the failed contract dimension, then audit the complete revised method again. Serialization, parser, token-limit, infrastructure, or stochastic generation failures are not evidence that the target task should be simplified.

Passing deterministic checks is necessary but not sufficient. Inspect accepted examples for semantic correctness, benchmark-aligned difficulty, complete supervision, reasoning quality where applicable, and diversity across the batch. Rewrite must remain conditioned on its source task rather than collapse into repeated free-form templates.
