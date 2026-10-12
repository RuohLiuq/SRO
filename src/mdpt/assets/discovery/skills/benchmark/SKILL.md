---
name: benchmark
description: Understand every configured benchmark's capability, task contract, difficulty, scorer, and evaluation-data boundary before selecting data.
---

Begin by studying every benchmark listed in `task.json`. Use its packaged read-only evaluator interface, scorer, templates, documentation, task format, and representative non-evaluation examples to understand what successful training data should teach.

For each benchmark, create a concise working description of:

- the capability and knowledge being measured;
- the model-facing input and expected output;
- how correctness is determined;
- the approximate difficulty and content distribution;
- the properties valid training examples must preserve;
- the kinds of non-evaluation examples likely to transfer.

Use this description when comparing data sources and designing augmentation. Similarity includes capability, task form, output contract, verifier behavior, and difficulty rather than storage schema alone.

Target evaluation questions, answers, solutions, tool-call targets, grader rubrics, canaries, and transformations of them are reserved for evaluation. Keep them out of notes, training sources, Rewrite inputs, diagnostics, generated-data prompts, examples, and audit fixtures.
