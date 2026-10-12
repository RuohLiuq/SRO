---
name: handoff
description: Validate and submit Discovery artifacts to the automatic post-training system.
---

Treat the workspace `task.json` as immutable and authoritative. Do not recreate or modify its base model, domains, benchmarks, or budget. Runtime credentials are owned by the framework and must not appear in Discovery artifacts.

Prepare these Agent-authored files in the current workspace:

```text
sources.json
methods/<method_id>.json
search_space.json
initial_recipe.json
```

Inspect the exact public schemas before writing them:

```bash
mdpt discovery schema --name sources
mdpt discovery schema --name method
mdpt discovery schema --name search-space
mdpt discovery schema --name recipe
```

Initialize and validate the Agent-authored artifacts:

```bash
mdpt discovery init \
  --sources sources.json \
  --space search_space.json \
  --methods-dir methods

mdpt discovery validate
```

Complete method edits before the final initialization so every audit digest corresponds exactly to the registered method contract. Audit every method:

```bash
mdpt discovery audit-method \
  --method methods/<method_id>.json \
  --samples 5
```

Use source, row, and method feedback to repair only the corresponding artifact. After changing any source, method, or search-space file, run `mdpt discovery init` again and repeat the affected method audits.

Submit the complete Discovery output:

```bash
mdpt discovery handoff \
  --initial-recipe initial_recipe.json
```

Handoff validates the initial Recipe as an executability witness, runs Source Audit, checks every current method digest against a passing Method Audit, and writes the accepted handoff under `handoff/`. The automatic Search does not execute this witness as its first trial; it constructs a framework-owned seed from all accepted base sources without Rewrite or Correction.

Discovery is complete only when `handoff/manifest.json` has status `accepted`. Verify it with:

```bash
mdpt discovery verify-handoff
```

Stop immediately after acceptance.
