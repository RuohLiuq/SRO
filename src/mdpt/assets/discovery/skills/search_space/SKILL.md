---
name: search-space
description: Assemble audited sources, augmentation methods, and mixtures into a finite SFT data search space.
---

Construct a finite SFT data search space that offers meaningful alternatives for every benchmark in `task.json`.

For each benchmark, include:

- several accurate and complementary audited base-source choices;
- accepted augmentation methods supported by its benchmark adapter;
- positive benchmark-mixture choices within its domain.

Quality and meaningful diversity matter more than nominal space size. Do not include near-duplicate choices or cosmetic variations of one Recipe.

For each domain, define valid mixtures over its configured benchmarks. If the task contains multiple domains, also define reasonable cross-domain allocations while keeping every configured objective covered. A single-domain task does not need artificial cross-domain choices. Always include an equal benchmark mixture for every domain and an equal cross-domain mixture; Search uses those declared choices for its framework-owned seed.

Use declared source and method IDs directly in the search-space artifact. Every reference must resolve to an audited declaration. The resulting space must contain multiple structurally different executable Recipes.

Create one valid initial Recipe drawn from the declared choices. It only demonstrates that the space is executable; Search does not execute it as the first trial. Search constructs its own first trial from all accepted base sources without Rewrite or Correction.

Discovery controls the SFT data search space only. The framework owns training execution, optimization, later RL decisions, and resource-safe runtime profiles.
