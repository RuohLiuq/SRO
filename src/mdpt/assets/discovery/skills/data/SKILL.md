---
name: data
description: Research, compare, and adapt high-quality non-evaluation data sources for every configured benchmark and data role.
---

Research several candidate datasets for every benchmark in `task.json`. Begin with established sources commonly used for the same or closely related capability, then consider alternatives when they add useful coverage. Compare actual examples from the exact configuration, revision, and split; dataset names and schemas alone are not evidence of suitability.

Judge candidates by:

- capability and content relevance;
- similarity of task form, output contract, and difficulty;
- supervision quality and provenance;
- reproducibility and license;
- useful differences from other source choices.

Select several genuinely useful base-training alternatives for each benchmark. Quality and meaningful diversity matter more than filling the search space with near-duplicates.

Prepare sources for the applicable roles:

- Base training may use a broader range of relevant examples and should preserve the source's natural instruction format.
- Rewrite input must contain reliable existing supervision that the target benchmark adapter can verify.
- Correction diagnostics must be disjoint from training data and close enough to the benchmark to reveal relevant model failures.

Use deterministic preprocessing to select relevant examples and faithfully convert existing fields into the source contract. A broad dataset may be filtered when it contains examples outside the target capability; if the relevant subset cannot be selected reliably, choose another dataset. Inspect normalized output after conversion.

Verifier-backed roles require genuine source supervision such as an explicit answer, label, structured tool call, or executable tests. Never manufacture missing oracle information during adaptation.

For every source, record its stable ID, target benchmark, dataset identity, revision, split, license, field mapping, role, benchmark adapter, and deterministic preprocessing. Use public dataset identifiers or workspace-relative paths; do not persist private host or cache paths. Evaluation data and its derivatives are excluded from every role.
