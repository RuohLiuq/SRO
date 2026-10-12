# Evol-Instruct: design idea and illustrative cases

Evol-Instruct treats data creation as evolution of an existing instruction rather than simple paraphrase. Its useful abstraction is to ask how an instruction can move through a space of capability demands while remaining a coherent task. Two broad directions are:

- Depth: keep the task family but demand a more involved capability, such as adding a meaningful constraint, making an implicit concept concrete, increasing reasoning depth, or requiring several dependent steps.
- Breadth: create a different task in the same capability neighborhood, at comparable or deliberately chosen difficulty, so the data covers more than surface variants of one seed.

These are meta-level questions, not a fixed list of transformations. For a new benchmark, infer the relevant evolution dimensions from its evaluator and examples. “Add a constraint” is useful only when the constraint changes the capability in a valid, scoreable way; adding names, numbers, or prose mechanically is not evolution by itself.

Illustrative case: a planning benchmark may reveal that the scorer values satisfying interacting temporal and resource conditions. The Agent could therefore consider evolution along constraint interaction or dependency depth. It should not copy a predetermined condition such as “add exactly two deadlines”; the actual condition should follow from the benchmark task.

Illustrative case: a programming benchmark may expose a callable interface and behavioral tests. The Agent might infer evolution dimensions such as edge-case coverage, composition of requirements, or a different problem in the same algorithmic neighborhood. The framework still regenerates the implementation and independent tests for the evolved task; it never copies the old implementation.

Illustrative case: a scientific multiple-choice benchmark may test causal discrimination among plausible alternatives. The Agent might explore deeper causal chains or stronger distractor discrimination. The rewritten question and choices must be followed by a separately generated complete reasoning trace and final label; old reasoning or labels are never copied into the new example.

The examples demonstrate how to derive directions from a benchmark. They are not domain templates and must not be copied into every search space.

Sources: [WizardLM paper](https://arxiv.org/abs/2304.12244), [official Evol-Instruct implementation](https://github.com/nlpxucan/WizardLM/tree/main/Evol_Instruct), and [WizardCoder paper](https://arxiv.org/abs/2306.08568).

