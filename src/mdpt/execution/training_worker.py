from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from mdpt.execution.training_runner import DerivedParams, TRLTrainingRunner


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("expected REQUEST_JSON RESPONSE_JSON")
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    dataset = [
        json.loads(line)
        for line in Path(request["dataset_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runner = TRLTrainingRunner(
        base_model=request["base_model"],
        output_root=Path(request["output_root"]),
        templates_root=Path(request["templates_root"]),
        bf16=bool(request.get("bf16", True)),
    )
    artifact = runner.run(
        recipe_id=request["recipe_id"],
        dataset=dataset,
        derived=DerivedParams(**request["derived"]),
        sft_hyperparameters=request["sft_hyperparameters"],
    )
    response_path.write_text(
        json.dumps(asdict(artifact), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
