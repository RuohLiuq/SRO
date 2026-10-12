from __future__ import annotations

import importlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DerivedParams:
    max_seq_len: int
    total_train_samples: int
    batch_size: int
    grad_accum: int
    warmup_steps: int
    effective_lr: float


@dataclass
class TrainArtifact:
    checkpoint_path: str
    stages_run: list[str] = field(default_factory=lambda: ["sft"])
    is_mock: bool = False
    info: dict[str, Any] = field(default_factory=dict)


_MODEL_PROFILES = {
    "Qwen/Qwen3-1.7B-Base": {"batch_size": 8, "grad_accum": 4, "max_seq_len": 8192},
    "Qwen/Qwen3-4B-Base": {"batch_size": 2, "grad_accum": 16, "max_seq_len": 8192},
}


def _require(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            f"Search training requires {name!r}; install MDPT with `pip install -e '.[search]'`"
        ) from exc


def _text(sample: dict[str, Any]) -> str:
    parts = [
        str(message.get("content") or "")
        for message in sample.get("messages", [])
        if isinstance(message, dict)
    ]
    if sample.get("tools"):
        parts.append(json.dumps(sample["tools"], ensure_ascii=False))
    for message in sample.get("messages", []):
        if isinstance(message, dict) and message.get("tool_calls"):
            parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    return " ".join(parts)


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def derive_params(
    *,
    base_model: str,
    samples: list[dict[str, Any]],
    sft_hyperparameters: dict[str, float | int],
) -> DerivedParams:
    profile = _MODEL_PROFILES.get(base_model)
    if profile is None:
        raise ValueError(f"unsupported base model profile: {base_model!r}")
    lengths = sorted(int(len(_text(sample).split()) * 1.3) + 1 for sample in samples)
    p95 = lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))] if lengths else 2048
    max_seq_len = min(max(2048, _next_power_of_two(p95)), profile["max_seq_len"])
    epochs = int(sft_hyperparameters.get("epochs", 3))
    batch_size = int(profile["batch_size"])
    grad_accum = int(profile["grad_accum"])
    steps = max(1, len(samples) * max(1, epochs) // max(1, batch_size * grad_accum))
    return DerivedParams(
        max_seq_len=max_seq_len,
        total_train_samples=len(samples),
        batch_size=batch_size,
        grad_accum=grad_accum,
        warmup_steps=max(1, int(steps * 0.05)),
        effective_lr=float(sft_hyperparameters.get("learning_rate", 1e-4)),
    )


def _render_tool_call_completion(message: dict[str, Any], eos_token: str) -> str:
    rendered: list[str] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        rendered.append(
            "<tool_call>\n"
            + json.dumps({"name": function.get("name"), "arguments": arguments}, ensure_ascii=False)
            + "\n</tool_call>"
        )
    if not rendered:
        return ""
    completion = "\n".join(rendered) + "<|im_end|>\n"
    if eos_token and eos_token != "<|im_end|>":
        completion += eos_token
    return completion


class TRLTrainingRunner:
    def __init__(
        self,
        *,
        base_model: str,
        output_root: Path,
        templates_root: Path,
        bf16: bool = True,
    ) -> None:
        if base_model not in _MODEL_PROFILES:
            raise ValueError(f"unsupported base model profile: {base_model!r}")
        self.base_model = base_model
        self.output_root = output_root
        self.templates_root = templates_root
        self.bf16 = bf16
        template_path = templates_root / "qwen3.jinja"
        if "qwen" not in base_model.lower() or not template_path.is_file():
            raise ValueError(f"no verified chat template for {base_model!r}")
        self.chat_template = template_path.read_text(encoding="utf-8")

    def _load_model_and_tokenizer(self):
        torch = _require("torch")
        transformers = _require("transformers")
        dtype = torch.bfloat16 if self.bf16 else torch.float16
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.chat_template = self.chat_template
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.config.use_cache = False
        return model, tokenizer

    def run(
        self,
        *,
        recipe_id: str,
        dataset: list[dict[str, Any]],
        derived: DerivedParams,
        sft_hyperparameters: dict[str, float | int],
    ) -> TrainArtifact:
        if not dataset:
            raise RuntimeError("SFT dataset is empty")
        datasets = _require("datasets")
        peft = _require("peft")
        torch = _require("torch")
        transformers = _require("transformers")

        seed = int(sft_hyperparameters.get("seed", 0))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(transformers, "set_seed"):
            transformers.set_seed(seed)

        model, tokenizer = self._load_model_and_tokenizer()
        has_tool_calls = any(
            sample.get("messages")
            and sample["messages"][-1].get("tool_calls")
            for sample in dataset
        )
        modules_to_save = ["lm_head"] if has_tool_calls else None
        rank = int(sft_hyperparameters.get("lora_rank", 16))
        lora_config = peft.LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            modules_to_save=modules_to_save,
        )
        model = peft.get_peft_model(model, lora_config)

        processed: list[dict[str, list[int]]] = []
        suffix = tokenizer.eos_token or ""
        for sample in dataset:
            messages = sample.get("messages") or []
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                continue
            tools = sample.get("tools") if isinstance(sample.get("tools"), list) else None
            prompt = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                **({"tools": tools} if tools else {}),
            )
            assistant = messages[-1]
            if assistant.get("tool_calls"):
                completion = _render_tool_call_completion(assistant, suffix)
                if not completion:
                    continue
            else:
                completion = str(assistant.get("content") or "") + suffix
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            full_ids = tokenizer(
                prompt + completion,
                add_special_tokens=False,
                truncation=True,
                max_length=derived.max_seq_len,
            ).input_ids
            if len(full_ids) <= len(prompt_ids):
                continue
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
            processed.append(
                {
                    "input_ids": full_ids,
                    "labels": labels[: len(full_ids)],
                    "attention_mask": [1] * len(full_ids),
                }
            )
        if not processed:
            raise RuntimeError("SFT preprocessing produced no usable rows")

        class DataCollator:
            def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
                max_length = max(len(item["input_ids"]) for item in features)
                input_ids, labels, attention = [], [], []
                for item in features:
                    padding = max_length - len(item["input_ids"])
                    input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
                    labels.append(item["labels"] + [-100] * padding)
                    attention.append(item["attention_mask"] + [0] * padding)
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention, dtype=torch.long),
                }

        output_dir = self.output_root / recipe_id / "sft"
        output_dir.mkdir(parents=True, exist_ok=True)
        args = transformers.TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=int(sft_hyperparameters.get("epochs", 3)),
            per_device_train_batch_size=derived.batch_size,
            gradient_accumulation_steps=derived.grad_accum,
            learning_rate=derived.effective_lr,
            bf16=self.bf16,
            seed=seed,
            data_seed=seed,
            logging_steps=10,
            save_strategy="no",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
        )

        class OrderedTrainer(transformers.Trainer):
            def _get_train_sampler(self, *args, **kwargs):
                del args, kwargs
                from torch.utils.data import SequentialSampler

                return SequentialSampler(self.train_dataset) if self.train_dataset is not None else None

        trainer = OrderedTrainer(
            model=model,
            args=args,
            train_dataset=datasets.Dataset.from_list(processed),
            data_collator=DataCollator(),
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return TrainArtifact(
            checkpoint_path=str(output_dir),
            info={
                "n_train": len(processed),
                "backend": "lora",
                "structured_tool_calls": has_tool_calls,
                "derived": derived.__dict__,
            },
        )
