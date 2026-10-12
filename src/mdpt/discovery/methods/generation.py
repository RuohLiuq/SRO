from __future__ import annotations

import gc
import os
import time
from typing import Any


class FrameworkGenerator:
    """Launcher-controlled text generator used only inside MDPT augmentation."""

    def __init__(
        self,
        runtime: dict[str, Any],
        *,
        deadline_monotonic: float | None = None,
    ) -> None:
        del runtime
        self.deadline_monotonic = deadline_monotonic
        self.local_model_path = str(os.environ.get("MDPT_AUGMENTATION_LOCAL_MODEL") or "").strip()
        self.model = str(os.environ.get("MDPT_AUGMENTATION_MODEL") or "").strip()
        self.base_url = os.environ.get("MDPT_AUGMENTATION_BASE_URL")
        self.api_key = os.environ.get("MDPT_AUGMENTATION_API_KEY")
        self.temperature = float(os.environ.get("MDPT_AUGMENTATION_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.environ.get("MDPT_AUGMENTATION_MAX_TOKENS", "2048"))
        self.timeout = float(os.environ.get("MDPT_AUGMENTATION_TIMEOUT_SECONDS", "300"))
        self._client: Any = None
        self._tokenizer: Any = None
        self._local_model: Any = None
        if self.local_model_path:
            return
        if not self.model or not self.base_url or not self.api_key:
            raise RuntimeError(
                "framework augmentation generator is not configured; the launcher must provide either "
                "MDPT_AUGMENTATION_LOCAL_MODEL or the framework model/base-url/key environment"
            )
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=0)

    def _load_local(self) -> None:
        if self._local_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._local_model = AutoModelForCausalLM.from_pretrained(
            self.local_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self._local_model.eval()

    def complete(self, prompt: str) -> str:
        if self.local_model_path:
            if self.deadline_monotonic is not None:
                raise RuntimeError(
                    "local augmentation is disabled for Search because it cannot enforce the Trial/Search hard deadline"
                )
            import torch

            self._load_local()
            try:
                rendered = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                )
            except Exception:
                rendered = prompt
            encoded = self._tokenizer(rendered, return_tensors="pt").to(self._local_model.device)
            with torch.no_grad():
                generated = self._local_model.generate(
                    **encoded,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.temperature > 0,
                    temperature=max(self.temperature, 1e-5),
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            return self._tokenizer.decode(
                generated[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True
            ).strip()
        remaining = (
            self.deadline_monotonic - time.monotonic()
            if self.deadline_monotonic is not None
            else None
        )
        if remaining is not None and remaining <= 0:
            raise TimeoutError("augmentation request exceeded the Trial/Search deadline")
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=min(self.timeout, remaining) if remaining is not None else self.timeout,
        )
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        if getattr(choice, "finish_reason", None) == "length":
            raise RuntimeError("augmentation model exhausted max_tokens before returning usable data")
        if not content:
            raise RuntimeError("augmentation model returned empty content")
        return content

    def close(self) -> None:
        self._client = None
        self._tokenizer = None
        self._local_model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
