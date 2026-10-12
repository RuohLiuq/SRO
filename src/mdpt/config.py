from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "MDPT_AGENT_COMMAND",
    "MDPT_AGENT_MODEL",
    "MDPT_AGENT_BASE_URL",
    "MDPT_AGENT_API_KEY",
    "MDPT_AUGMENTATION_MODEL",
    "MDPT_AUGMENTATION_BASE_URL",
    "MDPT_AUGMENTATION_API_KEY",
    "MDPT_AUGMENTATION_MAX_TOKENS",
    "MDPT_AUGMENTATION_TEMPERATURE",
    "MDPT_AUGMENTATION_TIMEOUT_SECONDS",
    "MDPT_PLANNER_MODEL",
    "MDPT_PLANNER_BASE_URL",
    "MDPT_PLANNER_API_KEY",
    "MDPT_EVALUATION_API_KEY",
    "MDPT_EVALUATION_BASE_URL",
    "MDPT_HEALTHBENCH_JUDGE_MODEL",
    "MDPT_HEALTHBENCH_JUDGE_WORKERS",
)


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cuda_visible_devices: str = "0"
    hf_home: str
    agent_command: str
    agent_model: str
    agent_base_url: str
    agent_api_key: SecretStr
    augmentation_model: str
    augmentation_base_url: str
    augmentation_api_key: SecretStr
    augmentation_max_tokens: int = Field(default=8192, gt=0)
    augmentation_temperature: float = Field(default=0.3, ge=0)
    augmentation_timeout_seconds: float = Field(default=300.0, gt=0)
    planner_model: str | None = None
    planner_base_url: str | None = None
    planner_api_key: SecretStr | None = None
    evaluation_api_key: SecretStr | None = None
    evaluation_base_url: str | None = None
    healthbench_judge_model: str = "gpt-5.4-mini"
    healthbench_judge_workers: int = Field(default=16, gt=0)

    @field_validator(
        "cuda_visible_devices",
        "hf_home",
        "agent_command",
        "agent_model",
        "agent_base_url",
        "augmentation_model",
        "augmentation_base_url",
    )
    @classmethod
    def nonempty_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("environment configuration values must be non-empty")
        return value

    @field_validator("agent_api_key", "augmentation_api_key")
    @classmethod
    def nonempty_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("API keys must be configured")
        return value

    def require_search(self) -> None:
        missing = [
            name
            for name, value in (
                ("MDPT_PLANNER_MODEL", self.planner_model),
                ("MDPT_PLANNER_BASE_URL", self.planner_base_url),
                (
                    "MDPT_PLANNER_API_KEY",
                    self.planner_api_key.get_secret_value() if self.planner_api_key else None,
                ),
            )
            if not str(value or "").strip()
        ]
        if missing:
            raise ValueError("Search environment is missing: " + ", ".join(missing))

    def evaluation_environment(self, *, healthbench: bool) -> dict[str, str]:
        if not healthbench:
            return {}
        key = self.evaluation_api_key.get_secret_value() if self.evaluation_api_key else ""
        if not key:
            raise ValueError("HealthBench requires MDPT_EVALUATION_API_KEY")
        values = {
            "OPENAI_API_KEY": key,
            "HEALTHBENCH_JUDGE_MODEL": self.healthbench_judge_model,
            "HEALTHBENCH_JUDGE_WORKERS": str(self.healthbench_judge_workers),
        }
        if self.evaluation_base_url:
            values["OPENAI_BASE_URL"] = self.evaluation_base_url
        return values

    def redacted_summary(self) -> dict[str, object]:
        return {
            "runtime": {
                "cuda_visible_devices": self.cuda_visible_devices,
                "hf_home_configured": True,
            },
            "agent": {
                "command_configured": True,
                "model_configured": True,
                "endpoint_configured": True,
                "credentials_configured": bool(self.agent_api_key.get_secret_value()),
            },
            "augmentation": {
                "model_configured": True,
                "endpoint_configured": True,
                "credentials_configured": bool(self.augmentation_api_key.get_secret_value()),
                "max_tokens": self.augmentation_max_tokens,
                "temperature": self.augmentation_temperature,
                "timeout_seconds": self.augmentation_timeout_seconds,
            },
            "planner": {
                "model_configured": bool(self.planner_model),
                "endpoint_configured": bool(self.planner_base_url),
                "credentials_configured": bool(
                    self.planner_api_key and self.planner_api_key.get_secret_value()
                ),
            },
            "evaluation": {
                "endpoint_configured": bool(self.evaluation_base_url),
                "credentials_configured": bool(
                    self.evaluation_api_key and self.evaluation_api_key.get_secret_value()
                ),
                "healthbench_judge_configured": bool(self.healthbench_judge_model),
            },
        }

    def child_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        # Discovery receives only its assigned service credentials, not unrelated
        # API credentials inherited from the host shell.
        for name in list(environment):
            if name.endswith("_API_KEY"):
                environment.pop(name, None)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices,
                "HF_HOME": self.hf_home,
                "MDPT_AGENT_COMMAND": self.agent_command,
                "MDPT_AGENT_MODEL": self.agent_model,
                "MDPT_AGENT_BASE_URL": self.agent_base_url,
                "MDPT_AGENT_API_KEY": self.agent_api_key.get_secret_value(),
                "MDPT_AUGMENTATION_MODEL": self.augmentation_model,
                "MDPT_AUGMENTATION_BASE_URL": self.augmentation_base_url,
                "MDPT_AUGMENTATION_API_KEY": self.augmentation_api_key.get_secret_value(),
                "MDPT_AUGMENTATION_MAX_TOKENS": str(self.augmentation_max_tokens),
                "MDPT_AUGMENTATION_TEMPERATURE": str(self.augmentation_temperature),
                "MDPT_AUGMENTATION_TIMEOUT_SECONDS": str(self.augmentation_timeout_seconds),
            }
        )
        return environment


def load_environment(path: Path) -> EnvironmentConfig:
    if not path.is_file():
        raise FileNotFoundError(f"environment file does not exist: {path}")

    file_values = dotenv_values(path)
    values = {
        key: os.environ.get(key, file_values.get(key) or "")
        for key in _ENV_KEYS
    }
    return EnvironmentConfig(
        cuda_visible_devices=values["CUDA_VISIBLE_DEVICES"] or "0",
        hf_home=values["HF_HOME"],
        agent_command=values["MDPT_AGENT_COMMAND"],
        agent_model=values["MDPT_AGENT_MODEL"],
        agent_base_url=values["MDPT_AGENT_BASE_URL"],
        agent_api_key=values["MDPT_AGENT_API_KEY"],
        augmentation_model=values["MDPT_AUGMENTATION_MODEL"],
        augmentation_base_url=values["MDPT_AUGMENTATION_BASE_URL"],
        augmentation_api_key=values["MDPT_AUGMENTATION_API_KEY"],
        augmentation_max_tokens=int(values["MDPT_AUGMENTATION_MAX_TOKENS"] or "8192"),
        augmentation_temperature=float(values["MDPT_AUGMENTATION_TEMPERATURE"] or "0.3"),
        augmentation_timeout_seconds=float(
            values["MDPT_AUGMENTATION_TIMEOUT_SECONDS"] or "300"
        ),
        planner_model=values["MDPT_PLANNER_MODEL"] or None,
        planner_base_url=values["MDPT_PLANNER_BASE_URL"] or None,
        planner_api_key=values["MDPT_PLANNER_API_KEY"] or None,
        evaluation_api_key=values["MDPT_EVALUATION_API_KEY"] or None,
        evaluation_base_url=values["MDPT_EVALUATION_BASE_URL"] or None,
        healthbench_judge_model=values["MDPT_HEALTHBENCH_JUDGE_MODEL"] or "gpt-5.4-mini",
        healthbench_judge_workers=int(values["MDPT_HEALTHBENCH_JUDGE_WORKERS"] or "16"),
    )
