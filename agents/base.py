from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class AgentResponse:
    agent_name: str
    system_role: str
    reasoning_summary: str
    structured_output: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseAgent:
    agent_name = "base_agent"
    system_role = "Base deterministic agent"
    prompt_filename = "base_agent.md"

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger.getChild(self.agent_name)
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        prompt_path = PROMPTS_DIR / self.prompt_filename
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""

    def log_event(self, message: str, **payload: Any) -> None:
        if payload:
            self.logger.info("%s | %s", message, json.dumps(payload, sort_keys=True))
        else:
            self.logger.info("%s", message)

    def build_response(
        self,
        reasoning_summary: str,
        structured_output: dict[str, Any],
    ) -> AgentResponse:
        response = AgentResponse(
            agent_name=self.agent_name,
            system_role=self.system_role,
            reasoning_summary=reasoning_summary,
            structured_output=structured_output,
        )
        self.log_event("response_ready", response=response.to_dict())
        return response
