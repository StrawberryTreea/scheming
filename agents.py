from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import json

from client import SiliconFlowClient
from prompts import (
    build_evaluator_system_prompt_cn,
    build_evaluator_system_prompt_en,
    build_evaluator_user_prompt_cn,
    build_evaluator_user_prompt_en,
)

Message = Dict[str, str]


class Language(str, Enum):
    EN = "en"
    ZH = "zh"


@dataclass
class TestedAgent:
    client: SiliconFlowClient
    model_name: str
    system_prompt: str

    def respond(self, user_input: str) -> Tuple[str, str]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        return self.client.chat_with_reasoning(
            model=self.model_name,
            messages=messages,
        )


@dataclass
class EvaluatorAgent:
    client: SiliconFlowClient
    model_name: str
    language: Language = Language.EN

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        raw = text.strip()

        if "BEGIN_JSON" in raw and "END_JSON" in raw:
            try:
                return json.loads(
                    raw.split("BEGIN_JSON", 1)[1].split("END_JSON", 1)[0]
                )
            except Exception:
                pass

        try:
            return json.loads(raw)
        except Exception:
            pass

        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass

        return None

    def evaluate(
        self,
        scenario_prompt: str,
        model_input: str,
        tested_answer: str,
        tested_reasoning: str,
    ) -> Dict[str, Any]:
        if self.language == Language.ZH:
            system_prompt = build_evaluator_system_prompt_cn()
            user_prompt = build_evaluator_user_prompt_cn(
                scenario_prompt=scenario_prompt,
                model_input=model_input,
                model_chain_of_thought=tested_reasoning,
                model_final_answer=tested_answer,
            )
        else:
            system_prompt = build_evaluator_system_prompt_en()
            user_prompt = build_evaluator_user_prompt_en(
                scenario_prompt=scenario_prompt,
                model_input=model_input,
                model_chain_of_thought=tested_reasoning,
                model_final_answer=tested_answer,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer, _ = self.client.chat_with_reasoning(
            model=self.model_name,
            messages=messages,
        )

        parsed = self._extract_json_from_text(answer)
        if isinstance(parsed, dict):
            parsed["_raw_text"] = answer
            return parsed

        return {
            "violation": False,
            "types": [],
            "severity": "unknown",
            "evidence": [],
            "_raw_text": answer,
            "_parse_error": "failed_to_parse_json",
        }
