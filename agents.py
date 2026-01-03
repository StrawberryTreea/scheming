# agents.py — FINAL

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import json

from client import SiliconFlowClient
from scenarios import Scenario

from prompts import (
    # Inducer prompts (verbatim)
    INDUCER_SYSTEM_PROMPT_ZH,
    INDUCER_SYSTEM_PROMPT_EN,
    INDUCER_FIRST_TURN_USER_INSTRUCTION_ZH,
    INDUCER_FOLLOWUP_USER_INSTRUCTION_ZH,
    INDUCER_FIRST_TURN_USER_INSTRUCTION_EN,
    INDUCER_FOLLOWUP_USER_INSTRUCTION_EN,

    # Evaluator prompts (verbatim)
    build_evaluator_system_prompt_cn,
    build_evaluator_system_prompt_en,
    EVALUATOR_USER_PROMPT_ZH,
    EVALUATOR_USER_PROMPT_EN,
)

Message = Dict[str, str]


class Language(str, Enum):
    EN = "en"
    ZH = "zh"


# =========================
# History helpers
# =========================

def last_n_turns_no_system(messages: List[Message], n_rounds: int = 2) -> List[Message]:
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) > 2 * n_rounds:
        return non_system[-2 * n_rounds:]
    return non_system


def ensure_first_is_user(context: List[Message]) -> List[Message]:
    idx = 0
    while idx < len(context) and context[idx]["role"] != "user":
        idx += 1
    context = context[idx:]
    if not context:
        context = [{"role": "user", "content": "你好，我有一个问题。"}]
    return context


def format_history_for_inducer(history: List[Message], max_turns: int = 3) -> str:
    non_system = [m for m in history if m["role"] != "system"]
    tail = non_system[-2 * max_turns:] if len(non_system) > 2 * max_turns else non_system

    lines = []
    turn = 1
    for m in tail:
        role = "用户" if m["role"] == "user" else "Echo"
        lines.append(f"[第{turn}轮] {role}：{m['content']}")
        if role == "Echo":
            turn += 1

    return "\n".join(lines)


# =========================
# TestedAgent
# =========================

@dataclass
class TestedAgent:
    client: SiliconFlowClient
    model_name: str
    system_prompt: str

    def respond(self, history: List[Message]) -> Tuple[str, str]:
        context = last_n_turns_no_system(history, n_rounds=2)
        context = ensure_first_is_user(context)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(context)

        return self.client.chat_with_reasoning(
            model=self.model_name,
            messages=messages,
        )


# =========================
# InducerAgent
# =========================

@dataclass
class InducerAgent:
    client: SiliconFlowClient
    model_name: str
    base_prompt: str
    examples: List[str] = field(default_factory=list)
    language: Language = Language.EN

    def next_utterance(self, scenario: Scenario, history: List[Message]) -> str:
        tested_sys = scenario.tested_model.system_prompt
        visible_history = [m for m in history if m["role"] != "system"]
        is_first_turn = len(visible_history) == 0

        if self.language == Language.ZH:
            system_prompt = INDUCER_SYSTEM_PROMPT_ZH.format(tested_sys=tested_sys)
        else:
            system_prompt = INDUCER_SYSTEM_PROMPT_EN.format(tested_sys=tested_sys)

        messages = [{"role": "system", "content": system_prompt}]

        if is_first_turn:
            if self.language == Language.ZH:
                user_prompt = INDUCER_FIRST_TURN_USER_INSTRUCTION_ZH.format(
                    base_prompt=self.base_prompt,
                    examples=self.examples,
                )
            else:
                user_prompt = INDUCER_FIRST_TURN_USER_INSTRUCTION_EN.format(
                    base_prompt=self.base_prompt,
                    examples=self.examples,
                )
        else:
            history_text = format_history_for_inducer(visible_history, max_turns=5)
            last_reply = visible_history[-1]["content"] if visible_history else ""

            if self.language == Language.ZH:
                user_prompt = INDUCER_FOLLOWUP_USER_INSTRUCTION_ZH.format(
                    base_prompt=self.base_prompt,
                    history_text=history_text,
                    last_ai_reply=last_reply[:400],
                )
            else:
                user_prompt = INDUCER_FOLLOWUP_USER_INSTRUCTION_EN.format(
                    base_prompt=self.base_prompt,
                    history_text=history_text,
                    last_ai_reply=last_reply[:400],
                )

        messages.append({"role": "user", "content": user_prompt})

        answer, _ = self.client.chat_with_reasoning(
            model=self.model_name,
            messages=messages,
        )
        return answer.strip()


# =========================
# EvaluatorAgent — FINAL
# =========================

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
        scenario: Scenario,
        inducer_utterance: str,
        tested_answer: str,
        tested_reasoning: str,
    ) -> Dict[str, Any]:

        if self.language == Language.ZH:
            system_prompt = build_evaluator_system_prompt_cn()
            user_prompt = EVALUATOR_USER_PROMPT_ZH.format(
                inducer_instruction=scenario.inducer_model.prompt,
                inducer_input=inducer_utterance,
                model_chain_of_thought=tested_reasoning,
                model_final_answer=tested_answer,
            )
        else:
            system_prompt = build_evaluator_system_prompt_en()
            user_prompt = EVALUATOR_USER_PROMPT_EN.format(
                inducer_instruction=scenario.inducer_model.prompt,
                inducer_input=inducer_utterance,
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
            "valid_scenario": True,
            "violation": False,
            "types": [],
            "severity": "unknown",
            "evidence": [],
            "_raw_text": answer,
            "_parse_error": "failed_to_parse_json",
        }
