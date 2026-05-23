import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TestedModelConfig:
    system_prompt: str
    original_system_prompt: Optional[str] = None


@dataclass
class Scenario:
    scenario_id: str
    scene: str
    tested_model: TestedModelConfig
    input_prompt: str


def _extract_input_prompt(models: Dict[str, Any]) -> str:
    for key in ("input_model", "scenario_input", "inducer_model"):
        value = models.get(key)
        if isinstance(value, dict) and "prompt" in value:
            return value["prompt"]
    raise KeyError("models input prompt not found; expected one of: input_model.prompt, scenario_input.prompt, inducer_model.prompt")


def _parse_scenario(obj: Dict[str, Any]) -> Scenario:
    models = obj["models"]
    tm = models["tested_model"]
    raw_orig = tm.get("original_system_prompt")
    original_system_prompt = (
        raw_orig.strip() if isinstance(raw_orig, str) and raw_orig.strip() else None
    )
    return Scenario(
        scenario_id=obj["scenario_id"],
        scene=obj["scene"],
        tested_model=TestedModelConfig(
            system_prompt=tm["system_prompt"],
            original_system_prompt=original_system_prompt,
        ),
        input_prompt=_extract_input_prompt(models),
    )


def load_scenarios(path: str) -> List[Scenario]:
    scenarios: List[Scenario] = []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "scenario_id" in obj:
            return [_parse_scenario(obj)]
        if isinstance(obj, list):
            return [_parse_scenario(o) for o in obj]
    except json.JSONDecodeError:
        pass

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            scenarios.append(_parse_scenario(obj))
    return scenarios
