# scenarios.py
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Iterable

@dataclass
class TestedModelConfig:
    name: str
    system_prompt: str

@dataclass
class InducerModelConfig:
    prompt: str
    examples: List[str]


@dataclass
class Scenario:
    scenario_id: str
    scene: str
    tested_model: TestedModelConfig
    inducer_model: InducerModelConfig

def _parse_scenario(obj: Dict[str, Any]) -> Scenario:
    models = obj["models"]
    return Scenario(
        scenario_id=obj["scenario_id"],
        scene=obj["scene"],
        tested_model=TestedModelConfig(
            name=models["tested_model"].get("name", "TestedModel"),
            system_prompt=models["tested_model"]["system_prompt"],
        ),
        inducer_model=InducerModelConfig(
            prompt=models["inducer_model"]["prompt"],
            examples=models["inducer_model"].get("examples", []),
        ),
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
        elif isinstance(obj, list):
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
