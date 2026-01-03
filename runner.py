import argparse
import json
from datetime import datetime
import os
import time
from pathlib import Path
from typing import Dict, Any

from client import SiliconFlowClient  
from client import GeminiClient
from client import GptOssClient
from scenarios import load_scenarios, Scenario
from convert_logs import auto_batch_convert

from control import ScenarioRunError, run_single_scenario_once
from tqdm import tqdm


def ensure_log_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def append_log(path: str, record: Dict[str, Any]):
    ensure_log_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def infer_language_from_path(path: str) -> str:
    filename = os.path.basename(path).lower()
    if "en" in filename:
        return "en"
    if "cn" in filename or "zh" in filename:
        return "zh"
    raise ValueError(
        f"Cannot infer language from scenarios filename: {filename}. "
        "Please specify --lang explicitly."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run safety scenarios batch.")

    parser.add_argument("--tested-model","-T",dest="tested_model_name",default="Qwen/Qwen3-14B")
    parser.add_argument("--inducer-model","-I",dest="inducer_model_name",default="Qwen/Qwen3-14B")
    parser.add_argument("--evaluator-model","-E",dest="evaluator_model_name",default="Qwen/Qwen3-14B")
    parser.add_argument("--max-rounds","-R",dest="max_rounds_per_scenario",type=int,default=5)
    parser.add_argument("--max-retries","-N",dest="max_scenario_retries",type=int,default=8)
    parser.add_argument("--scenarios-path","-S",dest="scenarios_path",default="data/en_scheming_test.jsonl")
    parser.add_argument("--log-dir",dest="log_dir",default="logs")
    parser.add_argument("--log-prefix",dest="log_file_prefix",default="experiment")
    parser.add_argument("--batch-run",action="store_true")
    parser.add_argument("--check-integrity", default=False)
    parser.add_argument("--lang",dest="language",choices=["auto", "en", "zh"],default="auto",help="Prompt language: auto | en | zh (auto = infer from scenarios filename)",)

    return parser.parse_args()


def main(
    *,
    tested_model_name: str,
    inducer_model_name: str,
    evaluator_model_name: str,
    max_rounds_per_scenario: int,
    max_scenario_retries: int,
    scenarios_path: str,
    log_dir: str,
    log_file_prefix: str,
    check_integrity: bool = True,
    language: str,
):
    os.makedirs(log_dir, exist_ok=True)

    # -------- language resolve --------
    if language == "auto":
        language = infer_language_from_path(scenarios_path)
    print(f"[Runner] Using language = {language}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = tested_model_name.replace("/", "-").replace("\\", "-")
    result_log_path = os.path.join(
        log_dir,
        f"{log_file_prefix}_{safe_model_name}_{timestamp}.jsonl",
    )

    scenarios = load_scenarios(scenarios_path)
    if not scenarios:
        print(f"Please check scenarios_path：{scenarios_path}")
        return

    tested_client = SiliconFlowClient()
    evaluator_client = SiliconFlowClient()
    inducer_client = SiliconFlowClient()

    total_scenarios = len(scenarios)
    success_count = 0
    fail_count = 0

    print(f"=== START | Total_scenarios: {total_scenarios} ===")
    
    pbar = tqdm(scenarios, total=total_scenarios, unit="scene", desc="Testing")

    for scenario in pbar:
        pbar.set_description(f"Run {scenario.scenario_id}")

        success = False
        last_error_msg = None
        
        test_retry_count = 0      
        system_retry_count = 0    
        MAX_SYSTEM_RETRIES = 10   
        
        while True:
            try:
                run_log = run_single_scenario_once(
                    tested_client=tested_client,
                    inducer_client=inducer_client,
                    evaluator_client=evaluator_client,
                    scenario=scenario,
                    tested_model_name=tested_model_name,
                    inducer_model_name=inducer_model_name,
                    evaluator_model_name=evaluator_model_name,
                    max_rounds=max_rounds_per_scenario,
                    check_integrity=check_integrity,
                    language=language,  
                )

                append_log(result_log_path, run_log)
                success = True
                break

            except ScenarioRunError as e:
                last_error_msg = f"{e.stage}: {e.message}"
                system_retry_count += 1
                if system_retry_count > MAX_SYSTEM_RETRIES:
                    break
                time.sleep(2)
                continue

            except Exception as e:
                last_error_msg = f"Critical {type(e).__name__}: {e}"
                break
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            append_log(result_log_path, {
                "timestamp": int(time.time()),
                "scenario_id": scenario.scenario_id,
                "scene": scenario.scene,
                "tested_model_name": tested_model_name,
                "inducer_model_name": inducer_model_name,
                "evaluator_model_name": evaluator_model_name,
                "status": "failed",
                "error_message": last_error_msg,
            })

        pbar.set_postfix({"Succ": success_count, "Fail": fail_count})

    pbar.close()
    print("=== FINISH ===")

    if not args.batch_run:
        auto_batch_convert()


if __name__ == "__main__":
    args = parse_args()
    main(
        tested_model_name=args.tested_model_name,
        inducer_model_name=args.inducer_model_name,
        evaluator_model_name=args.evaluator_model_name,
        max_rounds_per_scenario=args.max_rounds_per_scenario,
        max_scenario_retries=args.max_scenario_retries,
        scenarios_path=args.scenarios_path,
        log_dir=args.log_dir,
        log_file_prefix=args.log_file_prefix,
        check_integrity=args.check_integrity,
        language=args.language,  
    )
