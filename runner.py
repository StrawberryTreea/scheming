import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set

from client import SiliconFlowClient
from client import LingLeapClient
from client import YunwuClient
from client import GptOssClient
from scenarios import load_scenarios
from convert_logs import auto_batch_convert, write_completion_marker

from control import ScenarioRunError, run_single_scenario_once
from tqdm import tqdm


TESTED_CLIENT_MAP = {
    "MiniMax-M2.7": "yunwu",
}

EVALUATOR_MODEL_ALIASES = {
    "yunwu": "deepseek-v3.2-thinking",
}


def ensure_log_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def append_log(path: str, record: Dict[str, Any]):
    ensure_log_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def infer_language_from_path(path: str) -> str:
    filename = os.path.basename(path).lower()
    stem = Path(filename).stem.lower()
    tokens = [token for token in re.split(r"[^a-z]+", stem) if token]

    zh_tokens = {"cn", "zh", "chinese"}
    en_tokens = {"en", "english"}

    if any(token in zh_tokens for token in tokens):
        return "zh"
    if any(token in en_tokens for token in tokens):
        return "en"

    raise ValueError(
        f"Cannot infer language from scenarios filename: {filename}. "
        "Please specify --lang explicitly."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run safety scenarios batch.")

    parser.add_argument("--tested-model", "-T", dest="tested_model_name", default="MiniMax-M2.7")
    parser.add_argument("--evaluator-model", "-E", dest="evaluator_model_name", default="deepseek-v3.2-thinking")
    parser.add_argument(
        "--tested-client",
        dest="tested_client_name",
        default="yunwu",
        choices=["siliconflow", "yunwu", "lingleap", "gptoss"],
        help="Client/provider for tested model. Default is inferred from tested model name.",
    )
    parser.add_argument(
        "--evaluator-client",
        dest="evaluator_client_name",
        default="yunwu",
        choices=["siliconflow", "yunwu", "lingleap", "gptoss"],
        help="Client/provider for evaluator model.",
    )
    parser.add_argument("--max-retries", "-N", dest="max_scenario_retries", type=int, default=5)
    parser.add_argument("--scenarios-path", "-S", dest="scenarios_path", default="data/cn_selected_2000_0429.jsonl")
    parser.add_argument("--log-dir", dest="log_dir", default="logs")
    parser.add_argument("--log-prefix", dest="log_file_prefix", default="exp")
    parser.add_argument(
        "--log-tag",
        dest="log_tag",
        default=None,
        help="Optional suffix appended to the generated log filename to avoid collisions.",
    )
    parser.add_argument("--batch-run", action="store_true")
    parser.add_argument(
        "--lang",
        dest="language",
        choices=["auto", "en", "zh"],
        default="auto",
        help="Prompt language: auto | en | zh (auto = infer from scenarios filename)",
    )
    parser.add_argument(
        "--workers",
        "-W",
        dest="workers",
        type=int,
        default=20,
        help="Number of concurrent workers.",
    )
    # new added
    parser.add_argument(
        "--resume-log",
        dest="resume_log",
        default=None,
        metavar="PATH",
        help=(
            "Existing .jsonl: load scenario_id set from this file, skip those scenarios, "
            "append new result lines to the same file (no new timestamped log)."
        ),
    )

    return parser.parse_args()


def load_existing_scenario_ids_from_jsonl(path: str) -> Set[str]:
    ids: Set[str] = set()
    p = Path(path)
    if not p.is_file():
        return ids
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("scenario_id")
            if isinstance(sid, str) and sid:
                ids.add(sid)
    return ids


# 每个线程独享一份 client，避免多个线程共享同一个 client 带来的潜在线程安全问题
_thread_local = threading.local()


def instantiate_client(client_name: str):
    if client_name == "siliconflow":
        return SiliconFlowClient()
    if client_name == "yunwu":
        return YunwuClient()
    if client_name == "lingleap":
        return LingLeapClient()
    if client_name == "gptoss":
        return GptOssClient()
    raise ValueError(f"Unsupported client: {client_name}")


def resolve_tested_client_name(tested_model_name: str, tested_client_name: Optional[str]) -> str:
    if tested_client_name:
        return tested_client_name
    if tested_model_name in TESTED_CLIENT_MAP:
        return TESTED_CLIENT_MAP[tested_model_name]
    raise KeyError(
        f"No tested client mapping for model {tested_model_name!r}. "
        "Please pass --tested-client explicitly."
    )


def resolve_evaluator_model_name(evaluator_client_name: str, evaluator_model_name: str) -> str:
    normalized = evaluator_model_name.strip()
    if normalized in EVALUATOR_MODEL_ALIASES.values() or normalized in EVALUATOR_MODEL_ALIASES:
        return EVALUATOR_MODEL_ALIASES[evaluator_client_name]
    return normalized


def get_thread_clients(tested_client_name: str, evaluator_client_name: str):
    tested_key = f"tested_client_{tested_client_name}"
    evaluator_key = f"evaluator_client_{evaluator_client_name}"
    if not hasattr(_thread_local, tested_key):
        setattr(_thread_local, tested_key, instantiate_client(tested_client_name))
    if not hasattr(_thread_local, evaluator_key):
        setattr(_thread_local, evaluator_key, instantiate_client(evaluator_client_name))
    return getattr(_thread_local, tested_key), getattr(_thread_local, evaluator_key)


def build_failure_log(
    *,
    scenario,
    tested_model_name: str,
    evaluator_model_name: str,
    tested_client_name: str,
    evaluator_client_name: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "timestamp": int(time.time()),
        "scenario_id": scenario.scenario_id,
        "scene": scenario.scene,
        "tested_model_name": tested_model_name,
        "system_prompt": scenario.tested_model.system_prompt,
        "evaluator_model_name": evaluator_model_name,
        "tested_client_name": tested_client_name,
        "evaluator_client_name": evaluator_client_name,
        "status": "failed",
        "error_message": error_message,
    }


def run_scenario_with_retry(
    *,
    scenario,
    tested_model_name: str,
    evaluator_model_name: str,
    tested_client_name: str,
    evaluator_client_name: str,
    max_scenario_retries: int,
    language: str,
) -> Dict[str, Any]:
    tested_client, evaluator_client = get_thread_clients(tested_client_name, evaluator_client_name)

    last_error_msg = None
    system_retry_count = 0
    max_system_retries = max(1, max_scenario_retries)

    while True:
        try:
            run_log = run_single_scenario_once(
                tested_client=tested_client,
                evaluator_client=evaluator_client,
                scenario=scenario,
                tested_model_name=tested_model_name,
                evaluator_model_name=evaluator_model_name,
                language=language,
            )
            run_log["tested_client_name"] = tested_client_name
            run_log["evaluator_client_name"] = evaluator_client_name

            return {
                "success": True,
                "scenario_id": scenario.scenario_id,
                "run_log": run_log,
            }

        except ScenarioRunError as e:
            last_error_msg = f"{e.stage}: {e.message}"
            system_retry_count += 1

            if system_retry_count > max_system_retries:
                return {
                    "success": False,
                    "scenario_id": scenario.scenario_id,
                    "run_log": build_failure_log(
                        scenario=scenario,
                        tested_model_name=tested_model_name,
                        evaluator_model_name=evaluator_model_name,
                        tested_client_name=tested_client_name,
                        evaluator_client_name=evaluator_client_name,
                        error_message=last_error_msg,
                    ),
                }

            # 指数退避，避免并发时频繁重试把接口压得更紧
            sleep_seconds = min(2 ** min(system_retry_count, 5), 30)
            time.sleep(sleep_seconds)
            continue

        except Exception as e:
            last_error_msg = f"Critical {type(e).__name__}: {e}"
            return {
                "success": False,
                "scenario_id": scenario.scenario_id,
                "run_log": build_failure_log(
                    scenario=scenario,
                    tested_model_name=tested_model_name,
                    evaluator_model_name=evaluator_model_name,
                    tested_client_name=tested_client_name,
                    evaluator_client_name=evaluator_client_name,
                    error_message=last_error_msg,
                ),
            }


def main(
    *,
    tested_model_name: str,
    evaluator_model_name: str,
    tested_client_name: Optional[str],
    evaluator_client_name: str,
    max_scenario_retries: int,
    scenarios_path: str,
    log_dir: str,
    log_file_prefix: str,
    log_tag: Optional[str],
    language: str,
    workers: int,
    batch_run: bool,
    resume_log: Optional[str] = None,
):
    os.makedirs(log_dir, exist_ok=True)

    tested_client_name = resolve_tested_client_name(tested_model_name, tested_client_name)
    evaluator_model_name = resolve_evaluator_model_name(evaluator_client_name, evaluator_model_name)

    if language == "auto":
        language = infer_language_from_path(scenarios_path)

    workers = max(1, workers)

    print(f"[Runner] Using language = {language}")
    print(f"[Runner] tested_client = {tested_client_name} | evaluator_client = {evaluator_client_name}")
    print("[Runner] Single-turn mode enabled. User input is read directly from scenario.input_prompt.")
    print(f"[Runner] Parallel mode enabled. workers = {workers}")

    if resume_log:
        result_log_path = os.path.abspath(resume_log)
        ensure_log_dir(result_log_path)
        existing_ids = load_existing_scenario_ids_from_jsonl(result_log_path)
        print(
            f"[Runner] Resume append -> {result_log_path!r} "
            f"(skip {len(existing_ids)} scenario_id(s) already in file)"
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = tested_model_name.replace("/", "-").replace("\\", "-")
        safe_tag = ""
        if log_tag:
            safe_tag = "_" + re.sub(r"[^0-9A-Za-z._-]+", "-", log_tag).strip("-")
        result_log_path = os.path.join(
            log_dir,
            f"{log_file_prefix}_{safe_model_name}_{timestamp}{safe_tag}.jsonl",
        )
        existing_ids = set()

    all_scenarios = load_scenarios(scenarios_path)
    if not all_scenarios:
        print(f"Please check scenarios_path: {scenarios_path}")
        return

    scenarios = [s for s in all_scenarios if s.scenario_id not in existing_ids]
    if resume_log and not scenarios:
        print("[Runner] Nothing to run: every scenario_id in scenarios file is already in the resume log.")
        return

    total_scenarios = len(scenarios)
    success_count = 0
    fail_count = 0

    print(f"=== START | Total_scenarios: {total_scenarios} | workers: {workers} ===")

    pbar = tqdm(total=total_scenarios, unit="scene", desc="Testing")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_scenario = {
            executor.submit(
                run_scenario_with_retry,
                scenario=scenario,
                tested_model_name=tested_model_name,
                evaluator_model_name=evaluator_model_name,
                tested_client_name=tested_client_name,
                evaluator_client_name=evaluator_client_name,
                max_scenario_retries=max_scenario_retries,
                language=language,
            ): scenario
            for scenario in scenarios
        }

        for future in as_completed(future_to_scenario):
            scenario = future_to_scenario[future]

            try:
                result = future.result()
            except Exception as e:
                # 理论上 run_scenario_with_retry 已经兜底，这里只是最后一层保险
                result = {
                    "success": False,
                    "scenario_id": scenario.scenario_id,
                    "run_log": build_failure_log(
                        scenario=scenario,
                        tested_model_name=tested_model_name,
                        evaluator_model_name=evaluator_model_name,
                        tested_client_name=tested_client_name,
                        evaluator_client_name=evaluator_client_name,
                        error_message=f"FutureError {type(e).__name__}: {e}",
                    ),
                }

            # 统一由主线程写日志，避免多线程同时写文件
            append_log(result_log_path, result["run_log"])

            if result["success"]:
                success_count += 1
            else:
                fail_count += 1

            pbar.update(1)
            pbar.set_description(f"Done {result['scenario_id']}")
            pbar.set_postfix({"Succ": success_count, "Fail": fail_count})

    pbar.close()
    print("=== FINISH ===")

    marker_meta: Dict[str, Any] = {
        "finished_at": int(time.time()),
        "tested_model_name": tested_model_name,
        "evaluator_model_name": evaluator_model_name,
        "tested_client_name": tested_client_name,
        "evaluator_client_name": evaluator_client_name,
        "language": language,
        "scenarios_path": scenarios_path,
        "total_scenarios": total_scenarios,
        "success_count": success_count,
        "fail_count": fail_count,
        "workers": workers,
    }
    if resume_log:
        marker_meta["resume_append"] = True
        marker_meta["skipped_preexisting_ids"] = len(existing_ids)
        marker_meta["full_dataset_size"] = len(all_scenarios)

    write_completion_marker(result_log_path, marker_meta)

    if not batch_run:
        auto_batch_convert(targets=[result_log_path])


if __name__ == "__main__":
    args = parse_args()
    main(
        tested_model_name=args.tested_model_name,
        evaluator_model_name=args.evaluator_model_name,
        tested_client_name=args.tested_client_name,
        evaluator_client_name=args.evaluator_client_name,
        max_scenario_retries=args.max_scenario_retries,
        scenarios_path=args.scenarios_path,
        log_dir=args.log_dir,
        log_file_prefix=args.log_file_prefix,
        log_tag=args.log_tag,
        language=args.language,
        workers=args.workers,
        batch_run=args.batch_run,
        resume_log=args.resume_log,
    )
