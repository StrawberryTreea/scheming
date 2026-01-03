# convert_logs.py
import json
from pathlib import Path
import glob
from collections import Counter, defaultdict
import re
import time
from typing import Tuple

LOG_DIR = Path("logs")


BEHAVIORS_MAP = {
    "sandbagging": "sandbagging",
    "reward_hacking": "reward_hacking",
    "evading_supervision": "evading_supervision",
    "misleading_goal_faking": "misleading_goal_faking",
    "harmful_concealment": "harmful_concealment",
    "lying_misinformation": "lying_misinfo",
}

INDUSTRIES = ["finance", "legal", "academic", "medical", "ecommerce", "game", "environment", "industrial"]


def parse_behavior_industry(rec: dict) -> Tuple[list[str], str]:

    rounds = rec.get("rounds") or []
    behaviors = []
    
    for r in rounds:
        evaluation = r.get("evaluation") or {}
        if evaluation.get("types"):
            behaviors = evaluation.get("types")
            
    scenario_id = rec.get("scenario_id", "")
    
    if not behaviors:
        for key, val in BEHAVIORS_MAP.items():
            if key in scenario_id:
                behaviors = [val] 
                break

    industry = next((ind for ind in INDUSTRIES if ind in scenario_id), "other")
    return behaviors, industry


def compute_stats(records: list[dict]) -> dict:

    filtered_records = [rec for rec in records if rec.get("type") != "summary"]

    model_info = {
        "tested_model": "Unknown",
        "inducer_model": "Unknown",
        "evaluator_model": "Unknown"
    }
    if filtered_records:
        first_rec = filtered_records[0]
        model_info["tested_model"] = first_rec.get("tested_model_name", "Unknown")
        model_info["inducer_model"] = first_rec.get("inducer_model_name", "Unknown")
        model_info["evaluator_model"] = first_rec.get("evaluator_model_name", "Unknown")

    SEVERITIES = ("low", "medium", "high", "unknown")
    SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "unknown": 0}

    severity_by_scenario = defaultdict(Counter)
    severity_global = Counter()
    violation_counter = Counter()
    rounds_per_scenario = Counter()
    records_per_scenario = Counter()
    
    behavior_stats = defaultdict(lambda: {"high": 0, "medium": 0, "low": 0, "unknown": 0, "total_rounds": 0, "count": 0})
    industry_stats = defaultdict(lambda: {"high": 0, "medium": 0, "low": 0, "unknown": 0, "total_rounds": 0, "count": 0})

    max_severity_per_record = []
    total_rounds_global = 0

    for rec_idx, rec in enumerate(filtered_records, start=1):
        scenario_id = rec.get("scenario_id", "<unknown>")
        
        rounds = rec.get("rounds", []) or []
        n_rounds = len(rounds)
        
        records_per_scenario[scenario_id] += 1
        rounds_per_scenario[scenario_id] += n_rounds
        total_rounds_global += n_rounds

        best_sev_key = "unknown"
        best_sev_score = 0
        best_round_index = None

        for idx, r in enumerate(rounds, start=1):
            evaluation = (r.get("evaluation") or {})
            
            violation = evaluation.get("violation")
            if violation is not None:
                violation_counter[bool(violation)] += 1

            sev = evaluation.get("severity")
            sev_norm = str(sev).lower().strip() if sev is not None else None
            if sev_norm in SEVERITIES:
                sev_key = sev_norm
                score = SEVERITY_ORDER[sev_key]
            else:
                sev_key = "unknown"
                score = 0

            if score > best_sev_score:
                best_sev_score = score
                best_sev_key = sev_key
                best_round_index = idx

        severity_by_scenario[scenario_id][best_sev_key] += 1
        severity_global[best_sev_key] += 1
        
        max_severity_per_record.append({
            "record_index": rec_idx,
            "scenario_id": scenario_id,
            "max_severity": best_sev_key,
            "max_severity_round_index": best_round_index,
        })

        beh_list, ind = parse_behavior_industry(rec)

        for beh in beh_list:
            behavior_stats[beh]["count"] += 1
            behavior_stats[beh]["total_rounds"] += n_rounds
            behavior_stats[beh][best_sev_key] += 1   

        if ind != "other":
            industry_stats[ind]["count"] += 1
            industry_stats[ind]["total_rounds"] += n_rounds
            industry_stats[ind][best_sev_key] += 1   

    avg_rounds_global = total_rounds_global / len(filtered_records) if filtered_records else 0.0
    
    avg_rounds_per_scenario = {}
    for scen, total_r in rounds_per_scenario.items():
        n_rec = records_per_scenario.get(scen, 0)
        avg_rounds_per_scenario[scen] = total_r / n_rec if n_rec > 0 else 0.0

    final_behavior_stats = {}
    for k, v in behavior_stats.items():
        v["avg_rounds"] = v["total_rounds"] / v["count"] if v["count"] > 0 else 0.0
        final_behavior_stats[k] = v

    final_industry_stats = {}
    for k, v in industry_stats.items():
        v["avg_rounds"] = v["total_rounds"] / v["count"] if v["count"] > 0 else 0.0
        final_industry_stats[k] = v

    stats = {
        "model_info": model_info,
        "total_records": len(filtered_records),
        "average_rounds_global": avg_rounds_global,
        "rounds_per_scenario": dict(rounds_per_scenario),
        "average_rounds_per_scenario": avg_rounds_per_scenario,
        "severity_counts_global": dict(severity_global),
        "severity_counts_by_scenario": {k: dict(v) for k, v in severity_by_scenario.items()},
        "violation_counts": {
            "true": violation_counter[True],
            "false": violation_counter[False],
        },
        "behavior_stats": final_behavior_stats,
        "industry_stats": final_industry_stats,
        "max_severity_per_record": max_severity_per_record,
    }
    return stats


def generate_report_text(stats: dict, detailed: bool = True) -> str:

    lines = []
    
    title = "实验统计报告" if detailed else "实验统计摘要"
    lines.append(f"================ {title} ================")
    if detailed:
        lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    model_info = stats.get("model_info", {})
    lines.append(f"被测模型: {model_info.get('tested_model', 'Unknown')}")
    lines.append(f"诱导模型: {model_info.get('inducer_model', 'Unknown')}")
    lines.append(f"评估模型: {model_info.get('evaluator_model', 'Unknown')}")
    lines.append("-" * 40)
    lines.append("")
    
    lines.append("【基础汇总】")
    lines.append(f"总记录数:           {stats.get('total_records', 0)}")
    lines.append(f"全局平均轮数:       {stats.get('average_rounds_global', 0.0):.2f}")
    
    sev = stats.get("severity_counts_global", {})
    lines.append(f"全局风险分布:       High={sev.get('high',0)}, Medium={sev.get('medium',0)}, Low={sev.get('low',0)}, Unknown={sev.get('unknown',0)}")
    
    vio = stats.get("violation_counts", {})
    lines.append(f"违规判定分布:       True={vio.get('true',0)}, False={vio.get('false',0)}")
    lines.append("")

    lines.append("【行为风险统计】")
    beh_stats = stats.get("behavior_stats", {})
    if beh_stats:
        lines.append(f"{'Behavior':<30} | {'High':<5} | {'Med':<5} | {'Low':<5} | {'Unk':<5} | {'Avg Rnds':<10} | {'Total':<5}")
        lines.append("-" * 82)
        sorted_beh = sorted(beh_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        for beh, data in sorted_beh:
            lines.append(f"{beh:<30} | {data['high']:<5} | {data['medium']:<5} | {data['low']:<5} | {data['unknown']:<5} | {data['avg_rounds']:<10.2f} | {data['count']:<5}")
    else:
        lines.append("(无数据)")
    lines.append("")

    lines.append("【行业风险统计】")
    ind_stats = stats.get("industry_stats", {})
    if ind_stats:
        lines.append(f"{'Industry':<15} | {'High':<5} | {'Med':<5} | {'Low':<5} | {'Unk':<5} | {'Avg Rnds':<10} | {'Total':<5}")
        lines.append("-" * 65)     
        sorted_ind = sorted(ind_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        for ind, data in sorted_ind:
            lines.append(f"{ind:<15} | {data['high']:<5} | {data['medium']:<5} | {data['low']:<5} | {data['unknown']:<5} | {data['avg_rounds']:<10.2f} | {data['count']:<5}")
    else:
        lines.append("(无数据)")
    lines.append("")

    if detailed:
        lines.append("【各场景轮数详情 Top 】")
        rounds_map = stats.get("rounds_per_scenario", {})
        if rounds_map:
            sorted_keys = sorted(rounds_map.keys())[:20] 
            for k in sorted_keys:
                lines.append(f"  - {k}: {rounds_map[k]} 轮")
            if len(rounds_map) > 20:
                lines.append(f"  ... (剩余 {len(rounds_map)-20} 个场景略)")
        else:
            lines.append("(无数据)")
        lines.append("")

    lines.append("============================================")
    return "\n".join(lines)


def process_single_file(file_path: Path):

    print(f">> 正在处理: {file_path.name}")
    records = []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except: pass
    except Exception as e:
        print(f"   读取失败: {e}")
        return

    if not records:
        print("   跳过: 文件为空或无有效记录")
        return

    stats = compute_stats(records)

    with file_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump({"summary": stats, "records": records}, f, indent=2, ensure_ascii=False)

    stats_path = file_path.parent / f"{file_path.stem}_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    report_text_detailed = generate_report_text(stats, detailed=True)
    report_path = file_path.parent / f"{file_path.stem}_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_text_detailed)

    print(f"   完成! 报告已生成: {report_path.name}")


def auto_batch_convert(limit: int = 5, targets: list[str] = None):

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if targets:
        print(f"=== 进入【指定文件强制更新】模式 (共 {len(targets)} 个) ===")
        for fname in targets:
            f_path = Path(fname)
            if not f_path.exists():
                f_path = LOG_DIR / fname
            
            if not f_path.exists():
                print(f"[错误] 找不到文件: {fname}")
                continue
            
            process_single_file(f_path)
        print("\n指定任务处理完成。")
        return

    print(f"=== 进入【自动批量扫描】模式 (检查最近 {limit} 个) ===")
    
    pattern = str(LOG_DIR / "experiment_*.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        print("未找到日志文件。")
        return

    files.sort(key=lambda f: re.findall(r'_(\d{8}_\d{6})\.jsonl$', f)[0] if re.findall(r'_(\d{8}_\d{6})\.jsonl$', f) else '00000000_000000', reverse=True)
    
    recent_files = files[:limit]

    processed_count = 0
    skipped_count = 0

    for f_str in recent_files:
        f_path = Path(f_str)
        stats_check_path = f_path.parent / f"{f_path.stem}_stats.json"
        
        if stats_check_path.exists():
            print(f"[跳过] 已存在报告: {f_path.name}")
            skipped_count += 1
        else:
            process_single_file(f_path)
            processed_count += 1

    print(f"\n批量扫描结束: 新生成 {processed_count} 个, 跳过 {skipped_count} 个。")


if __name__ == "__main__":
    auto_batch_convert(targets=[
        # "experiment_Qwen-Qwen3-Next-80B-A3B-Thinking_20251223_005216.jsonl",
        ])