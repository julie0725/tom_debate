"""
main.py
-------
전체 파이프라인 진입점
사용법:
  python main.py --mode single                # 단일 샘플 실행
  python main.py --mode full_system           # BigToM + HiToM 전체 실행 → outputs/results_full_system/
  python main.py --mode bigtom                # BigToM 단독 실행 → outputs/results_bigtom/
  python main.py --mode hitom                 # HiToM 단독 실행 → outputs/results_hitom/
  python main.py --mode no_agent_ablation     # 에이전트 1개 제거 ablation
  python main.py --mode single_agent_ablation # 에이전트 2개 제거 ablation
  python main.py --mode supervisor_ablation   # supervisor 제거 ablation
  python main.py --mode no_debate_ablation    # 토론 제거 ablation
  python main.py --mode eval                  # 저장된 결과 평가만
"""
from dotenv import load_dotenv
load_dotenv()
import argparse
import copy
import csv
import logging
import yaml
from pathlib import Path

from user.ai_user import AIUser
from evaluation.evaluator import Evaluator
from evaluation.no_agent_ablation import AblationRunner as NoAgentAblationRunner
from evaluation.single_agent_ablation import SingleAgentAblationRunner
from evaluation.no_supervisor_ablation import SupervisorAblationRunner
from evaluation.no_debate_ablation import NoDebateAblationRunner
from evaluation.max_rounds_ablation import MaxRoundsAblationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    "bigtom": {
        "path": "data/bigtom/bigtom_raw.csv",
        "display_name": "BigToM",
    },
    "hitom": {
        "path": "data/hitom/Hi-ToM_data_raw.json",
        "display_name": "HiToM",
    },
}


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single(config: dict):
    print("\n" + "="*50)
    print("  PRISM ToM Reasoning System")
    print("  자연어로 시나리오를 입력하세요.")
    print("  (등장인물, 사건, 질문을 자유롭게 입력)")
    print("="*50)
    raw_text = input("\n입력: ").strip()
    if not raw_text:
        print("입력이 없습니다.")
        return

    ai_user = AIUser(config=config)
    state = ai_user.submit_from_text(raw_text=raw_text)

    print("\n" + "="*50)
    print("  FINAL RESULT")
    print("="*50)
    print(f"  Status       : {state.status}")
    print(f"  Debate round : {state.debate_round}")
    print(f"  Q1 (Belief)  : {state.final_answer.get_value('q1')}")
    print(f"  Q2 (Desire)  : {state.final_answer.get_value('q2')}")
    print(f"  Q3 (Action)  : {state.final_answer.get_value('q3')}")
    print("="*50)


def _run_batch(config: dict, dataset_path: str, limit: int = None, dataset_name: str = None) -> dict:
    """Internal: run pipeline on one dataset and evaluate."""
    if "evaluation" not in config:
        config["evaluation"] = {}
    ai_user = AIUser(config=config)
    ai_user.submit_from_dataset(dataset_path, limit=limit)
    results_file = config["evaluation"]["results_file"]
    output_file = results_file.replace("results_", "evaluation_").replace(".jsonl", ".json")
    evaluator = Evaluator(output_dir=config["evaluation"]["output_dir"])
    return evaluator.evaluate_from_jsonl(
        results_file=results_file,
        output_file=output_file,
        dataset_name=dataset_name,
        condition="PRISM",
    )


def _run_dataset_to_dir(config: dict, dataset_key: str, out_dir: str, limit: int = None):
    """Run one dataset, save all outputs to out_dir, return (summary, display_name)."""
    dc = DATASET_CONFIGS[dataset_key]
    cfg = copy.deepcopy(config)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if "evaluation" not in cfg:
        cfg["evaluation"] = {}
    cfg["evaluation"]["output_dir"] = out_dir

    summary = _run_batch(cfg, dc["path"], limit=limit, dataset_name=dc["display_name"])

    results_file = cfg["evaluation"].get("results_file", "results.jsonl")
    jsonl_path = Path(out_dir) / results_file
    samples_csv = Path(out_dir) / "prism_samples.csv"
    Evaluator(output_dir=out_dir).save_samples_csv(
        jsonl_path=str(jsonl_path),
        output_path=str(samples_csv),
        dataset_name=dc["display_name"],
        system_name="PRISM",
    )
    print(f"  Samples CSV : {samples_csv}")
    return summary, dc["display_name"]


def run_full_system(config: dict, limit: int = None):
    """BigToM + HiToM 전체 실행 → outputs/results_full_system/"""
    ROOT = "outputs/results_full_system"
    all_results = {}
    for dataset_key, dc in DATASET_CONFIGS.items():
        print(f"\n{'#'*60}")
        print(f"  [PRISM] Dataset: {dc['display_name']}")
        print(f"{'#'*60}")
        summary, display_name = _run_dataset_to_dir(config, dataset_key, f"{ROOT}/{dataset_key}/", limit=limit)
        all_results[display_name] = summary
    _save_prism_csv(all_results, output_dir=ROOT + "/")


def run_dataset(config: dict, dataset_key: str, limit: int = None):
    """단일 데이터셋 실행 → outputs/results_{dataset_key}/"""
    dc = DATASET_CONFIGS[dataset_key]
    print(f"\n{'#'*60}")
    print(f"  [PRISM] Dataset: {dc['display_name']}")
    print(f"{'#'*60}")
    out_dir = f"outputs/results_{dataset_key}/"
    summary, display_name = _run_dataset_to_dir(config, dataset_key, out_dir, limit=limit)
    _save_prism_csv({display_name: summary}, output_dir=out_dir)


def _none_to_empty(row: dict) -> dict:
    return {k: ("" if v is None else v) for k, v in row.items()}


def _save_prism_csv(all_results: dict, output_dir: str) -> None:
    csv_path = Path(output_dir) / "prism_results.csv"
    fieldnames = [
        "dataset", "system", "total",
        "q1_accuracy", "q2_accuracy", "q3_accuracy", "joint_accuracy",
        "debate_trigger_rate", "majority_vote_rate",
        "avg_debate_rounds", "avg_rounds_debated",
        "avg_elapsed_sec", "total_elapsed_sec", "throughput_samples_per_hour",
        "total_prompt_tokens", "total_completion_tokens",
        "total_cost_usd", "avg_cost_per_sample",
    ]
    rows = []
    for dataset_name, s in all_results.items():
        if not s:
            continue
        rows.append(_none_to_empty({
            "dataset": dataset_name,
            "system": "PRISM",
            "total": s.get("total"),
            "q1_accuracy": s.get("q1_belief_accuracy"),
            "q2_accuracy": s.get("q2_desire_accuracy"),
            "q3_accuracy": s.get("q3_action_accuracy"),
            "joint_accuracy": s.get("joint_accuracy"),
            "debate_trigger_rate": s.get("debate_trigger_rate"),
            "majority_vote_rate": s.get("majority_vote_rate"),
            "avg_debate_rounds": s.get("avg_debate_rounds"),
            "avg_rounds_debated": s.get("avg_debate_rounds_among_debated"),
            "avg_elapsed_sec": s.get("avg_elapsed_sec"),
            "total_elapsed_sec": s.get("total_elapsed_sec"),
            "throughput_samples_per_hour": s.get("throughput_samples_per_hour"),
            "total_prompt_tokens": s.get("total_prompt_tokens"),
            "total_completion_tokens": s.get("total_completion_tokens"),
            "total_cost_usd": s.get("total_cost_usd"),
            "avg_cost_per_sample": s.get("avg_cost_per_sample"),
        }))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results CSV : {csv_path}")


def run_no_agent_ablation(config: dict, dataset_path: str, limit: int = None):
    NoAgentAblationRunner(base_config=config, dataset_path=dataset_path, limit=limit).run_all()


def run_single_agent_ablation(config: dict, dataset_path: str, limit: int = None):
    SingleAgentAblationRunner(base_config=config, dataset_path=dataset_path, limit=limit).run_all()


def run_supervisor_ablation(config: dict, limit: int = None):
    SupervisorAblationRunner(base_config=config, limit=limit).run_all()


def run_no_debate_ablation(config: dict, limit: int = None):
    NoDebateAblationRunner(base_config=config, limit=limit).run_all()


def run_max_rounds_ablation(config: dict, limit: int = None):
    MaxRoundsAblationRunner(base_config=config, limit=limit).run_all()


def run_eval_only(config: dict):
    evaluator = Evaluator(output_dir=config["evaluation"]["output_dir"])
    evaluator.evaluate_from_jsonl()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToM Multi-Agent Debate System")
    parser.add_argument(
        "--mode",
        choices=["single", "full_system", "bigtom", "hitom",
                 "no_agent_ablation", "single_agent_ablation",
                 "supervisor_ablation", "no_debate_ablation",
                 "max_rounds_ablation", "eval"],
        default="single",
        help="실행 모드",
    )
    parser.add_argument("--config", default="config/config.yaml", help="설정 파일 경로")
    parser.add_argument("--dataset", default="data/hitom/Hi-ToM_data.json", help="ablation용 데이터셋 경로")
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 샘플 수")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "single":
        run_single(config)
    elif args.mode == "full_system":
        run_full_system(config, limit=args.limit)
    elif args.mode in DATASET_CONFIGS:
        run_dataset(config, args.mode, limit=args.limit)
    elif args.mode == "no_agent_ablation":
        run_no_agent_ablation(config, args.dataset, limit=args.limit)
    elif args.mode == "single_agent_ablation":
        run_single_agent_ablation(config, args.dataset, limit=args.limit)
    elif args.mode == "supervisor_ablation":
        run_supervisor_ablation(config, limit=args.limit)
    elif args.mode == "no_debate_ablation":
        run_no_debate_ablation(config, limit=args.limit)
    elif args.mode == "max_rounds_ablation":
        run_max_rounds_ablation(config, limit=args.limit)
    elif args.mode == "eval":
        run_eval_only(config)
