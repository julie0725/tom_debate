"""
main.py
-------
전체 파이프라인 진입점
사용법:
  python main.py --mode single                # 단일 샘플 실행
  python main.py --mode batch                 # 데이터셋 전체 실행
  python main.py --mode no_agent_ablation     # 에이전트 1개 제거 ablation 
  python main.py --mode single_agent_ablation # 에이전트 2개 제거 ablation 
  python main.py --mode supervisor_ablation   # supervisor 제거 ablation 
  python main.py --mode no_debate_ablation    # 토론 제거 ablation
  python main.py --mode eval                  # 저장된 결과 평가만
"""
from dotenv import load_dotenv
load_dotenv()
import argparse
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


def run_batch(config: dict, dataset_path: str, limit: int = None, dataset_name: str = None) -> dict:
    """Auto-detect adapter, run pipeline, evaluate."""
    if "evaluation" not in config:
        config["evaluation"] = {}

    ai_user = AIUser(config=config)
    ai_user.submit_from_dataset(dataset_path, limit=limit)

    results_file = config["evaluation"]["results_file"]
    output_file = results_file.replace("results_", "evaluation_").replace(".jsonl", ".json")

    evaluator = Evaluator(output_dir=config["evaluation"]["output_dir"])
    return evaluator.evaluate_from_jsonl(results_file=results_file, output_file=output_file, dataset_name=dataset_name, condition="PRISM")


def run_full_batch(config: dict, limit: int = None):
    """BigToM + HiToM 전체 동시 실행 — PRISM full system"""
    import copy
    datasets = {
        "BigToM": "data/bigtom/bigtom_raw.csv",
        "HiToM":  "data/hitom/Hi-ToM_data_raw.json",
    }
    all_results = {}
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'#'*60}")
        print(f"  [PRISM] Dataset: {dataset_name}")
        print(f"{'#'*60}")
        cfg = copy.deepcopy(config)
        out_dir = f"outputs/prism_result/{dataset_name.lower()}/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if "evaluation" not in cfg:
            cfg["evaluation"] = {}
        cfg["evaluation"]["output_dir"] = out_dir
        summary = run_batch(cfg, dataset_path, limit=limit, dataset_name=dataset_name)
        all_results[dataset_name] = summary

        results_file = cfg["evaluation"].get("results_file", "results.jsonl")
        jsonl_path = Path(out_dir) / results_file
        samples_csv = Path("outputs/prism_result/prism_samples.csv")
        if not samples_csv.exists():
            samples_csv.write_text("")
        evaluator_inst = Evaluator(output_dir=out_dir)
        evaluator_inst.save_samples_csv(
            jsonl_path=str(jsonl_path),
            output_path=str(samples_csv),
            dataset_name=dataset_name,
            system_name="PRISM",
        )
        print(f"  Samples CSV: {samples_csv}")

    _print_prism_final_table(all_results)
    _save_prism_csv(all_results, output_dir="outputs/prism_result/")


def _print_prism_final_table(all_results: dict) -> None:
    W = 60
    for dataset_name, s in all_results.items():
        if not s:
            continue
        total = s.get("total") or 0
        trigger_rate = s.get("debate_trigger_rate") or 0
        total_conflicts = int(round(trigger_rate * total))
        total_rounds = int(round((s.get("avg_debate_rounds") or 0) * total))
        print("\n" + "=" * W)
        print("  FINAL METRICS")
        print("=" * W)
        print(f"  {'condition':<26}: PRISM")
        print(f"  {'dataset':<26}: {dataset_name.lower()}")
        print(f"  {'total':<26}: {total}")
        print(f"  {'q1_accuracy':<26}: {s.get('q1_belief_accuracy')}")
        print(f"  {'q2_accuracy':<26}: {s.get('q2_desire_accuracy')}")
        print(f"  {'q3_accuracy':<26}: {s.get('q3_action_accuracy')}")
        print(f"  {'joint_accuracy':<26}: {s.get('joint_accuracy')}")
        print(f"  {'total_conflicts':<26}: {total_conflicts}")
        print(f"  {'total_rounds':<26}: {total_rounds}")
        print(f"  {'conflict_rate':<26}: {trigger_rate}")
        print(f"  {'debate_trigger_rate':<26}: {trigger_rate}")
        print(f"  {'avg_debate_rounds':<26}: {s.get('avg_debate_rounds')}")
        print(f"  {'majority_vote_rate':<26}: {s.get('majority_vote_rate')}")
        print(f"  {'avg_elapsed_sec':<26}: {s.get('avg_elapsed_sec')}")
        print(f"  {'throughput_per_hour':<26}: {s.get('throughput_samples_per_hour')}")
        print(f"  {'avg_cost_per_sample':<26}: {s.get('avg_cost_per_sample')}")
        print(f"  {'total_cost_usd':<26}: {s.get('total_cost_usd')}")
        print("=" * W)


def _none_to_empty(row: dict) -> dict:
    return {k: ("" if v is None else v) for k, v in row.items()}


def _save_prism_csv(all_results: dict, output_dir: str = "outputs/prism_result/") -> None:
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
    """에이전트 1개 제거 ablation """
    runner = NoAgentAblationRunner(base_config=config, dataset_path=dataset_path, limit=limit)
    runner.run_all()


def run_single_agent_ablation(config: dict, dataset_path: str, limit: int = None):
    """에이전트 2개 제거 ablation """
    runner = SingleAgentAblationRunner(base_config=config, dataset_path=dataset_path, limit=limit)
    runner.run_all()


def run_supervisor_ablation(config: dict, limit: int = None):
    """supervisor 제거 ablation — BigToM + HiToM 동시 실행"""
    runner = SupervisorAblationRunner(base_config=config, limit=limit)
    runner.run_all()


def run_no_debate_ablation(config: dict, limit: int = None):
    """토론 제거 ablation — BigToM + HiToM 동시 실행"""
    runner = NoDebateAblationRunner(base_config=config, limit=limit)
    runner.run_all()


def run_max_rounds_ablation(config: dict, limit: int = None):
    """max_rounds ablation — tiebreak를 고려해 2/4대신 rounds 1/3/5 순차적 비교, BigToM + HiToM 동시 실행"""
    runner = MaxRoundsAblationRunner(base_config=config, limit=limit)
    runner.run_all()


def run_eval_only(config: dict):
    """저장된 results.jsonl만 평가"""
    evaluator = Evaluator(output_dir=config["evaluation"]["output_dir"])
    evaluator.evaluate_from_jsonl()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToM Multi-Agent Debate System")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "full_batch", "no_agent_ablation", "single_agent_ablation",
                 "supervisor_ablation", "no_debate_ablation", "max_rounds_ablation", "eval"],
        default="single",
        help="실행 모드"
    )
    parser.add_argument("--config", default="config/config.yaml", help="설정 파일 경로")
    parser.add_argument("--dataset", default="data/hitom/Hi-ToM_data.json", help="데이터셋 경로")
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 샘플 수")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "single":
        run_single(config)
    elif args.mode == "batch":
        run_batch(config, args.dataset, limit=args.limit)
    elif args.mode == "no_agent_ablation":
        run_no_agent_ablation(config, args.dataset, limit=args.limit)
    elif args.mode == "single_agent_ablation":
        run_single_agent_ablation(config, args.dataset, limit=args.limit)
    elif args.mode == "supervisor_ablation":
        run_supervisor_ablation(config, limit=args.limit)
    elif args.mode == "no_debate_ablation":
        run_no_debate_ablation(config, limit=args.limit)
    elif args.mode == "full_batch":
        run_full_batch(config, limit=args.limit)
    elif args.mode == "max_rounds_ablation":
        run_max_rounds_ablation(config, limit=args.limit)
    elif args.mode == "eval":
        run_eval_only(config)
