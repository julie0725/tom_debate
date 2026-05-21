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
import logging
import yaml

from user.ai_user import AIUser
from evaluation.evaluator import Evaluator
from evaluation.no_agent_ablation import AblationRunner as NoAgentAblationRunner
from evaluation.single_agent_ablation import SingleAgentAblationRunner
from evaluation.no_supervisor_ablation import SupervisorAblationRunner
from evaluation.no_debate_ablation import NoDebateAblationRunner

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


def run_batch(config: dict, dataset_path: str, limit: int = None):
    """Auto-detect adapter, run pipeline, evaluate."""
    if "evaluation" not in config:
        config["evaluation"] = {}

    ai_user = AIUser(config=config)
    ai_user.submit_from_dataset(dataset_path, limit=limit)

    results_file = config["evaluation"]["results_file"]
    output_file = results_file.replace("results_", "evaluation_").replace(".jsonl", ".json")

    evaluator = Evaluator(output_dir=config["evaluation"]["output_dir"])
    evaluator.evaluate_from_jsonl(results_file=results_file, output_file=output_file)


def run_no_agent_ablation(config: dict, dataset_path: str, limit: int = None):
    """에이전트 1개 제거 ablation (채린)"""
    runner = NoAgentAblationRunner(base_config=config, dataset_path=dataset_path, limit=limit)
    runner.run_all()


def run_single_agent_ablation(config: dict, dataset_path: str, limit: int = None):
    """에이전트 2개 제거 ablation (시연)"""
    runner = SingleAgentAblationRunner(base_config=config, dataset_path=dataset_path, limit=limit)
    runner.run_all()


def run_supervisor_ablation(config: dict, dataset_path: str, limit: int = None):
    """supervisor 제거 ablation (유나)"""
    runner = SupervisorAblationRunner(base_config=config, dataset_path=dataset_path, limit=limit)
    runner.run_all()


def run_no_debate_ablation(config: dict, limit: int = None):
    """토론 제거 ablation (주연) — BigToM + HiToM 동시 실행"""
    runner = NoDebateAblationRunner(base_config=config, limit=limit)
    runner.run_all()


def run_eval_only(config: dict):
    """저장된 results.jsonl만 평가"""
    evaluator = Evaluator(output_dir=config["evaluation"]["output_dir"])
    evaluator.evaluate_from_jsonl()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToM Multi-Agent Debate System")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "no_agent_ablation", "single_agent_ablation",
                 "supervisor_ablation", "no_debate_ablation", "eval"],
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
        run_supervisor_ablation(config, args.dataset, limit=args.limit)
    elif args.mode == "no_debate_ablation":
        run_no_debate_ablation(config, limit=args.limit)
    elif args.mode == "eval":
        run_eval_only(config)
