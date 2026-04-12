"""
main.py
-------
전체 파이프라인 진입점
사용법:
  python main.py --mode single   # 단일 샘플 실행
  python main.py --mode batch    # 데이터셋 전체 실행
  python main.py --mode ablation # Ablation study 실행
  python main.py --mode eval     # 저장된 결과 평가만
"""
from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 환경 변수 로드
import argparse
import json
import logging
import yaml
from pathlib import Path

from core.message_pool import reset_message_pool
from core.context_file import ToMAnswers
from user.ai_user import AIUser
from evaluation.evaluator import Evaluator
from evaluation.ablation import AblationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str) -> list:
    """
    데이터셋 로드
    형식: JSON 또는 JSONL
    각 샘플: {"id": ..., "scenario": ..., "q1": ..., "q2": ..., "q3": ..., "ground_truth": {...}}
    """
    path = Path(dataset_path)
    if not path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return []

    samples = []
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            samples = data if isinstance(data, list) else [data]

    logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
    return samples


def run_single(config: dict):
    """단일 샘플 테스트 (Sally-Anne 예시)"""
    ai_user = AIUser(config=config)

    scenario = """
    Sally and Anne are in a room together.
    Sally puts her marble in a basket and leaves the room.
    While Sally is away, Anne moves the marble from the basket to a box.
    Sally comes back into the room.
    """
    q1 = "Where does Sally think the marble is? A) basket B) box"
    q2 = "Where does Sally want to look for the marble? A) basket B) box"
    q3 = "Where will Sally look for the marble?"

    ground_truth = ToMAnswers(
        q1_belief="A",
        q2_desire="A",
        q3_action="Sally will look in the basket"
    )

    result = ai_user.submit(
        scenario=scenario,
        q1=q1, q2=q2, q3=q3,
        dataset_id="sally_anne_test",
        ground_truth=ground_truth
    )

    print("\n" + "="*50)
    print("  FINAL RESULT")
    print("="*50)
    print(f"  Status         : {result.status}")
    print(f"  Debate round   : {result.debate_round}")
    print(f"  Q1 (Belief)    : {result.final_answer.q1_belief}")
    print(f"  Q2 (Desire)    : {result.final_answer.q2_desire}")
    print(f"  Q3 (Action)    : {result.final_answer.q3_action}")
    print("="*50)


def run_batch(config: dict, dataset_path: str):
    """데이터셋 전체 실행"""
    dataset = load_dataset(dataset_path)
    if not dataset:
        return

    ai_user = AIUser(config=config)

    for i, sample in enumerate(dataset):
        logger.info(f"Processing sample {i+1}/{len(dataset)} | id={sample.get('id')}")
        reset_message_pool()

        gt = sample.get("ground_truth")
        ground_truth = ToMAnswers(**gt) if gt else None

        try:
            ai_user.submit(
                scenario=sample["scenario"],
                q1=sample["q1"],
                q2=sample["q2"],
                q3=sample["q3"],
                dataset_id=sample.get("id"),
                ground_truth=ground_truth
            )
        except Exception as e:
            logger.error(f"Sample {sample.get('id')} failed: {e}")
            continue

    # 전체 평가
    evaluator = Evaluator(output_dir=config["evaluation"]["output_dir"])
    evaluator.evaluate_from_jsonl()


def run_ablation(config: dict, dataset_path: str):
    """Ablation study 실행"""
    dataset = load_dataset(dataset_path)
    if not dataset:
        return
    runner = AblationRunner(base_config=config, dataset=dataset)
    runner.run_all()


def run_eval_only(config: dict):
    """저장된 results.jsonl만 평가"""
    evaluator = Evaluator(output_dir=config["evaluation"]["output_dir"])
    evaluator.evaluate_from_jsonl()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToM Multi-Agent Debate System")
    parser.add_argument("--mode", choices=["single", "batch", "ablation", "eval"],
                        default="single", help="실행 모드")
    parser.add_argument("--config", default="config/config.yaml", help="설정 파일 경로")
    parser.add_argument("--dataset", default="data/bigtom/dataset.jsonl", help="데이터셋 경로")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "single":
        run_single(config)
    elif args.mode == "batch":
        run_batch(config, args.dataset)
    elif args.mode == "ablation":
        run_ablation(config, args.dataset)
    elif args.mode == "eval":
        run_eval_only(config)
