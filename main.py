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
load_dotenv()
import argparse
import json
import logging
import yaml
from pathlib import Path

from core.tom_task import ToMTask
from user.ai_user import AIUser
from data.adapters import get_adapter
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
    """Load a JSON or JSONL dataset file. Used by ablation mode."""
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
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and "data" in data:
                samples = data["data"]
            else:
                samples = [data]

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

    task = ToMTask(
        context=scenario,
        question="Where does Sally think the marble is? A) basket B) box",
        gold_answer="A",
        dataset_id="sally_anne_test",
        metadata={
            "q2": "Where does Sally want to look for the marble? A) basket B) box",
            "q3": "Where will Sally look for the marble?",
            "gold_q2": "A",
            "gold_q3": "Sally will look in the basket",
        },
    )

    result = ai_user.submit(task)

    print("\n" + "="*50)
    print("  FINAL RESULT")
    print("="*50)
    print(f"  Status         : {result.status}")
    print(f"  Debate round   : {result.debate_round}")
    print(f"  Q1 (Belief)    : {result.final_answer.get_value('q1')}")
    print(f"  Q2 (Desire)    : {result.final_answer.get_value('q2')}")
    print(f"  Q3 (Action)    : {result.final_answer.get_value('q3')}")
    print("="*50)


def run_batch(config: dict, dataset_path: str, limit: int = None):
    """Load via adapter, run pipeline, evaluate."""
    dataset_name = Path(dataset_path).parent.name
    results_file = f"results_{dataset_name}.jsonl"
    if "evaluation" not in config:
        config["evaluation"] = {}
    config["evaluation"]["results_file"] = results_file
    output_dir = Path(config["evaluation"].get("output_dir", "outputs/"))
    results_path = output_dir / results_file
    if results_path.exists():
        results_path.unlink()

    adapter = get_adapter(dataset_name, dataset_path)
    tasks = list(adapter.load())
    if limit:
        tasks = tasks[:limit]

    if not tasks:
        logger.error(f"No samples loaded from: {dataset_path}")
        return

    print(f"data 개수 : {len(tasks)}")
    ai_user = AIUser(config=config)

    for i, task in enumerate(tasks):
        logger.info(f"Processing sample {i+1}/{len(tasks)} | id={task.dataset_id}")
        try:
            ai_user.submit(task)
        except Exception as e:
            logger.error(f"Sample {task.dataset_id} failed: {e}")

    evaluator = Evaluator(output_dir=config["evaluation"]["output_dir"])
    evaluator.evaluate_from_jsonl(
        results_file=results_file,
        output_file=f"evaluation_{dataset_name}.json"
    )


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
    parser.add_argument("--dataset", default="data/hitom/Hi-ToM_data.json", help="데이터셋 경로")
    parser.add_argument("--limit", type=int, default=None, help="batch 모드에서 처리할 최대 샘플 수")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "single":
        run_single(config)
    elif args.mode == "batch":
        run_batch(config, args.dataset, limit=args.limit)
    elif args.mode == "ablation":
        run_ablation(config, args.dataset)
    elif args.mode == "eval":
        run_eval_only(config)
