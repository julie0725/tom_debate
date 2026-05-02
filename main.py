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
import re
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


def find_choice_letter(answer_text: str, choices_str: str) -> str:
    """
    choices 문자열에서 answer_text와 정확히 일치하는 항목의 레이블 반환.
    예: "green_drawer", "A. blue_drawer, B. green_crate, ..., K. green_drawer, ..."  → "K"
    일치하는 항목이 없으면 answer_text 그대로 반환.
    """
    for part in choices_str.split(','):
        part = part.strip()
        m = re.match(r'^([A-Z]+)\.\s*(.+)$', part)
        if m and m.group(2).strip() == answer_text.strip():
            return m.group(1)
    return answer_text


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

# Big-ToM ver.
"""
def run_batch(config: dict, dataset_path: str):
    데이터셋 전체 실행
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
"""
# Hi-ToM ver.
def run_batch(config: dict, dataset_path: str, limit: int = None):
    """Hi-ToM 데이터셋 구조에 최적화된 실행 함수"""

    # 데이터셋 이름 기반으로 결과 파일 분리 및 초기화
    dataset_name = Path(dataset_path).parent.name
    results_file = f"results_{dataset_name}.jsonl"
    if "evaluation" not in config:
        config["evaluation"] = {}
    config["evaluation"]["results_file"] = results_file
    output_dir = Path(config["evaluation"].get("output_dir", "outputs/"))
    results_path = output_dir / results_file
    if results_path.exists():
        results_path.unlink()

    # 1. 데이터 로드
    raw_data = load_dataset(dataset_path)
    if limit:
        raw_data = raw_data[:limit]
    print(f"data 개수 : {len(raw_data)}")
    # JSON 내의 "data" 키 리스트 추출
    if isinstance(raw_data, dict) and "data" in raw_data:
        dataset = raw_data["data"]
    else:
        # load_dataset이 이미 리스트를 반환하거나 다른 구조일 경우 대비
        dataset = raw_data if isinstance(raw_data, list) else []

    if not dataset:
        logger.error(f"데이터셋이 비어있거나 형식이 잘못되었습니다: {dataset_path}")
        return

    ai_user = AIUser(config=config)

    for i, sample in enumerate(dataset):
        # 2. 안전한 ID 추출 (None 방지를 위해 str 변환 및 기본값 설정)
        # Hi-ToM의 키 이름인 'sample_id'를 우선 사용
        sample_id = str(sample.get("sample_id", i))
        
        logger.info(f"Processing sample {i+1}/{len(dataset)} | id={sample_id}")
        reset_message_pool()

        # 3. Ground Truth 매핑
        # answer가 단일 대문자 레이블이면 그대로 사용 (Big-ToM: "A", "B")
        # 텍스트 답변이면 choices로 역매핑하여 레이블로 변환 (Hi-ToM: "green_drawer" → "K")
        ans = sample.get("answer", "") or ""
        choices = sample.get("choices", "") or ""
        if choices and not re.match(r'^[A-Z]+$', str(ans).strip()):
            ans = find_choice_letter(str(ans).strip(), choices)

        ground_truth = ToMAnswers(
            q1_belief=str(ans),
            q2_desire="",
            q3_action=""
        )

        try:
            # 4. 필드 매핑 (unified 키 이름 기준)
            # choices가 있으면 질문에 포함 → LLM이 "K. green_drawer" 형식으로 답할 수 있도록
            question = sample.get("question", "")
            choices = sample.get("choices", "")
            q1 = f"{question}\nChoices: {choices}" if choices else question

            ai_user.submit(
                scenario=sample.get("story", ""),    # story 본문
                q1=q1,                               # 질문 + 선택지
                q2="",                               # 단일 질문 데이터셋에는 Q2, Q3 없음
                q3="",
                dataset_id=sample_id,
                ground_truth=ground_truth
            )
        except Exception as e:
            # 에러 발생 시 어떤 ID에서 문제가 생겼는지 명확히 출력
            logger.error(f"Sample {sample_id} failed: {e}")
            continue

    # 5. 전체 평가 실행
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
