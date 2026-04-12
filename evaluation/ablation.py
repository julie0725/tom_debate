"""
ablation.py
-----------
Ablation study 실험 러너
config.yaml의 플래그를 바꿔가며 자동으로 실험 조건 변경
논문 Table 재현용
"""

import copy
import json
import logging
from pathlib import Path

from core.message_pool import reset_message_pool
from user.ai_user import AIUser
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


# 실험 조건 정의
# 논문에서 비교할 ablation 설정들
ABLATION_CONDITIONS = [
    {
        "name": "full_system",
        "description": "Full system: all agents + debate",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {"use_debate": True}
        }
    },
    {
        "name": "no_debate",
        "description": "No debate: all agents, no debate",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {"use_debate": False}
        }
    },
    {
        "name": "agent1_only",
        "description": "Single agent: agent1 only",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": False, "use_agent3": False},
            "debate": {"use_debate": False}
        }
    },
    {
        "name": "agent2_only",
        "description": "Single agent: agent2 only",
        "overrides": {
            "agents": {"use_agent1": False, "use_agent2": True, "use_agent3": False},
            "debate": {"use_debate": False}
        }
    },
    {
        "name": "agent3_only",
        "description": "Single agent: agent3 only",
        "overrides": {
            "agents": {"use_agent1": False, "use_agent2": False, "use_agent3": True},
            "debate": {"use_debate": False}
        }
    },
    {
        "name": "no_agent3",
        "description": "No higher-order agent: agent1 + agent2 only",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": False},
            "debate": {"use_debate": True}
        }
    }
]


class AblationRunner:
    def __init__(self, base_config: dict, dataset: list, output_dir: str = "outputs/ablation/"):
        """
        base_config: config.yaml 로드한 dict
        dataset: [{"scenario": ..., "q1": ..., "q2": ..., "q3": ..., "ground_truth": {...}, "id": ...}, ...]
        """
        self.base_config = base_config
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> dict:
        """모든 ablation 조건 실행"""
        all_results = {}

        for condition in ABLATION_CONDITIONS:
            logger.info(f"\n[Ablation] Running: {condition['name']}")
            print(f"\n{'='*50}")
            print(f"  Condition: {condition['name']}")
            print(f"  {condition['description']}")
            print(f"{'='*50}")

            condition_output_dir = self.output_dir / condition["name"]
            condition_output_dir.mkdir(parents=True, exist_ok=True)

            # config 복사 후 조건 적용
            cfg = copy.deepcopy(self.base_config)
            for section, overrides in condition["overrides"].items():
                cfg[section].update(overrides)
            cfg["evaluation"]["output_dir"] = str(condition_output_dir) + "/"

            # 실험 실행
            ai_user = AIUser(config=cfg)
            evaluator = Evaluator(output_dir=str(condition_output_dir) + "/")

            for sample in self.dataset:
                reset_message_pool()
                from core.context_file import ToMAnswers
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
                    logger.error(f"[Ablation] Sample {sample.get('id')} failed: {e}")

            # 조건별 평가
            summary = evaluator.evaluate_from_jsonl()
            all_results[condition["name"]] = {
                "description": condition["description"],
                "summary": summary
            }

        # 전체 비교 저장
        comparison_path = self.output_dir / "ablation_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        self._print_comparison(all_results)
        return all_results

    def _print_comparison(self, all_results: dict) -> None:
        print("\n" + "="*70)
        print("  ABLATION STUDY COMPARISON")
        print("="*70)
        print(f"  {'Condition':<20} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Joint':>8}")
        print("-"*70)
        for name, data in all_results.items():
            s = data.get("summary", {})
            print(
                f"  {name:<20} "
                f"{s.get('q1_belief_accuracy', 0):>8.2%} "
                f"{s.get('q2_desire_accuracy', 0):>8.2%} "
                f"{s.get('q3_action_accuracy', 0):>8.2%} "
                f"{s.get('joint_accuracy', 0):>8.2%}"
            )
        print("="*70 + "\n")
