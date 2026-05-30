"""
max_rounds_ablation.py
----------------------
토론 최대 라운드 수 ablation study

목적:
  max_rounds(1 / 3 / 5)에 따른 성능 변화를 측정.
  Big ToM / Hi ToM 데이터셋 각각 결과 출력.

조건:
  rounds_1 : 최대 1라운드 토론
  rounds_3 : 최대 3라운드 토론 (기본값)
  rounds_5 : 최대 5라운드 토론

사용:
  python main.py --mode max_rounds_ablation
"""

import copy
import csv
import json
import logging
from pathlib import Path

from user.ai_user import AIUser
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


ABLATION_CONDITIONS = [
    {
        "name": "rounds_1",
        "description": "Max 1 debate round",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {"use_debate": True, "max_rounds": 1},
            "supervisor": {"use_correction": True},
        },
    },
    {
        "name": "rounds_3",
        "description": "Max 3 debate rounds (default)",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {"use_debate": True, "max_rounds": 3},
            "supervisor": {"use_correction": True},
        },
    },
    {
        "name": "rounds_5",
        "description": "Max 5 debate rounds",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {"use_debate": True, "max_rounds": 5},
            "supervisor": {"use_correction": True},
        },
    },
]


class MaxRoundsAblationRunner:
    def __init__(
        self,
        base_config: dict,
        bigtom_path: str = "data/bigtom/bigtom.csv",
        hitom_path: str = "data/hitom/Hi-ToM_data.json",
        output_dir: str = "outputs/ablation_max_rounds/",
        limit: int = None,
    ):
        self.base_config = base_config
        self.dataset_paths = {
            "BigToM": bigtom_path,
            "HiToM": hitom_path,
        }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit

    def run_all(self) -> dict:
        all_results = {}

        for dataset_name, dataset_path in self.dataset_paths.items():
            logger.info(f"\n[MaxRoundsAblation] Dataset: {dataset_name}")
            print(f"\n{'#'*60}")
            print(f"  Dataset: {dataset_name}")
            print(f"{'#'*60}")

            dataset_results = {}
            for condition in ABLATION_CONDITIONS:
                dataset_results[condition["name"]] = self._run_one(
                    condition, dataset_name, dataset_path
                )
            all_results[dataset_name] = dataset_results

        comparison_path = self.output_dir / "ablation_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        self._save_csv(all_results)
        self._print_comparison(all_results)
        return all_results

    def _run_one(self, condition: dict, dataset_name: str, dataset_path: str) -> dict:
        logger.info(f"\n[MaxRoundsAblation] Running: {condition['name']} on {dataset_name}")
        print(f"\n{'='*60}")
        print(f"  Condition : {condition['name']}")
        print(f"  Dataset   : {dataset_name}")
        print(f"  {condition['description']}")
        print(f"{'='*60}")

        condition_output_dir = self.output_dir / dataset_name.lower() / condition["name"]
        condition_output_dir.mkdir(parents=True, exist_ok=True)

        cfg = copy.deepcopy(self.base_config)
        for section, overrides in condition["overrides"].items():
            if section not in cfg:
                cfg[section] = {}
            cfg[section].update(overrides)

        if "evaluation" not in cfg:
            cfg["evaluation"] = {}
        cfg["evaluation"]["output_dir"] = str(condition_output_dir) + "/"

        ai_user = AIUser(config=cfg)
        try:
            ai_user.submit_from_dataset(dataset_path, limit=self.limit)
        except Exception as e:
            logger.error(f"[MaxRoundsAblation] {condition['name']} on {dataset_name} failed: {e}")
            return {"description": condition["description"], "summary": {}}

        results_file = cfg.get("evaluation", {}).get("results_file")
        if not results_file:
            return {"description": condition["description"], "summary": {}}

        results_filename = Path(results_file).name
        summary_filename = results_filename.replace("results_", "evaluation_").replace(".jsonl", ".json")

        evaluator = Evaluator(output_dir=str(condition_output_dir) + "/")
        summary = evaluator.evaluate_from_jsonl(
            results_file=results_filename,
            output_file=summary_filename,
        )
        return {"description": condition["description"], "summary": summary}

    def _print_comparison(self, all_results: dict) -> None:
        for dataset_name, dataset_results in all_results.items():
            print("\n" + "=" * 96)
            print(f"  MAX ROUNDS ABLATION — {dataset_name}")
            print("=" * 96)
            header = (
                f"  {'Condition':<12} {'Q1':>8} {'Q2':>8} {'Q3':>8} "
                f"{'Joint':>8} {'Conflicts':>12} {'Avg Rounds':>12} {'Avg(debated)':>14}"
            )
            print(header)
            print("-" * 96)

            for name, data in dataset_results.items():
                s = data.get("summary", {})
                total = s.get("total", 0) or 0
                trigger_rate = s.get("debate_trigger_rate", 0) or 0
                conflicts = int(round(trigger_rate * total))
                avg_rounds = s.get("avg_debate_rounds", 0) or 0
                among = s.get("avg_debate_rounds_among_debated")

                print(
                    f"  {name:<12} "
                    f"{(s.get('q1_belief_accuracy', 0) or 0):>8.2%} "
                    f"{(s.get('q2_desire_accuracy', 0) or 0):>8.2%} "
                    f"{(s.get('q3_action_accuracy', 0) or 0):>8.2%} "
                    f"{(s.get('joint_accuracy', 0) or 0):>8.2%} "
                    f"{f'{conflicts}/{total}':>12} "
                    f"{avg_rounds:>12.2f} "
                    f"{f'{among:.2f}' if among is not None else 'N/A':>14}"
                )

            print("=" * 96)
            print("  Q1/Q2/Q3 : 질문 유형별 정확도 (Belief / Desire / Action)")
            print("  Joint    : q1, q2, q3 모두 정답인 비율")
            print("  Conflicts: 초기 추론 불일치 → 토론 진입한 샘플 수 / 전체")
            print("  Avg Rounds: 샘플당 평균 토론 라운드 수 (전체 기준)")
            print("  Avg(debated): 토론 진입 샘플만의 평균 라운드 수")
            print("=" * 80 + "\n")

    def _save_csv(self, all_results: dict) -> None:
        csv_path = self.output_dir / "ablation_comparison.csv"
        fieldnames = [
            "dataset", "condition", "q1_accuracy", "q2_accuracy",
            "q3_accuracy", "joint_accuracy", "conflicts", "total", "avg_rounds", "avg_rounds_debated"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for dataset_name, dataset_results in all_results.items():
                for name, data in dataset_results.items():
                    s = data.get("summary", {})
                    total = s.get("total", 0) or 0
                    trigger_rate = s.get("debate_trigger_rate", 0) or 0
                    among = s.get("avg_debate_rounds_among_debated")
                    writer.writerow({
                        "dataset": dataset_name,
                        "condition": name,
                        "q1_accuracy": round(s.get("q1_belief_accuracy", 0) or 0, 4),
                        "q2_accuracy": round(s.get("q2_desire_accuracy", 0) or 0, 4),
                        "q3_accuracy": round(s.get("q3_action_accuracy", 0) or 0, 4),
                        "joint_accuracy": round(s.get("joint_accuracy", 0) or 0, 4),
                        "conflicts": int(round(trigger_rate * total)),
                        "total": total,
                        "avg_rounds": round(s.get("avg_debate_rounds", 0) or 0, 2),
                        "avg_rounds_debated": round(among, 2) if among is not None else "",
                    })
        logger.info(f"[MaxRoundsAblation] CSV saved: {csv_path}")
