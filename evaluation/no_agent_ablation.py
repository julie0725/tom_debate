"""
no_agent_ablation.py
--------------------
"에이전트 하나씩 제거" ablation study 실험 러너

목적:
  3개 에이전트(Semantic / Ego / Observer) 중 하나씩 제거했을 때
  각 조건에서의 정확도를 측정.
  full_system 대비 비교는 팀원 결과를 받아 발표 자료에서 수동으로 진행.

에이전트 매핑:
  Agent1 = Semantic Agent (맥락 분석, 진실/거짓 판단)
  Agent2 = Ego Agent      (인물 본인 관점, belief state 추적)
  Agent3 = Observer Agent (타인 관점, 고차원 ToM)

본 파일은 3개 조건만 실행: no_semantic / no_ego / no_observer

사용:
  python main.py --mode no_agent_ablation --dataset data/bigtom/bigtom.csv
"""

import copy
import json
import logging
from pathlib import Path

from user.ai_user import AIUser
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


# 실험 조건 정의
# tiebreak_agent: 다수결 1:1 동점 시 우선 채택할 에이전트 번호 (가장 고차)
ABLATION_CONDITIONS = [
    {
        "name": "no_semantic",
        "description": "Without Semantic agent: Ego + Observer debate only",
        "overrides": {
            "agents": {"use_agent1": False, "use_agent2": True, "use_agent3": True},
            "debate": {"use_debate": True, "tiebreak_agent": 3}
        }
    },
    {
        "name": "no_ego",
        "description": "Without Ego agent: Semantic + Observer debate only",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": False, "use_agent3": True},
            "debate": {"use_debate": True, "tiebreak_agent": 3}
        }
    },
    {
        "name": "no_observer",
        "description": "Without Observer agent: Semantic + Ego debate only",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": False},
            "debate": {"use_debate": True, "tiebreak_agent": 2}
        }
    },
]


class AblationRunner:
    """
    조건별로 config를 변형해 같은 데이터셋을 반복 실행.

    핵심 흐름:
        1. 조건별 출력 폴더 생성
        2. base config를 deepcopy → overrides 적용
        3. AIUser 인스턴스 새로 생성 (use_agent* 플래그 따라 활성 에이전트만 생성)
        4. dataset_path를 어댑터에 넘겨 모든 샘플 자동 처리
        5. Evaluator로 정확도 집계
        6. 모든 조건 끝나면 비교 표 출력
    """

    def __init__(
        self,
        base_config: dict,
        dataset_path: str,
        output_dir: str = "outputs/ablation_hitom_temp0/",
        limit: int = None,
    ):
        self.base_config = base_config
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit

    def run_all(self) -> dict:
        all_results = {}

        for condition in ABLATION_CONDITIONS:
            logger.info(f"\n[Ablation] Running: {condition['name']}")
            print(f"\n{'='*60}")
            print(f"  Condition: {condition['name']}")
            print(f"  {condition['description']}")
            print(f"{'='*60}")

            condition_output_dir = self.output_dir / condition["name"]
            condition_output_dir.mkdir(parents=True, exist_ok=True)

            # config 복사 후 조건 적용
            cfg = copy.deepcopy(self.base_config)
            for section, overrides in condition["overrides"].items():
                if section not in cfg:
                    cfg[section] = {}
                cfg[section].update(overrides)

            # 결과 파일이 조건 폴더 안에 저장되도록 지정
            if "evaluation" not in cfg:
                cfg["evaluation"] = {}
            cfg["evaluation"]["output_dir"] = str(condition_output_dir) + "/"

            # AIUser 새 인스턴스 (활성 에이전트만 생성됨)
            ai_user = AIUser(config=cfg)

            # 데이터셋 전체 자동 처리 (어댑터가 CSV/JSON 형식 자동 감지)
            try:
                ai_user.submit_from_dataset(self.dataset_path, limit=self.limit)
            except Exception as e:
                logger.error(f"[Ablation] {condition['name']} dataset run failed: {e}")
                continue

            results_file = cfg.get("evaluation", {}).get("results_file")
            if not results_file:
                logger.error(f"[Ablation] No results_file produced for {condition['name']}")
                continue

            # results_file이 경로일 수도 있으니 파일명만 추출
            results_filename = Path(results_file).name
            summary_filename = results_filename.replace("results_", "evaluation_").replace(".jsonl", ".json")

            evaluator = Evaluator(output_dir=str(condition_output_dir) + "/")
            summary = evaluator.evaluate_from_jsonl(
                results_file=results_filename,
                output_file=summary_filename,
            )   

            all_results[condition["name"]] = {
                "description": condition["description"],
                "summary": summary,
            }

        # 전체 비교 결과 저장 + 콘솔 출력
        comparison_path = self.output_dir / "ablation_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        self._print_comparison(all_results)
        return all_results

    def _print_comparison(self, all_results: dict) -> None:
        """
        조건별 비교 표 출력.
        full_system과의 비교(하락폭)는 본 파일에서 다루지 않음.
        팀원의 full_system 결과를 받아 발표 자료에서 수동으로 비교.
        """
        print("\n" + "=" * 80)
        print("  ABLATION STUDY (no_semantic / no_ego / no_observer)")
        print("=" * 80)
        header = (
            f"  {'Condition':<14} {'Q1':>8} {'Q2':>8} {'Q3':>8} "
            f"{'Joint':>8} {'Conflicts':>12} {'Avg Rounds':>12}"
        )
        print(header)
        print("-" * 80)

        for name, data in all_results.items():
            s = data.get("summary", {})

            total = s.get("total", 0) or 0
            trigger_rate = s.get("debate_trigger_rate", 0) or 0
            conflicts = int(round(trigger_rate * total))
            conflicts_str = f"{conflicts}/{total}"

            avg_rounds = s.get("avg_debate_rounds", 0) or 0

            print(
                f"  {name:<14} "
                f"{(s.get('q1_belief_accuracy', 0) or 0):>8.2%} "
                f"{(s.get('q2_desire_accuracy', 0) or 0):>8.2%} "
                f"{(s.get('q3_action_accuracy', 0) or 0):>8.2%} "
                f"{(s.get('joint_accuracy', 0) or 0):>8.2%} "
                f"{conflicts_str:>12} "
                f"{avg_rounds:>12.2f}"
            )

        print("=" * 80)
        print("  Q1/Q2/Q3 : 질문 유형별 정확도 (Belief / Desire / Action)")
        print("  Joint    : q1, q2, q3 모두 정답인 비율 (가장 엄격)")
        print("  Conflicts: 초기 추론에서 의견 불일치 → 토론 진입한 샘플 수 / 전체")
        print("  Avg Rounds: 샘플당 평균 토론 라운드 수")
        print("=" * 80 + "\n")