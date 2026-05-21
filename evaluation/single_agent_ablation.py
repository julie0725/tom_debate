"""
single_agent_ablation.py
------------------------
"에이전트 2개 제거" ablation study

목적:
  3개 에이전트 중 하나만 남겼을 때 각 에이전트의 단독 성능 측정.

조건:
  agent1_only : Semantic Agent만 단독 추론
  agent2_only : Ego Agent만 단독 추론
  agent3_only : Observer Agent만 단독 추론

사용:
  python main.py --mode single_agent_ablation --dataset data/bigtom/bigtom.csv
"""

import copy
import json
import logging
from pathlib import Path

from user.ai_user import AIUser
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


ABLATION_CONDITIONS = [
    # TODO : 조건 정의 맞게 유동적으로 채우주세요! 
    # {
    #     "name": "agent1_only",
    #     "description": "Semantic Agent only",
    #     "overrides": {
    #         "agents": {"use_agent1": True, "use_agent2": False, "use_agent3": False},
    #         "debate": {"use_debate": False},
    #     },
    # },
]


class SingleAgentAblationRunner:
    def __init__(
        self,
        base_config: dict,
        dataset_path: str,
        output_dir: str = "outputs/ablation_single_agent/",
        limit: int = None,
    ):
        self.base_config = base_config
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit

    def run_all(self) -> dict:
        # TODO : 구현 필요
        raise NotImplementedError("구현 필요")
