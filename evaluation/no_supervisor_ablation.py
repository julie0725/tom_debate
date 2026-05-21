"""
supervisor_ablation.py
-----------------------
Supervisor 보정 유무 ablation study 
목적:
  하네스 엔지니어링 기법(supervisor correction)이 실제로 성능에 기여하는지 검증.
  full_system(보정 있음) vs no_supervisor(보정 없음) 비교.

조건:
  full_system   : 모든 에이전트 + 토론 + supervisor 보정 후 재추론
  no_supervisor : 모든 에이전트 + 토론 + supervisor 보정 없이 재추론

사용:
  python main.py --mode supervisor_ablation --dataset data/bigtom/bigtom.csv
"""

import copy
import json
import logging
from pathlib import Path

from user.ai_user import AIUser
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


ABLATION_CONDITIONS = [
    # TODO : 조건 정의에 맞게 유동적으로 채워주세요 !
    # {
    #     "name": "full_system",
    #     "description": "Full system with supervisor correction",
    #     "overrides": {
    #         "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
    #         "debate": {"use_debate": True},
    #         "supervisor": {"use_correction": True},
    #     },
    # },
    # {
    #     "name": "no_supervisor",
    #     "description": "No supervisor correction",
    #     "overrides": {
    #         "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
    #         "debate": {"use_debate": True},
    #         "supervisor": {"use_correction": False},
    #     },
    # },
]


class SupervisorAblationRunner:
    def __init__(
        self,
        base_config: dict,
        dataset_path: str,
        output_dir: str = "outputs/ablation_supervisor/",
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
