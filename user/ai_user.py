"""
ai_user.py
----------
AI User: 시스템의 진입점
- 시나리오와 질문을 받아 context file(ToMState) 생성
- 전역 메시지 풀에 publish
- 감독관 실행 후 최종 답변 수신
"""

import asyncio
import logging
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

from core.context_file import ToMState, ToMAnswers
from core.message_pool import MessagePool, get_message_pool, reset_message_pool
from supervisor.supervisor import Supervisor

logger = logging.getLogger(__name__)


class AIUser:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get("evaluation", {}).get("output_dir", "outputs/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        scenario: str,
        q1: str,
        q2: str,
        q3: str,
        dataset_id: str = None,
        ground_truth: ToMAnswers = None
    ) -> ToMState:
        """
        사용자 입력을 받아 전체 파이프라인 실행 후 결과 반환
        """
        # 1. Context file 생성
        state = ToMState(
            scenario=scenario,
            questions={"q1": q1, "q2": q2, "q3": q3},
            dataset_id=dataset_id,
            ground_truth=ground_truth
        )

        # 2. 메시지 풀 초기화 & publish
        reset_message_pool()
        pool = get_message_pool()
        pool.publish(state, publisher="ai_user")
        logger.info(f"[AIUser] Published context file | dataset_id={dataset_id}")

        # 3. 감독관 실행
        supervisor = Supervisor(pool=pool, config=self.config)
        final_state = asyncio.run(supervisor.run())

        # 4. 결과 저장 (jsonl)
        self._save_result(final_state)

        # 5. 디버깅용 MD 출력
        md_path = self.output_dir / f"{dataset_id or 'result'}.md"
        md_path.write_text(pool.dump_markdown(), encoding="utf-8")

        logger.info(f"[AIUser] Pipeline complete | status={final_state.status}")
        return final_state

    def _save_result(self, state: ToMState) -> None:
        """실험 결과를 jsonl에 append (논문 실험 재현용)"""
        log_path = self.output_dir / "results.jsonl"
        record = {
            "timestamp": datetime.now().isoformat(),
            "dataset_id": state.dataset_id,
            "debate_round": state.debate_round,
            "debate_triggered": state.debate_triggered,
            "majority_vote_applied": state.majority_vote_applied,
            "final_answer": asdict(state.final_answer),
            "ground_truth": asdict(state.ground_truth) if state.ground_truth else None,
            "agent_outputs": state.agent_outputs
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
