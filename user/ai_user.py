"""
ai_user.py
----------
AI User: 시스템의 진입점
- ToMTask를 받아 context file(ToMState) 생성
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
from core.message_pool import get_message_pool, reset_message_pool
from core.tom_task import ToMTask
from supervisor.supervisor import Supervisor

logger = logging.getLogger(__name__)


class AIUser:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get("evaluation", {}).get("output_dir", "outputs/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def submit(self, task: ToMTask) -> ToMState:
        """Run the full pipeline for a single ToMTask and return the final state."""
        # Build questions list from task (primary question + optional q2/q3 from metadata)
        questions = [{"id": "q1", "text": task.question}]
        if task.metadata.get("q2"):
            questions.append({"id": "q2", "text": task.metadata["q2"]})
        if task.metadata.get("q3"):
            questions.append({"id": "q3", "text": task.metadata["q3"]})

        # Build ground_truth as dataset-independent ToMAnswers
        ground_truth = None
        if task.gold_answer is not None:
            gt_answers = [{"id": "q1", "value": task.gold_answer}]
            if task.metadata.get("gold_q2"):
                gt_answers.append({"id": "q2", "value": task.metadata["gold_q2"]})
            if task.metadata.get("gold_q3"):
                gt_answers.append({"id": "q3", "value": task.metadata["gold_q3"]})
            ground_truth = ToMAnswers(answers=gt_answers)

        state = ToMState(
            scenario=task.context,
            questions=questions,
            dataset_id=task.dataset_id,
            ground_truth=ground_truth,
        )

        reset_message_pool()
        pool = get_message_pool()
        pool.publish(state, publisher="ai_user")
        logger.info(f"[AIUser] Published context file | dataset_id={task.dataset_id}")

        supervisor = Supervisor(pool=pool, config=self.config)
        final_state = asyncio.run(supervisor.run())

        self._save_result(final_state)

        md_path = self.output_dir / f"{task.dataset_id or 'result'}.md"
        md_path.write_text(pool.dump_markdown(), encoding="utf-8")

        logger.info(f"[AIUser] Pipeline complete | status={final_state.status}")
        return final_state

    def _save_result(self, state: ToMState) -> None:
        """Append result to the per-dataset JSONL file."""
        results_file = self.config.get("evaluation", {}).get("results_file", "results.jsonl")
        log_path = self.output_dir / results_file
        record = {
            "timestamp": datetime.now().isoformat(),
            "dataset_id": state.dataset_id,
            "debate_round": state.debate_round,
            "debate_triggered": state.debate_triggered,
            "majority_vote_applied": state.majority_vote_applied,
            "final_answer": asdict(state.final_answer),
            "ground_truth": asdict(state.ground_truth) if state.ground_truth else None,
            "agent_outputs": state.agent_outputs,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
