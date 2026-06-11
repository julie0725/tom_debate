"""
user/ablation_ai_user.py
------------------------
E / F / G 조건 라우팅 AIUser 확장.
기존 ai_user.py 수정 없음.

- E (backend="camel")        : CamelSupervisor 사용
- F (backend="camel_baseline"): Supervisor 전체 우회, CamelBaseline 직접 실행
- G (use_single_agent=True)   : SingleAgentSupervisor 사용
"""
import asyncio
import logging
from dataclasses import asdict

from core.context_file import ToMState, ToMAnswers
from core.message_pool import get_message_pool, reset_message_pool
from core.tom_task import ToMTask
from user.ai_user import AIUser

logger = logging.getLogger(__name__)


class AblationAIUser(AIUser):

    # ── 내부 실행자 오버라이드 ────────────────────────────────────────────────

    def _submit(self, task: ToMTask) -> ToMState:
        backend = self.config.get("system", {}).get("backend", "")
        use_single = self.config.get("agents", {}).get("use_single_agent", False)

        # F: Supervisor 완전 우회
        if backend == "camel_baseline":
            return self._submit_camel_baseline(task)

        # E / G: 기존 _submit() 흐름 유지, Supervisor만 교체
        questions = [{"id": "q1", "text": task.question}]
        if task.metadata.get("q2"):
            questions.append({"id": "q2", "text": task.metadata["q2"]})
        if task.metadata.get("q3"):
            questions.append({"id": "q3", "text": task.metadata["q3"]})

        ground_truth = None
        if task.gold_answer is not None:
            gt_answers = [{"id": "q1", "value": task.gold_answer}]
            if task.metadata.get("gold_q2"):
                gt_answers.append({"id": "q2", "value": task.metadata["gold_q2"]})
            if task.metadata.get("gold_q3"):
                gt_answers.append({"id": "q3", "value": task.metadata["gold_q3"]})
            ground_truth = ToMAnswers(answers=gt_answers)

        common_state = self.extractor.extract(task)

        state = ToMState(
            scenario=task.context,
            questions=questions,
            dataset_id=task.dataset_id,
            ground_truth=ground_truth,
            reasoning_type=common_state.reasoning_type,
            characters=[c.name for c in common_state.characters],
            common_state=common_state.to_dict(),
        )

        reset_message_pool()
        pool = get_message_pool()
        pool.publish(state, publisher="ai_user")
        logger.info(f"[AblationAIUser] Published | id={task.dataset_id}")

        supervisor = self._make_supervisor(pool, backend, use_single)
        final_state = asyncio.run(supervisor.run())

        # 기존 _save_result() 그대로 호출 (수정 없음)
        self._save_result(final_state)

        md_path = self.output_dir / f"{task.dataset_id or 'result'}.md"
        md_path.write_text(pool.dump_markdown(), encoding="utf-8")

        logger.info(f"[AblationAIUser] Complete | status={final_state.status}")
        return final_state

    # ── Supervisor 선택 ───────────────────────────────────────────────────────

    def _make_supervisor(self, pool, backend: str, use_single: bool):
        from supervisor.ablation_supervisor import CamelSupervisor, SingleAgentSupervisor
        from supervisor.supervisor import Supervisor

        if backend == "camel":
            return CamelSupervisor(pool=pool, config=self.config)
        if use_single:
            return SingleAgentSupervisor(pool=pool, config=self.config)
        return Supervisor(pool=pool, config=self.config)

    # ── F 조건 전용 실행자 ────────────────────────────────────────────────────

    def _submit_camel_baseline(self, task: ToMTask) -> ToMState:
        """F 조건: CamelBaseline 단독 실행 후 기존 _save_result()로 저장."""
        from core.camel_baseline import CamelBaseline

        questions = [{"id": "q1", "text": task.question}]
        if task.metadata.get("q2"):
            questions.append({"id": "q2", "text": task.metadata["q2"]})
        if task.metadata.get("q3"):
            questions.append({"id": "q3", "text": task.metadata["q3"]})

        ground_truth = None
        if task.gold_answer is not None:
            gt_answers = [{"id": "q1", "value": task.gold_answer}]
            if task.metadata.get("gold_q2"):
                gt_answers.append({"id": "q2", "value": task.metadata["gold_q2"]})
            if task.metadata.get("gold_q3"):
                gt_answers.append({"id": "q3", "value": task.metadata["gold_q3"]})
            ground_truth = ToMAnswers(answers=gt_answers)

        sys_cfg = self.config.get("system", {})
        result = CamelBaseline(
            model=sys_cfg.get("model", "gpt-4o-mini"),
            max_tokens=sys_cfg.get("max_tokens", 2000),
        ).run({"scenario": task.context, "questions": questions})

        # ToMState에 최소 필드만 채움 (debate 없음)
        state = ToMState(
            scenario=task.context,
            questions=questions,
            dataset_id=task.dataset_id or "camel_baseline",
            ground_truth=ground_truth,
            reasoning_type="camel_baseline",
            characters=[],
            common_state={},
        )
        state.final_answer = ToMAnswers(answers=result.get("tom_answers", []))
        state.status = "completed"

        # 기존 _save_result() 그대로 호출
        self._save_result(state)
        return state
