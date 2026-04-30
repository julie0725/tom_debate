"""
debate.py
"""
import asyncio
import logging
from collections import Counter
from dataclasses import asdict

from core.context_file import ToMState, ToMAnswers
from core.message_pool import MessagePool

logger = logging.getLogger(__name__)


class DebateManager:
    def __init__(self, agents: dict, max_rounds: int, tiebreak_agent: int = 3):
        self.agents = agents
        self.max_rounds = max_rounds
        self.tiebreak_agent = tiebreak_agent

    async def run_debate(
        self,
        pool: MessagePool,
        supervisor_call_fn,
        supervisor_correction_fn=None,
        run_logger=None    # ← 추가된 파라미터
    ) -> ToMAnswers:

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"[Debate] Round {round_num} / {self.max_rounds}")
            pool.update_debate_round(round_num)

            state = pool.get_state()
            outputs_before = dict(state.agent_outputs)  # ← 재추론 전 저장

            await self._re_reason_all(pool)

            state = pool.get_state()
            result = supervisor_call_fn(state, debate_round=round_num)
            logger.info(f"[Debate] Round {round_num} agreement={result.get('agreement')}")

            # ↓ 라운드 로그 (추가)
            if run_logger:
                run_logger.log_debate_round(
                    round_num=round_num,
                    debate_context=state.debate_context,
                    agent_outputs_before=outputs_before,
                    agent_outputs_after=dict(state.agent_outputs),
                    supervisor_result=result
                )
                run_logger.log_agent_outputs(state.agent_outputs, label=f"debate_round_{round_num:02d}")
                run_logger.log_context_file(asdict(state), label=f"debate_round_{round_num:02d}")

            if result.get("agreement"):
                logger.info(f"[Debate] Consensus reached at round {round_num}")
                return self._extract_answer(result)

        # max_rounds 초과
        logger.info("[Debate] Max rounds reached. Supervisor analyzing errors...")

        if supervisor_correction_fn:
            state = pool.get_state()
            correction = supervisor_correction_fn(state)
            pool.update_supervisor_correction(correction)

            if run_logger:
                run_logger.log_supervisor_correction(correction)  # ← 추가

            pool.update_debate_context({})
            logger.info("[Debate] Re-reasoning from scratch...")

            await self._re_reason_fresh(pool)

            state = pool.get_state()
            final_result = supervisor_call_fn(state, debate_round=self.max_rounds + 1)

            # ↓ fresh reInfer 로그 (추가)
            if run_logger:
                run_logger.log_agent_outputs(state.agent_outputs, label="fresh_reInfer")
                run_logger.log_context_file(asdict(state), label="after_correction_reInfer")

            if final_result.get("agreement"):
                logger.info("[Debate] Consensus reached after correction.")
                return self._extract_answer(final_result)

        logger.info("[Debate] Applying majority vote.")
        state = pool.get_state()
        state.majority_vote_applied = True
        return self._majority_vote(state)

    async def _re_reason_all(self, pool: MessagePool) -> None:
        state = pool.get_state()
        debate_context = {"round": state.debate_round}
        for agent_id in self.agents:
            other_outputs = {
                aid: out
                for aid, out in state.agent_outputs.items()
                if aid != f"agent{agent_id}" and out is not None
            }
            debate_context[f"agent{agent_id}"] = {"other_outputs": other_outputs}

        pool.update_debate_context(debate_context)

        state = pool.get_state()
        state_dict = asdict(state)

        async def re_reason_agent(agent_id, agent):
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, agent.reason, state_dict, None)
            return agent_id, output

        tasks = [re_reason_agent(aid, agent) for aid, agent in self.agents.items()]
        results = await asyncio.gather(*tasks)
        for agent_id, output in results:
            pool.update_agent_output(agent_id, output)

    async def _re_reason_fresh(self, pool: MessagePool) -> None:
        state = pool.get_state()
        state_dict = asdict(state)

        async def re_reason_agent(agent_id, agent):
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, agent.reason, state_dict, None)
            return agent_id, output

        tasks = [re_reason_agent(aid, agent) for aid, agent in self.agents.items()]
        results = await asyncio.gather(*tasks)
        for agent_id, output in results:
            pool.update_agent_output(agent_id, output)

    def _majority_vote(self, state: ToMState) -> ToMAnswers:
        outputs = [
            state.agent_outputs.get(f"agent{i}")
            for i in [1, 2, 3]
            if state.agent_outputs.get(f"agent{i}") is not None
        ]

        def vote_for_question(q_key: str) -> str:
            answers = [
                out["tom_answers"].get(q_key, "")
                for out in outputs
                if out and out.get("tom_answers")
            ]
            if not answers:
                return ""
            counter = Counter(answers)
            top = counter.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                return top[0][0]
            tiebreak_out = state.agent_outputs.get(f"agent{self.tiebreak_agent}")
            if tiebreak_out and tiebreak_out.get("tom_answers"):
                return tiebreak_out["tom_answers"].get(q_key, top[0][0])
            return top[0][0]

        result = ToMAnswers(
            q1_belief=vote_for_question("q1_belief"),
            q2_desire=vote_for_question("q2_desire"),
            q3_action=vote_for_question("q3_action")
        )
        logger.info(f"[Debate] Majority vote result: {result}")
        return result

    def _extract_answer(self, supervisor_result: dict) -> ToMAnswers:
        fa = supervisor_result.get("final_answer", {})
        return ToMAnswers(
            q1_belief=fa.get("q1_belief"),
            q2_desire=fa.get("q2_desire"),
            q3_action=fa.get("q3_action")
        )