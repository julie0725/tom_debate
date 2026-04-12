"""
debate.py
---------
Debate Loop 관리
- 라운드별 Agent 재추론 (다른 Agent 출력 공유)
- 만장일치 체크
- max_rounds 초과 시 다수결
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
        self.tiebreak_agent = tiebreak_agent  # 동점 시 우선 Agent (기본: Agent 3)

    async def run_debate(self, pool: MessagePool, supervisor_call_fn) -> ToMAnswers:
        """
        토론 루프 실행
        - 각 라운드: 다른 에이전트 출력 공유 → 재추론 → 감독관 재판단
        - 만장일치 or max_rounds 도달 시 종료
        """
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"[Debate] Round {round_num} / {self.max_rounds}")
            pool.update_debate_round(round_num)
            state = pool.get_state()

            # 다른 Agent 출력을 공유하며 재추론
            updated_outputs = await self._re_reason_all(state)
            for agent_id, output in updated_outputs.items():
                pool.update_agent_output(agent_id, output)

            # 감독관 재판단
            state = pool.get_state()
            result = supervisor_call_fn(state, debate_round=round_num)
            logger.info(f"[Debate] Round {round_num} agreement={result.get('agreement')}")

            if result.get("agreement"):
                logger.info(f"[Debate] Consensus reached at round {round_num}")
                return self._extract_answer(result)

        # max_rounds 초과 → 다수결
        logger.info(f"[Debate] Max rounds reached. Applying majority vote.")
        state = pool.get_state()
        state.majority_vote_applied = True
        return self._majority_vote(state)

    async def _re_reason_all(self, state: ToMState) -> dict:
        """모든 Agent가 다른 Agent 출력을 보고 재추론"""
        state_dict = asdict(state)

        async def re_reason_agent(agent_id, agent):
            # 다른 에이전트들의 출력을 debate_context로 전달
            other_outputs = {
                f"agent{aid}": out
                for aid, out in state.agent_outputs.items()
                if aid != f"agent{agent_id}" and out is not None
            }
            debate_context = {
                "round": state.debate_round,
                "other_outputs": other_outputs
            }
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None, agent.reason, state_dict, debate_context
            )
            logger.info(f"[Debate] Agent{agent_id} re-reasoned")
            return agent_id, output

        tasks = [re_reason_agent(aid, agent) for aid, agent in self.agents.items()]
        results = await asyncio.gather(*tasks)
        return {aid: output for aid, output in results}

    def _majority_vote(self, state: ToMState) -> ToMAnswers:
        """
        3개 질문 각각에 대해 다수결
        동점 시 tiebreak_agent(기본: Agent 3) 답변 채택
        """
        outputs = [
            state.agent_outputs.get(f"agent{i}")
            for i in [1, 2, 3]
            if state.agent_outputs.get(f"agent{i}") is not None
        ]

        def vote_for_question(q_key: str) -> str:
            answers = []
            for out in outputs:
                if out and out.get("tom_answers"):
                    answers.append(out["tom_answers"].get(q_key, ""))
            if not answers:
                return ""
            counter = Counter(answers)
            top = counter.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                return top[0][0]
            # 동점 → tiebreak_agent 답변 채택
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
