"""
debate.py
---------
Debate Loop 관리
- 라운드별 Agent 재추론 (MessagePool 통해 debate_context 공유 - 직접 전달 금지)
- 만장일치 체크
- max_rounds 초과 시: 감독관 오류 분석 → context file 수정 → 처음부터 재추론 → (불일치 시) 다수결
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

    async def run_debate(
        self,
        pool: MessagePool,
        supervisor_call_fn,
        supervisor_correction_fn=None
    ) -> ToMAnswers:
        """
        토론 루프 실행
        - 각 라운드: MessagePool에 debate_context 저장 → 재추론 → 감독관 재판단
        - 만장일치 → 종료
        - max_rounds 초과 → 감독관 오류 분석 → context file 수정 → 처음부터 재추론
          - 재추론 후 일치 → 종료
          - 재추론 후 불일치 → 다수결 (최후 수단)
        """
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"[Debate] Round {round_num} / {self.max_rounds}")
            pool.update_debate_round(round_num)

            # MessagePool에 debate_context 저장 후 재추론
            await self._re_reason_all(pool)

            # 감독관 재판단
            state = pool.get_state()
            result = supervisor_call_fn(state, debate_round=round_num)
            logger.info(f"[Debate] Round {round_num} agreement={result.get('agreement')}")

            if result.get("agreement"):
                logger.info(f"[Debate] Consensus reached at round {round_num}")
                return self._extract_answer(result)

        # max_rounds 초과 → 감독관 오류 분석 및 context file 수정
        logger.info("[Debate] Max rounds reached. Supervisor analyzing errors...")

        if supervisor_correction_fn:
            state = pool.get_state()
            correction = supervisor_correction_fn(state)
            pool.update_supervisor_correction(correction)

            # debate_context 초기화 (처음부터 재추론이므로 이전 토론 컨텍스트 제거)
            pool.update_debate_context({})
            logger.info("[Debate] Context file corrected. Re-reasoning from scratch...")

            # 처음부터 재추론 (supervisor_correction이 state_dict에 반영된 상태)
            await self._re_reason_fresh(pool)

            # 최종 판단
            state = pool.get_state()
            final_result = supervisor_call_fn(state, debate_round=self.max_rounds + 1)

            if final_result.get("agreement"):
                logger.info("[Debate] Consensus reached after supervisor correction.")
                return self._extract_answer(final_result)

        # 최후 수단: 다수결
        logger.info("[Debate] Applying majority vote as last resort.")
        state = pool.get_state()
        state.majority_vote_applied = True
        return self._majority_vote(state)

    async def _re_reason_all(self, pool: MessagePool) -> None:
        """
        토론 라운드 재추론
        - 각 에이전트별 debate_context(다른 에이전트 출력)를 MessagePool에 저장
        - 에이전트는 pool.get_state() → state_dict를 통해 debate_context 읽음
        - 직접 에이전트간 전달 없음
        """
        state = pool.get_state()

        # 에이전트별 debate_context 빌드 후 MessagePool 업데이트
        debate_context = {"round": state.debate_round}
        for agent_id in self.agents:
            other_outputs = {
                aid: out
                for aid, out in state.agent_outputs.items()
                if aid != f"agent{agent_id}" and out is not None
            }
            debate_context[f"agent{agent_id}"] = {"other_outputs": other_outputs}

        pool.update_debate_context(debate_context)

        # 업데이트된 state_dict로 병렬 재추론
        state = pool.get_state()
        state_dict = asdict(state)

        async def re_reason_agent(agent_id, agent):
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, agent.reason, state_dict, None)
            logger.info(f"[Debate] Agent{agent_id} re-reasoned via MessagePool")
            return agent_id, output

        tasks = [re_reason_agent(aid, agent) for aid, agent in self.agents.items()]
        results = await asyncio.gather(*tasks)

        for agent_id, output in results:
            pool.update_agent_output(agent_id, output)

    async def _re_reason_fresh(self, pool: MessagePool) -> None:
        """
        감독관 수정 후 처음부터 재추론
        - debate_context 없이 supervisor_correction만 반영
        - 에이전트는 state_dict의 supervisor_correction을 읽어 재추론
        """
        state = pool.get_state()
        state_dict = asdict(state)

        async def re_reason_agent(agent_id, agent):
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, agent.reason, state_dict, None)
            logger.info(f"[Debate] Agent{agent_id} re-reasoned fresh after correction")
            return agent_id, output

        tasks = [re_reason_agent(aid, agent) for aid, agent in self.agents.items()]
        results = await asyncio.gather(*tasks)

        for agent_id, output in results:
            pool.update_agent_output(agent_id, output)

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
