"""
debate.py
"""
import asyncio
import logging
import re
from collections import Counter
from dataclasses import asdict

from core.context_file import ToMState, ToMAnswers, get_answer_value
from core.message_pool import MessagePool

logger = logging.getLogger(__name__)


def _extract_choice_letter(text: str) -> str:
    if not text:
        return ""
    m = re.match(r'^([A-Z])(?:\.|\s|$)', text.strip())
    return m.group(1) if m else text.strip()


def _parse_final_answer(fa) -> dict:
    """Convert supervisor final_answer (list or legacy dict) to {q_id: value} map."""
    if isinstance(fa, list):
        return {a.get("id"): (a.get("value") or "") for a in fa if a.get("id")}
    if isinstance(fa, dict):
        result = {}
        for k, v in fa.items():
            if v:
                qid = k.replace("_belief", "").replace("_desire", "").replace("_action", "")
                result[qid] = v
        return result
    return {}


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
        run_logger=None
    ) -> ToMAnswers:

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"[Debate] Round {round_num} / {self.max_rounds}")
            pool.update_debate_round(round_num)

            state = pool.get_state()
            outputs_before = dict(state.agent_outputs)

            await self._re_reason_all(pool)

            state = pool.get_state()
            result = supervisor_call_fn(state, debate_round=round_num)
            logger.info(f"[Debate] Round {round_num} agreement={result.get('agreement')}")

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
                return self._extract_answer(result, state)

        logger.info("[Debate] Max rounds reached. Supervisor analyzing errors...")

        if supervisor_correction_fn:
            state = pool.get_state()
            correction = supervisor_correction_fn(state)
            pool.update_supervisor_correction(correction)

            if run_logger:
                run_logger.log_supervisor_correction(correction)

            pool.update_debate_context({})
            logger.info("[Debate] Re-reasoning from scratch...")

            await self._re_reason_fresh(pool)

            state = pool.get_state()
            final_result = supervisor_call_fn(state, debate_round=self.max_rounds + 1)

            if run_logger:
                run_logger.log_agent_outputs(state.agent_outputs, label="fresh_reInfer")
                run_logger.log_context_file(asdict(state), label="after_correction_reInfer")

            if final_result.get("agreement"):
                logger.info("[Debate] Consensus reached after correction.")
                return self._extract_answer(final_result, state)

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
            output = await loop.run_in_executor(None, agent.reason, state_dict)
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
            output = await loop.run_in_executor(None, agent.reason, state_dict)
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

        # Collect question IDs present in agent outputs
        q_ids: list[str] = []
        for out in outputs:
            if out:
                for a in (out.get("tom_answers") or []):
                    qid = a.get("id")
                    if qid and qid not in q_ids:
                        q_ids.append(qid)
        if not q_ids:
            q_ids = ["q1"]

        def vote_for_question(q_id: str) -> str:
            votes = [
                _extract_choice_letter(get_answer_value(out.get("tom_answers"), q_id))
                for out in outputs
                if out and out.get("tom_answers") is not None
            ]
            if not votes:
                return ""
            counter = Counter(votes)
            top = counter.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                return top[0][0]
            tiebreak = state.agent_outputs.get(f"agent{self.tiebreak_agent}")
            if tiebreak:
                tb_val = get_answer_value(tiebreak.get("tom_answers"), q_id)
                return _extract_choice_letter(tb_val) or top[0][0]
            return top[0][0]

        result = ToMAnswers(answers=[{"id": qid, "value": vote_for_question(qid)} for qid in q_ids])
        logger.info(f"[Debate] Majority vote result: {result}")
        return result

    def _extract_answer(self, supervisor_result: dict, state: ToMState = None) -> ToMAnswers:
        ans_map = _parse_final_answer(supervisor_result.get("final_answer"))

        if not ans_map.get("q1") and state is not None:
            for output in (asdict(state).get("agent_outputs") or {}).values():
                if not output:
                    continue
                tom_ans = output.get("tom_answers")
                q1_val = _extract_choice_letter(get_answer_value(tom_ans, "q1"))
                if q1_val:
                    # Collect all question answers from this agent as fallback
                    if isinstance(tom_ans, list):
                        for a in tom_ans:
                            qid = a.get("id")
                            if qid and not ans_map.get(qid):
                                ans_map[qid] = _extract_choice_letter(a.get("value", ""))
                    break

        return ToMAnswers(answers=[{"id": k, "value": v} for k, v in ans_map.items() if v])
