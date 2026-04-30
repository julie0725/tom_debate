"""
agent1_context.py
-----------------
Agent 1: 맥락 Agent
시나리오 전체를 파악하여 인물의 목표에 따른 진실/거짓 판단
출력: character_goal, truth_judgment, tom_answers
"""

import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class Agent1Context(BaseAgent):
    def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 2000, provider: str = "openai", base_url: str = None):
        super().__init__(agent_id=1, model=model, max_tokens=max_tokens, provider=provider, base_url=base_url)

    # 수정 
    def reason(self, state_dict: dict, debate_context: dict = None) -> dict:
        user_prompt = self._build_user_prompt(state_dict)
        raw = self._call_llm(user_prompt)
        parsed = self._parse_json_response(raw)

        answer_block = parsed.get("answer", {})
        tom_answer = answer_block.get("response", "")

        # truth_judgment: {event_idx: "verdict (rationale)"} dict로 정리
        event_judgments_raw = parsed.get("event_judgments", [])
        truth_judgment = {}
        if isinstance(event_judgments_raw, list):
            for ev in event_judgments_raw:
                idx = str(ev.get("idx", "?"))
                verdict = ev.get("verdict", "?")
                rationale = ev.get("rationale", "")
                truth_judgment[idx] = f"{verdict} ({rationale})"
        elif isinstance(event_judgments_raw, dict):
            truth_judgment = event_judgments_raw

        return {
            "agent_id": 1,
            "character_goal": parsed.get("character_goal", parsed.get("character", "")),
            "truth_judgment": truth_judgment,
            "tom_answer": tom_answer,
            "update_log": [],
            "belief_state": None,
            "reasoning": answer_block.get("rationale", ""),
            "tom_answers": {
                "q1_belief": tom_answer,
                "q2_desire": answer_block.get("q2_desire", ""),
                "q3_action": answer_block.get("q3_action", ""),
            }
        }
