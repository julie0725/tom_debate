"""
agent2_character.py
-------------------
Agent 2: 인물 Agent
인물의 입장에서 belief state를 단계별로 추적
핵심: 해당 인물이 아는 정보만으로 추론
출력: character_goal, update_log, belief_state, tom_answers
"""

import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class Agent2Character(BaseAgent):
    def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 2000, provider: str = "openai", base_url: str = None):
        super().__init__(agent_id=2, model=model, max_tokens=max_tokens, provider=provider, base_url=base_url)

    def reason(self, state_dict: dict, debate_context: dict = None) -> dict:
        user_prompt = self._build_user_prompt(state_dict, debate_context)
        raw = self._call_llm(user_prompt)
        parsed = self._parse_json_response(raw)

        return {
            "agent_id": 2,
            "character_goal": parsed.get("character_goal", ""),
            "truth_judgment": None,
            "update_log": parsed.get("update_log", []),
            "belief_state": parsed.get("belief_state", ""),
            "reasoning": parsed.get("reasoning", ""),
            "tom_answers": {
                "q1_belief": parsed.get("tom_answers", {}).get("q1_belief", ""),
                "q2_desire": parsed.get("tom_answers", {}).get("q2_desire", ""),
                "q3_action": parsed.get("tom_answers", {}).get("q3_action", "")
            }
        }
