"""
agent3_perspective.py
---------------------
Agent 3: 타인 관점 Agent (고차원 추론 전담)
"A는 B가 X를 알고 있다고 생각하는가?" 류의 2nd-order 이상 ToM 수행
Hi-ToM 기반 고차원 추론 담당
출력: character_goal, update_log, belief_state, tom_answers
"""

import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class Agent3Perspective(BaseAgent):
    def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 2000, provider: str = "openai", base_url: str = None):
        super().__init__(agent_id=3, model=model, max_tokens=max_tokens, provider=provider, base_url=base_url)

    def reason(self, state_dict: dict, debate_context: dict = None) -> dict:
        user_prompt = self._build_user_prompt(state_dict)
        raw = self._call_llm(user_prompt)
        parsed = self._parse_json_response(raw)

        return {
            "agent_id": 3,
            "character_goal": parsed.get("focal_character", ""),              # ← character_goal 없음, focal_character로 대체
            "truth_judgment": None,
            "update_log": parsed.get("intermediate_simulations", []),         # ← intermediate_simulations
            "belief_state": parsed.get("higher_order_beliefs", []),           # ← higher_order_beliefs
            "reasoning": parsed.get("answer", {}).get("rationale", ""),       # ← answer.rationale
            "tom_answers": {
                "q1_belief": parsed.get("answer", {}).get("response", ""),
                "q2_desire": parsed.get("answer", {}).get("question_order", ""),
                "q3_action": parsed.get("answer", {}).get("question", "")
            }
        }
