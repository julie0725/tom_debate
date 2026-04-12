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

    def reason(self, state_dict: dict, debate_context: dict = None) -> dict:
        """
        state_dict: ToMState dict
        debate_context: 토론 시 다른 에이전트 출력 포함
        반환: AgentOutput 형식 dict
        """
        user_prompt = self._build_user_prompt(state_dict)
        raw = self._call_llm(user_prompt)
        parsed = self._parse_json_response(raw)

        # 필수 필드 기본값 보장
        return {
            "agent_id": 1,
            "character_goal": parsed.get("character_goal", ""),
            "truth_judgment": parsed.get("event_judgments", []),   # ← truth_judgment → event_judgments
            "update_log": [],
            "belief_state": None,
            "reasoning": parsed.get("answer", {}).get("rationale", ""),  # ← reasoning 소스 변경
            "tom_answers": {
                "q1_belief": parsed.get("answer", {}).get("response", ""),   # ← 스키마에 맞게 변경
                "q2_desire": parsed.get("answer", {}).get("question_order", ""),
                "q3_action": parsed.get("answer", {}).get("question", "")
            }
        }
