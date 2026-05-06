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

    def reason(self, state_dict: dict) -> dict:
        user_prompt = self._build_user_prompt(state_dict)
        raw = self._call_llm(user_prompt)
        parsed = self._parse_json_response(raw)

        answer_block = parsed.get("answer", {})
        tom_answers = self._build_tom_answers(parsed, state_dict)

        raw_sims = parsed.get("intermediate_simulations", [])
        update_log = {}
        if isinstance(raw_sims, list):
            for sim in raw_sims:
                agent = sim.get("simulated_agent", "?")
                order = sim.get("simulation_order", 0)
                belief = sim.get("derived_belief", "?")
                evidence = sim.get("evidence_events", [])
                key = f"{order}" if evidence else f"sim_{order}"
                if key not in update_log:
                    update_log[key] = {}
                update_log[key][agent] = {"goal": "위치 추적", "belief": belief}
        elif isinstance(raw_sims, dict):
            update_log = raw_sims

        return {
            "agent_id": 3,
            "character_goal": parsed.get("focal_character", ""),
            "truth_judgment": None,
            "update_log": update_log,
            "belief_state": parsed.get("higher_order_beliefs", []),
            "tom_answer": answer_block.get("response", ""),
            "reasoning": answer_block.get("rationale", ""),
            "tom_answers": tom_answers,
        }
