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

    # 교체
    def reason(self, state_dict: dict, debate_context: dict = None) -> dict:
        user_prompt = self._build_user_prompt(state_dict)
        raw = self._call_llm(user_prompt)
        parsed = self._parse_json_response(raw)

        answer_block = parsed.get("answer", {})
        tom_answer = answer_block.get("response", "")

        # update_log: {event_idx: {character: {belief_state: ...}}} 형식
        raw_log = parsed.get("belief_update_log", parsed.get("update_log", []))
        update_log = {}
        if isinstance(raw_log, list):
            for ev in raw_log:
                idx = str(ev.get("idx", "?"))
                character_updates = {}
                for upd in ev.get("updates", []):
                    prop = upd.get("proposition", "unknown")
                    character_updates[prop] = {
                        "belief_state": f"{upd.get('before', '?')} → {upd.get('after', '?')}",
                        "confidence": upd.get("confidence", ""),
                        "rationale": upd.get("rationale", ""),
                    }
                update_log[idx] = {"character": character_updates}
        elif isinstance(raw_log, dict):
            update_log = raw_log

        return {
            "agent_id": 2,
            "character_goal": parsed.get("character_goal", parsed.get("character", "")),
            "truth_judgment": None,
            "update_log": update_log,
            "belief_state": parsed.get("final_belief_state", []),
            "tom_answer": tom_answer,
            "reasoning": answer_block.get("rationale", ""),
            "tom_answers": {
                "q1_belief": tom_answer,
                "q2_desire": answer_block.get("q2_desire", ""),
                "q3_action": answer_block.get("q3_action", ""),
            }
        }
