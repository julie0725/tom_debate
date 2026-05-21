"""
base_agent.py
-------------
모든 Assistant Agent의 공통 베이스 클래스
LLM 호출은 core/llm_client.py를 통해 처리
→ config.yaml의 provider만 바꾸면 OpenAI / Gemini / 기타 모델로 전환 가능
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from core.llm_client import get_llm_client, call_llm

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(
        self,
        agent_id: int,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2000,
        provider: str = "openai",
        base_url: str = None,
        temperature: float = 0.0
    ):
        self.agent_id = agent_id
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = get_llm_client(provider=provider, base_url=base_url)
        self.system_prompt = self._load_prompt()
        self.temperature = temperature 

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / f"agent{self.agent_id}_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        logger.warning(f"Prompt file not found: {prompt_path}")
        return ""

    @abstractmethod
    def reason(self, state_dict: dict) -> dict:
        pass

    def _build_tom_answers(self, parsed: dict, state_dict: dict) -> list:
        """Return tom_answers as [{"id": "q1", "value": "..."}, ...].

        Prefers the new root-level tom_answers list from the LLM; falls back to
        constructing it from answer.response for backward compatibility.
        """
        raw = parsed.get("tom_answers")
        if isinstance(raw, list) and raw:
            return raw
        answer_block = parsed.get("answer", {})
        primary = answer_block.get("response", "")
        questions = state_dict.get("questions", [])
        answers = []
        for q in questions:
            qid = q.get("id", "")
            if qid == "q1":
                answers.append({"id": "q1", "value": primary})
            elif qid == "q2":
                answers.append({"id": "q2", "value": answer_block.get("q2_desire", "")})
            elif qid == "q3":
                answers.append({"id": "q3", "value": answer_block.get("q3_action", "")})
        return answers or ([{"id": "q1", "value": primary}] if primary else [])

    def _call_llm(self, user_content: str) -> str:
        """LLM 호출 (provider 무관하게 동일 인터페이스)"""
        return call_llm(
            client=self.client,
            model=self.model,
            system_prompt=self.system_prompt,
            user_content=user_content,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

    def _parse_json_response(self, raw: str) -> dict:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Agent{self.agent_id} JSON parse error: {e}\nRaw: {raw}")
            return {}

    def _build_user_prompt(self, state_dict: dict) -> str:
        prompt = f"""
Current context file:
{json.dumps(state_dict, ensure_ascii=False, indent=2)}
"""
        # MessagePool 기반 debate_context 읽기 — critique/rebuttal phase 구조 처리
        pool_ctx = state_dict.get("debate_context", {})
        if pool_ctx:
            phase = pool_ctx.get("phase", "")
            round_num = pool_ctx.get("round", 1)
            agent_key = f"agent{self.agent_id}"

            if phase in ("critique", "combined"):
                # Collect critiques directed at this agent from other agents
                incoming = [
                    (key.replace("_critique", ""), val.get(f"critique_of_{agent_key}", ""))
                    for key, val in pool_ctx.items()
                    if key.endswith("_critique")
                    and key != f"{agent_key}_critique"
                    and val.get(f"critique_of_{agent_key}")
                ]
                if incoming:
                    prompt += f"\n[DEBATE CONTEXT - Round {round_num}]\n"
                    for critic, crit_text in incoming:
                        prompt += f"\n[CRITIQUE FROM {critic}]\n{crit_text}\n"
                    prompt += "\nReconsider your answer in light of the above critiques. You may update or maintain your position.\n"

            if phase == "combined":
                # Include this agent's own rebuttal for continuity
                my_rebuttal = pool_ctx.get(f"{agent_key}_rebuttal") or {}
                rebuttal_text = my_rebuttal.get("rebuttal", "")
                if rebuttal_text:
                    prompt += f"\n[YOUR PREVIOUS REBUTTAL]\n{rebuttal_text}\n"

        # Structured ToM state from Extractor layer
        common = state_dict.get("common_state")
        if common:
            prompt += f"""
[STRUCTURED ToM STATE]
Use this structured information to reason. Cite events by id e.g. [Event 3].

Events (with observers):
{json.dumps(common.get("events", []), ensure_ascii=False, indent=2)}

Characters (entry/exit tracking):
{json.dumps(common.get("characters", []), ensure_ascii=False, indent=2)}

Belief States (based on observation access only):
{json.dumps(common.get("belief_states", []), ensure_ascii=False, indent=2)}

Goals:
{json.dumps(common.get("goals", []), ensure_ascii=False, indent=2)}

Reasoning Type: {common.get("reasoning_type")}

Raw story is also provided above for reference.
Do NOT re-parse raw story — use structured state above as primary source.
"""

        # 감독관 오류 분석 결과 반영 (max_rounds 초과 후 처음부터 재추론 시)
        if state_dict.get("supervisor_correction"):
            prompt += f"""
[SUPERVISOR CORRECTION]
The supervisor analyzed previous reasoning errors and provides the following guidance:
{state_dict['supervisor_correction']}

Please re-reason from scratch, carefully incorporating the supervisor's correction.
"""

        questions = state_dict.get("questions", [])
        if questions:
            prompt += "\n[QUESTIONS — answer every question listed below]\n"
            for q in questions:
                prompt += f"  {q.get('id', '?')}: {q.get('text', '')}\n"
            prompt += "You MUST include one entry per question ID in tom_answers.\n"

        prompt += "\nRespond ONLY in valid JSON format.\n"
        return prompt
