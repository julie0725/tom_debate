"""
core/camel_baseline.py
----------------------
F 조건: 변형 없는 CAMEL-AI 단일 ChatAgent로 ToM 직접 추론.
Supervisor / 3-agent / Debate 구조를 전혀 사용하지 않음.

설치: pip install camel-ai
"""
import json
import logging
import re

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert at Theory of Mind (ToM) reasoning.
Given a scenario and questions about characters' beliefs, desires, or actions,
reason carefully and answer each question.

Rules:
- For multiple-choice questions, output ONLY the choice letter (e.g. "K").
- For open-ended questions, output a concise answer string.
- Output ONE valid JSON object. No markdown, no prose outside JSON.

Output schema:
{
  "tom_answers": [
    {"id": "<question_id>", "value": "<answer>"}
  ]
}"""


class CamelBaseline:
    """
    ablation_ai_user.py의 _submit_camel_baseline()에서 호출됨.
    """

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 2000):
        self.model = model
        self.max_tokens = max_tokens
        try:
            import camel  # noqa: F401
        except ImportError:
            raise ImportError("pip install camel-ai")

    def run(self, state_dict: dict) -> dict:
        """
        Args:
            state_dict: {"scenario": str, "questions": [{"id": str, "text": str}]}
        Returns:
            {"tom_answers": [{"id": str, "value": str}]}
        """
        from camel.agents import ChatAgent
        from camel.messages import BaseMessage
        from core.camel_wrapper import CamelClientWrapper

        camel_model = CamelClientWrapper.build_model(self.model, self.max_tokens)
        agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="ToM Expert",
                content=_SYSTEM_PROMPT,
            ),
            model=camel_model,
        )
        response = agent.step(
            BaseMessage.make_user_message(
                role_name="User",
                content=self._build_user_content(state_dict),
            )
        )
        raw = response.msgs[0].content if response.msgs else ""
        return self._parse(raw, state_dict)

    @staticmethod
    def _build_user_content(state_dict: dict) -> str:
        scenario = state_dict.get("scenario", "")
        questions = state_dict.get("questions", [])
        q_lines = "\n".join(
            f"  {q.get('id','?')}: {q.get('text','')}" for q in questions
        )
        return f"Scenario:\n{scenario}\n\nQuestions:\n{q_lines}\n\nAnswer every question."

    @staticmethod
    def _parse(raw: str, state_dict: dict) -> dict:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"[CamelBaseline] JSON parse error: {raw[:120]}")
            parsed = {}

        if not isinstance(parsed.get("tom_answers"), list):
            parsed["tom_answers"] = [
                {"id": q.get("id", "q1"), "value": ""}
                for q in state_dict.get("questions", [])
            ]
        return parsed
