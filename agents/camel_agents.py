"""
agents/camel_agents.py
----------------------
E 조건: 기존 Agent1Context / Agent2Character / Agent3Perspective를 상속,
_call_llm()만 CAMEL ChatAgent로 교체.
프롬프트·추론 로직·출력 파싱은 기존 코드 그대로 사용.
기존 agent 파일 수정 없음.
"""
import logging

from agents.agent1_context import Agent1Context
from agents.agent2_character import Agent2Character
from agents.agent3_perspective import Agent3Perspective
from core.camel_wrapper import CamelClientWrapper

logger = logging.getLogger(__name__)


def _camel_mixin(base_cls):
    """기존 Agent 클래스에 CAMEL _call_llm()을 주입하는 팩토리."""

    class CamelAgent(base_cls):
        def __init__(self, model, max_tokens, provider, base_url):
            # 부모 __init__: 프롬프트 로드 + 기존 client 초기화
            super().__init__(
                model=model,
                max_tokens=max_tokens,
                provider=provider,
                base_url=base_url,
            )
            # client를 CAMEL wrapper로 교체
            self._camel_client = CamelClientWrapper()

        def _call_llm(self, user_content: str) -> str:
            """기존 OpenAI call_llm 대신 CAMEL ChatAgent 호출."""
            return self._camel_client.call(
                system_prompt=self.system_prompt,
                user_content=user_content,
                model=self.model,
                max_tokens=self.max_tokens,
            )

    CamelAgent.__name__ = f"Camel{base_cls.__name__}"
    CamelAgent.__qualname__ = f"Camel{base_cls.__qualname__}"
    return CamelAgent


CamelAgent1 = _camel_mixin(Agent1Context)
CamelAgent2 = _camel_mixin(Agent2Character)
CamelAgent3 = _camel_mixin(Agent3Perspective)
