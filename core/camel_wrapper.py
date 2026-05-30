"""
core/camel_wrapper.py
---------------------
E 조건: CAMEL-AI ChatAgent를 기존 call_llm() 인터페이스 없이 직접 래핑.
agents/camel_agents.py에서 _call_llm()을 오버라이드해 사용.

설치: pip install camel-ai
"""
import logging

logger = logging.getLogger(__name__)

_MODEL_MAP = {
    "gpt-4o":        ("OPENAI", "GPT_4O"),
    "gpt-4o-mini":   ("OPENAI", "GPT_4O_MINI"),
    "gpt-4-turbo":   ("OPENAI", "GPT_4_TURBO"),
    "gpt-3.5-turbo": ("OPENAI", "GPT_3_5_TURBO"),
}


class CamelClientWrapper:
    """
    CAMEL ChatAgent을 단일 call() 메서드로 노출.
    에이전트마다 system_prompt가 다르므로 호출마다 새 ChatAgent 생성.
    """

    def __init__(self):
        try:
            import camel  # noqa: F401
        except ImportError:
            raise ImportError("pip install camel-ai")

    def call(
        self,
        system_prompt: str,
        user_content: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 2000,
    ) -> str:
        from camel.agents import ChatAgent
        from camel.messages import BaseMessage

        camel_model = self.build_model(model, max_tokens)
        agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="ToM Analyst",
                content=system_prompt,
            ),
            model=camel_model,
        )
        response = agent.step(
            BaseMessage.make_user_message(role_name="User", content=user_content)
        )
        if not response.msgs:
            logger.warning("[CamelWrapper] empty response")
            return ""
        return response.msgs[0].content

    @staticmethod
    def build_model(model_name: str, max_tokens: int):
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType, ModelType

        platform_str, type_str = _MODEL_MAP.get(model_name, ("OPENAI", "GPT_4O_MINI"))
        return ModelFactory.create(
            model_platform=getattr(ModelPlatformType, platform_str),
            model_type=getattr(ModelType, type_str),
            model_config_dict={"max_tokens": max_tokens},
        )
