"""
llm_client.py
-------------
LLM 추상화 레이어
config.yaml의 provider 설정만 바꾸면 OpenAI / Gemini / 기타 모델로 전환 가능

지원 provider:
  - openai   : gpt-4o, gpt-4o-mini, gpt-3.5-turbo 등
  - gemini   : gemini-1.5-pro, gemini-1.5-flash 등 (openai 호환 엔드포인트 사용)
  - custom   : base_url 직접 지정 (로컬 모델, Azure 등)
"""

import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


def get_llm_client(provider: str = "openai", base_url: str = None) -> OpenAI:
    """
    provider에 따라 적절한 LLM 클라이언트 반환
    모두 OpenAI-compatible API 형식을 사용함

    Args:
        provider : "openai" | "gemini" | "custom"
        base_url : custom 엔드포인트 URL (provider="custom" 시 필수)

    Returns:
        OpenAI 클라이언트 인스턴스 (호환 API)
    """
    if provider == "openai":
        # OPENAI_API_KEY 환경변수에서 자동 로드
        client = OpenAI()
        logger.info("[LLMClient] Provider: OpenAI")

    elif provider == "gemini":
        # Gemini는 OpenAI 호환 엔드포인트 제공
        # GEMINI_API_KEY 환경변수 필요
        import os
        api_key = os.environ.get("GEMINI_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        logger.info("[LLMClient] Provider: Gemini")

    elif provider == "custom":
        # 로컬 모델, Azure OpenAI, 기타 호환 API
        import os
        api_key = os.environ.get("CUSTOM_API_KEY", "none")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        logger.info(f"[LLMClient] Provider: Custom ({base_url})")

    else:
        logger.warning(f"[LLMClient] Unknown provider '{provider}', fallback to OpenAI")
        client = OpenAI()

    return client


def call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int = 2000
) -> str:
    """
    통일된 LLM 호출 함수
    모든 에이전트/감독관이 이 함수를 통해 호출

    Args:
        client       : get_llm_client()로 얻은 클라이언트
        model        : 모델명 (예: "gpt-3.5-turbo", "gemini-1.5-flash")
        system_prompt: 시스템 프롬프트
        user_content : 유저 메시지
        max_tokens   : 최대 토큰 수

    Returns:
        LLM 응답 텍스트
    """
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content
