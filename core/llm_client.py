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
import time
import threading
from openai import OpenAI, RateLimitError

logger = logging.getLogger(__name__)

MODEL_PRICING = {
    "gpt-3.5-turbo":  {"prompt": 0.50,  "completion": 1.50},
    "gpt-4o-mini":    {"prompt": 0.15,  "completion": 0.60},
    "gpt-4o":         {"prompt": 5.00,  "completion": 15.00},
    "gpt-4":          {"prompt": 30.00, "completion": 60.00},
}


class TokenCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, prompt: int, completion: int):
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0

    def get(self):
        with self._lock:
            return self.prompt_tokens, self.completion_tokens

    def cost(self, model: str) -> float:
        pricing = MODEL_PRICING.get(model) or MODEL_PRICING.get("gpt-3.5-turbo")
        p, c = self.get()
        return (p * pricing["prompt"] + c * pricing["completion"]) / 1_000_000


_thread_local = threading.local()


def set_token_counter(counter: "TokenCounter | None") -> None:
    _thread_local.active_counter = counter


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
    max_tokens: int = 2000,
    temperature: float = 0.0
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
        temperature  : 샘플링 온도 (0.0 = 결정적)

    Returns:
        LLM 응답 텍스트
    """
    max_retries = 6
    base_delay = 5.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            )
            usage = response.usage
            if usage:
                logger.info(
                    f"[LLMClient] tokens — prompt: {usage.prompt_tokens}, "
                    f"completion: {usage.completion_tokens}/{max_tokens} "
                    f"({'NEAR LIMIT' if usage.completion_tokens >= max_tokens * 0.9 else 'ok'})"
                )
                counter = getattr(_thread_local, "active_counter", None)
                if counter is not None:
                    counter.add(usage.prompt_tokens, usage.completion_tokens)
            return response.choices[0].message.content
        except RateLimitError:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"[LLMClient] Rate limit hit, retry {attempt + 1}/{max_retries - 1} in {delay:.0f}s")
                time.sleep(delay)
            else:
                raise
