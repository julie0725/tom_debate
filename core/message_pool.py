"""
message_pool.py
---------------
전역 메시지 풀: Publish-Subscribe 패턴
- AI User만 publish 가능
- 모든 Agent는 subscribe(읽기) 가능
- 감독관은 agent_outputs와 status 업데이트 가능
"""

import threading
import logging
from typing import Optional, Callable
from dataclasses import asdict
from core.context_file import ToMState

logger = logging.getLogger(__name__)


class MessagePool:
    """
    전역 메시지 풀
    thread-safe하게 설계 (asyncio 병렬 실행 대비)
    """

    def __init__(self):
        self._state: Optional[ToMState] = None
        self._lock = threading.RLock()
        self._subscribers: list[Callable] = []

    # ── Publish ──────────────────────────────────────────────
    def publish(self, state: ToMState, publisher: str = "ai_user") -> None:
        """
        AI User가 context file을 전역 풀에 등록
        감독관도 상태 업데이트 시 이 메서드 사용
        """
        with self._lock:
            self._state = state
            logger.info(f"[MessagePool] Published by {publisher} | status={state.status} | round={state.debate_round}")
            self._notify_subscribers(publisher)

    # ── Subscribe ─────────────────────────────────────────────
    def subscribe(self, callback: Callable) -> None:
        """상태 변경 시 호출될 콜백 등록 (현재는 단순 로깅용)"""
        self._subscribers.append(callback)

    def get_state(self) -> Optional[ToMState]:
        """현재 context file 읽기 (모든 Agent 접근 가능)"""
        with self._lock:
            return self._state

    # ── Supervisor 전용 업데이트 ───────────────────────────────
    def update_agent_output(self, agent_id: int, output: dict) -> None:
        """감독관이 Agent 출력을 context file에 기록"""
        with self._lock:
            if self._state is None:
                raise ValueError("No context file published yet")
            self._state.agent_outputs[f"agent{agent_id}"] = output
            logger.debug(f"[MessagePool] Agent{agent_id} output updated")

    def update_status(self, status: str) -> None:
        with self._lock:
            if self._state:
                self._state.status = status
                logger.info(f"[MessagePool] Status → {status}")

    def update_debate_round(self, round_num: int) -> None:
        with self._lock:
            if self._state:
                self._state.debate_round = round_num

    def update_debate_context(self, debate_context: dict) -> None:
        """감독관이 라운드별 debate_context를 MessagePool에 저장 (에이전트간 직접 전달 대체)"""
        with self._lock:
            if self._state:
                self._state.debate_context = debate_context
                logger.debug(f"[MessagePool] debate_context updated | round={debate_context.get('round')}")

    def update_supervisor_correction(self, correction: str) -> None:
        """감독관이 max_rounds 초과 후 오류 분석 결과를 context file에 기록"""
        with self._lock:
            if self._state:
                self._state.supervisor_correction = correction
                logger.info("[MessagePool] supervisor_correction updated")

    def set_final_answer(self, final_answer) -> None:
        with self._lock:
            if self._state:
                self._state.final_answer = final_answer
                self._state.status = "done"

    # ── 내부 ──────────────────────────────────────────────────
    def _notify_subscribers(self, publisher: str) -> None:
        for callback in self._subscribers:
            try:
                callback(self._state, publisher)
            except Exception as e:
                logger.warning(f"[MessagePool] Subscriber callback error: {e}")

    def dump_markdown(self) -> str:
        """현재 상태를 MD로 출력 (디버깅용)"""
        if self._state:
            return self._state.to_markdown()
        return "# Empty Message Pool"


# 싱글톤 전역 인스턴스
_pool_instance: Optional[MessagePool] = None

def get_message_pool() -> MessagePool:
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = MessagePool()
    return _pool_instance

def reset_message_pool() -> None:
    """테스트/실험 간 초기화용"""
    global _pool_instance
    _pool_instance = MessagePool()
