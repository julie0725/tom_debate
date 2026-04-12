"""
supervisor.py
-------------
감독관 에이전트
- Agent 1/2/3 병렬 실행 지시
- 답변 수집 및 일치 여부 판단
- 토론 trigger 및 최종 답변 결정
LLM 호출은 core/llm_client.py를 통해 처리 (provider 유동 전환 가능)
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from dataclasses import asdict

from core.context_file import ToMState, ToMAnswers
from core.message_pool import MessagePool
from core.llm_client import get_llm_client, call_llm
from agents.agent1_context import Agent1Context
from agents.agent2_character import Agent2Character
from agents.agent3_perspective import Agent3Perspective
from supervisor.debate import DebateManager

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(self, pool: MessagePool, config: dict):
        self.pool = pool
        self.config = config

        sys_cfg = config.get("system", {})
        self.model = sys_cfg.get("model", "gpt-3.5-turbo")
        self.max_tokens = sys_cfg.get("max_tokens", 2000)
        self.provider = sys_cfg.get("provider", "openai")
        self.base_url = sys_cfg.get("base_url", None)

        self.max_rounds = config.get("debate", {}).get("max_rounds", 3)
        self.use_debate = config.get("debate", {}).get("use_debate", True)
        self.tiebreak_agent = config.get("debate", {}).get("tiebreak_agent", 3)

        # 활성화된 Agent만 생성 (ablation 제어)
        agent_cfg = config.get("agents", {})
        self.agents = {}
        if agent_cfg.get("use_agent1", True):
            self.agents[1] = Agent1Context(
                model=self.model, max_tokens=self.max_tokens,
                provider=self.provider, base_url=self.base_url
            )
        if agent_cfg.get("use_agent2", True):
            self.agents[2] = Agent2Character(
                model=self.model, max_tokens=self.max_tokens,
                provider=self.provider, base_url=self.base_url
            )
        if agent_cfg.get("use_agent3", True):
            self.agents[3] = Agent3Perspective(
                model=self.model, max_tokens=self.max_tokens,
                provider=self.provider, base_url=self.base_url
            )

        self.debate_manager = DebateManager(self.agents, self.max_rounds, self.tiebreak_agent)

        # 감독관 LLM 클라이언트 (provider 무관)
        self.client = get_llm_client(provider=self.provider, base_url=self.base_url)
        self.system_prompt = self._load_prompt()
        self.correction_prompt = self._load_correction_prompt()

    def _load_prompt(self) -> str:
        p = Path(__file__).parent.parent / "prompts" / "supervisor_prompt.txt"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def _load_correction_prompt(self) -> str:
        p = Path(__file__).parent.parent / "prompts" / "supervisor_correction_prompt.txt"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # ── 메인 실행 ─────────────────────────────────────────────
    async def run(self) -> ToMState:
        state = self.pool.get_state()
        if state is None:
            raise ValueError("No context file in message pool. AI User must publish first.")

        logger.info("[Supervisor] Starting pipeline")
        self.pool.update_status("pending")

        # Step 1: Agent 병렬 추론
        agent_outputs = await self._run_agents_parallel(state)
        for agent_id, output in agent_outputs.items():
            self.pool.update_agent_output(agent_id, output)

        state = self.pool.get_state()

        # Step 2: 감독관 판단
        supervisor_result = self._call_supervisor(state, debate_round=0)
        logger.info(f"[Supervisor] Initial check: agreement={supervisor_result.get('agreement')}")

        # Step 3: 토론 여부 결정
        if supervisor_result.get("agreement") or not self.use_debate:
            final = self._extract_final_answer(supervisor_result)
            self.pool.set_final_answer(final)
            logger.info("[Supervisor] Agreement reached. No debate needed.")
        else:
            state = self.pool.get_state()
            state.debate_triggered = True
            self.pool.update_status("debating")
            logger.info("[Supervisor] Disagreement detected. Starting debate.")
            final = await self.debate_manager.run_debate(
                pool=self.pool,
                supervisor_call_fn=self._call_supervisor,
                supervisor_correction_fn=self._call_supervisor_correction
            )
            self.pool.set_final_answer(final)

        return self.pool.get_state()

    # ── Agent 병렬 실행 ───────────────────────────────────────
    async def _run_agents_parallel(self, state: ToMState) -> dict:
        state_dict = asdict(state)

        async def run_agent(agent_id, agent):
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, agent.reason, state_dict, None)
            logger.info(f"[Supervisor] Agent{agent_id} reasoning complete")
            return agent_id, output

        tasks = [run_agent(aid, agent) for aid, agent in self.agents.items()]
        results = await asyncio.gather(*tasks)
        return {aid: output for aid, output in results}

    # ── 감독관 LLM 호출 ───────────────────────────────────────
    def _call_supervisor(self, state: ToMState, debate_round: int) -> dict:
        state_dict = asdict(state)
        user_content = f"""
Current context file:
{json.dumps(state_dict, ensure_ascii=False, indent=2)}

Current debate_round: {debate_round}
Max rounds allowed: {self.max_rounds}

Your task:
- Compare agent answers for q1_belief, q2_desire, q3_action
- If debate_round >= max_rounds, apply majority vote
- Otherwise check agreement and decide next step

Respond ONLY in valid JSON format.
"""
        raw = call_llm(
            client=self.client,
            model=self.model,
            system_prompt=self.system_prompt,
            user_content=user_content,
            max_tokens=self.max_tokens
        )
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"[Supervisor] JSON parse error: {raw}")
            return {"agreement": False, "status": "debating", "final_answer": {}}

    def _call_supervisor_correction(self, state: ToMState) -> str:
        """
        max_rounds 초과 시 호출
        - 에이전트 출력의 불일치 원인 분석
        - context file에 기록할 오류 수정 지침 반환 (plain text)
        - 이후 에이전트들이 state_dict['supervisor_correction']을 읽고 처음부터 재추론
        """
        state_dict = asdict(state)
        user_content = f"""
Agents failed to reach consensus after {self.max_rounds} debate rounds.

Scenario:
{state_dict.get('scenario', '')}

Questions:
{json.dumps(state_dict.get('questions', {}), ensure_ascii=False, indent=2)}

Agent outputs:
{json.dumps(state_dict.get('agent_outputs', {}), ensure_ascii=False, indent=2)}

Analyze why the agents disagree, identify the reasoning error(s), and provide clear correction guidance.
"""
        correction = call_llm(
            client=self.client,
            model=self.model,
            system_prompt=self.correction_prompt,
            user_content=user_content,
            max_tokens=self.max_tokens
        )
        logger.info(f"[Supervisor] Correction generated: {correction[:100]}...")
        return correction

    def _extract_final_answer(self, supervisor_result: dict) -> ToMAnswers:
        fa = supervisor_result.get("final_answer", {})
        return ToMAnswers(
            q1_belief=fa.get("q1_belief"),
            q2_desire=fa.get("q2_desire"),
            q3_action=fa.get("q3_action")
        )
