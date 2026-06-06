"""
supervisor.py
"""
import asyncio
import json
import logging
from pathlib import Path
from dataclasses import asdict

from core.context_file import ToMState
from core.message_pool import MessagePool
from core.llm_client import get_llm_client, call_llm
from core.run_logger import RunLogger
from agents.agent1_context import Agent1Context
from agents.agent2_character import Agent2Character
from agents.agent3_perspective import Agent3Perspective
from supervisor.debate import DebateManager

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(self, pool: MessagePool, config: dict, progress_callback=None):
        self.pool = pool
        self.config = config

        sys_cfg = config.get("system", {})
        self.model = sys_cfg.get("model", "gpt-3.5-turbo")
        self.max_tokens = sys_cfg.get("max_tokens", 2000)
        self.provider = sys_cfg.get("provider", "openai")
        self.base_url = sys_cfg.get("base_url", None)
        self.temperature = sys_cfg.get("temperature", 0.0)

        self.max_rounds = config.get("debate", {}).get("max_rounds", 3)
        self.use_debate = config.get("debate", {}).get("use_debate", True)
        self.tiebreak_agent = config.get("debate", {}).get("tiebreak_agent", 3)
        self.use_correction = config.get("supervisor", {}).get("use_correction", True)

        self.progress_callback = progress_callback or (lambda msg, pct: None)

        agent_cfg = config.get("agents", {})
        self.agents = {}
        if agent_cfg.get("use_agent1", True):
            self.agents[1] = Agent1Context(
                model=self.model, max_tokens=self.max_tokens,
                provider=self.provider, base_url=self.base_url,
                temperature=self.temperature
            )
        if agent_cfg.get("use_agent2", True):
            self.agents[2] = Agent2Character(
                model=self.model, max_tokens=self.max_tokens,
                provider=self.provider, base_url=self.base_url,
                temperature=self.temperature
            )
        if agent_cfg.get("use_agent3", True):
            self.agents[3] = Agent3Perspective(
                model=self.model, max_tokens=self.max_tokens,
                provider=self.provider, base_url=self.base_url,
                temperature=self.temperature
            )

        self.client = get_llm_client(provider=self.provider, base_url=self.base_url)
        self.event_callback = None
        self.debate_manager = DebateManager(
            agents=self.agents,
            max_rounds=self.max_rounds,
            tiebreak_agent=self.tiebreak_agent,
            client=self.client,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            event_callback=self.event_callback,
        )
        self.correction_prompt = self._load_correction_prompt()

        self.run_logger: RunLogger = None
        self.output_dir = config.get("evaluation", {}).get("output_dir", "outputs/")

    def _load_correction_prompt(self) -> str:
        p = Path(__file__).parent.parent / "prompts" / "supervisor_correction_prompt.txt"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    async def run(self) -> ToMState:
        self.debate_manager.event_callback = self.event_callback or (lambda e, d: None)
        state = self.pool.get_state()
        if state is None:
            raise ValueError("No context file in message pool.")

        self.run_logger = RunLogger(output_dir=self.output_dir, dataset_id=state.dataset_id)

        logger.info("[Supervisor] Starting pipeline")
        self.pool.update_status("pending")
        self.run_logger.log_context_file(asdict(state), label="initial")

        try:
            agent_outputs = await self._run_agents_parallel(state)
            if self.event_callback:
                self.event_callback("agent_answer", {
                    "answers": {
                        f"agent{aid}": [{"id": a.get("id"), "value": a.get("value")}
                                        for a in (out or {}).get("tom_answers", [])]
                        for aid, out in agent_outputs.items()
                    }
                })
            self.progress_callback("에이전트 추론 완료", 40)  # hook
            for agent_id, output in agent_outputs.items():
                self.pool.update_agent_output(agent_id, output)

            state = self.pool.get_state()
            self.run_logger.log_agent_outputs(state.agent_outputs, label="initial")

            # 초기 출력 freeze — 이후 토론이 덮어써도 보존됨
            state.initial_agent_outputs = {
                k: dict(v) if v else None
                for k, v in state.agent_outputs.items()
            }

            self.run_logger.log_context_file(asdict(state), label="after_initial_reasoning")

            agreement, _ = self.debate_manager._check_agreement(state)
            logger.info(f"[Supervisor] Initial check: agreement={agreement}")

            if agreement or not self.use_debate:
                final = self.debate_manager._extract_answer_from_state(state)
                self.pool.set_final_answer(final)
                logger.info("[Supervisor] Agreement reached. No debate needed.")
                self.progress_callback("합의 완료", 80)  # hook
            else:
                state = self.pool.get_state()
                state.debate_triggered = True
                self.pool.update_status("debating")
                logger.info("[Supervisor] Disagreement detected. Starting debate.")
                correction_fn = self._call_supervisor_correction if self.use_correction else None
                self.progress_callback("토론 시작", 50)  # hook
                final = await self.debate_manager.run_debate(
                    pool=self.pool,
                    supervisor_correction_fn=correction_fn,
                    run_logger=self.run_logger
                )
                self.progress_callback("토론 완료", 80)  # hook
                self.pool.set_final_answer(final)
        finally:
            final_state = self.pool.get_state()
            self.run_logger.log_context_file(asdict(final_state), label="final")
            self.run_logger.log_final_summary(asdict(final_state))
            self.progress_callback("완료", 100)  # hook

        return final_state

    async def _run_agents_parallel(self, state: ToMState) -> dict:
        state_dict = asdict(state)

        async def run_agent(agent_id, agent):
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, agent.reason, state_dict)
            logger.info(f"[Supervisor] Agent{agent_id} reasoning complete")
            return agent_id, output

        tasks = [run_agent(aid, agent) for aid, agent in self.agents.items()]
        results = await asyncio.gather(*tasks)
        return {aid: output for aid, output in results}

    def _call_supervisor_correction(self, state: ToMState, flags: list = None) -> str:
        state_dict = asdict(state)

        filtered_outputs = {
            agent_key: {
                "reasoning": (output or {}).get("reasoning", ""),
                "belief_state": (output or {}).get("belief_state"),
                "tom_answers": (output or {}).get("tom_answers", []),
            }
            for agent_key, output in (state_dict.get("agent_outputs") or {}).items()
            if output
        }

        common = state_dict.get("common_state") or {}
        filtered_common = {
            "events": common.get("events", []),
            "characters": common.get("characters", []),
            "belief_states": common.get("belief_states", []),
            "goals": common.get("goals", []),
        }

        user_content = f"""Agents failed to reach consensus after {self.max_rounds} debate rounds.

Scenario (full text):
{state_dict.get("scenario", "")}

Questions:
{json.dumps(state_dict.get("questions", []), ensure_ascii=False, indent=2)}

Reasoning type: {state_dict.get("reasoning_type", "unknown")}

Structured ToM state (no answer information):
{json.dumps(filtered_common, ensure_ascii=False, indent=2)}

Debate flags (critique/rebuttal behavior per round):
{json.dumps(flags or [], ensure_ascii=False, indent=2)}

Agent outputs (reasoning chains and answers):
{json.dumps(filtered_outputs, ensure_ascii=False, indent=2)}

INSTRUCTIONS:
- reasoning_type tells you what order of belief reasoning is required.
  For 2nd-order or higher, the correct answer may require inferring what a character
  believes about another character's belief — this is valid even if not directly observable.
- agent_outputs.belief_state is each agent's own interpreted belief, not ground truth.
  Evaluate whether the reasoning chain is internally consistent and tracks epistemic access correctly.
  Do NOT confirm or infer the correct answer.
- Use flags to identify which agents ignored critiques and which accepted them.
- Identify whose reasoning correctly tracks each character's epistemic access given the scenario.
- Provide correction guidance to help agents reach consensus through better reasoning.
"""
        correction = call_llm(
            client=self.client,
            model=self.model,
            system_prompt=self.correction_prompt,
            user_content=user_content,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        logger.info(f"[Supervisor] Correction generated: {correction[:100]}...")
        return correction

