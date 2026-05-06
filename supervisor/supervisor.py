"""
supervisor.py
"""
import asyncio
import json
import logging
import re
from pathlib import Path
from dataclasses import asdict

from core.context_file import ToMState, ToMAnswers, get_answer_value
from core.message_pool import MessagePool
from core.llm_client import get_llm_client, call_llm
from core.run_logger import RunLogger
from agents.agent1_context import Agent1Context
from agents.agent2_character import Agent2Character
from agents.agent3_perspective import Agent3Perspective
from supervisor.debate import DebateManager

logger = logging.getLogger(__name__)


def _extract_choice_letter(text: str) -> str:
    if not text:
        return ""
    m = re.match(r'^([A-Z])(?:\.|\s|$)', text.strip())
    return m.group(1) if m else text.strip()


def _parse_final_answer(fa) -> dict:
    """Convert supervisor final_answer (list or legacy dict) to {q_id: value} map."""
    if isinstance(fa, list):
        return {a.get("id"): (a.get("value") or "") for a in fa if a.get("id")}
    if isinstance(fa, dict):
        result = {}
        for k, v in fa.items():
            if v:
                qid = k.replace("_belief", "").replace("_desire", "").replace("_action", "")
                result[qid] = v
        return result
    return {}


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
        self.client = get_llm_client(provider=self.provider, base_url=self.base_url)
        self.system_prompt = self._load_prompt()
        self.correction_prompt = self._load_correction_prompt()

        self.run_logger: RunLogger = None
        self.output_dir = config.get("evaluation", {}).get("output_dir", "outputs/")

    def _load_prompt(self) -> str:
        p = Path(__file__).parent.parent / "prompts" / "supervisor_prompt.txt"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def _load_correction_prompt(self) -> str:
        p = Path(__file__).parent.parent / "prompts" / "supervisor_correction_prompt.txt"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    async def run(self) -> ToMState:
        state = self.pool.get_state()
        if state is None:
            raise ValueError("No context file in message pool.")

        self.run_logger = RunLogger(output_dir=self.output_dir, dataset_id=state.dataset_id)

        logger.info("[Supervisor] Starting pipeline")
        self.pool.update_status("pending")
        self.run_logger.log_context_file(asdict(state), label="initial")

        try:
            agent_outputs = await self._run_agents_parallel(state)
            for agent_id, output in agent_outputs.items():
                self.pool.update_agent_output(agent_id, output)

            state = self.pool.get_state()
            self.run_logger.log_agent_outputs(state.agent_outputs, label="initial")
            self.run_logger.log_context_file(asdict(state), label="after_initial_reasoning")

            supervisor_result = self._call_supervisor(state, debate_round=0)
            logger.info(f"[Supervisor] Initial check: agreement={supervisor_result.get('agreement')}")

            if supervisor_result.get("agreement") or not self.use_debate:
                final = self._extract_final_answer(supervisor_result, state=state)
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
                    supervisor_correction_fn=self._call_supervisor_correction,
                    run_logger=self.run_logger
                )
                self.pool.set_final_answer(final)
        finally:
            final_state = self.pool.get_state()
            self.run_logger.log_context_file(asdict(final_state), label="final")
            self.run_logger.log_final_summary(asdict(final_state))

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

    def _call_supervisor(self, state: ToMState, debate_round: int) -> dict:
        state_dict = asdict(state)

        # Build per-agent answer table: {agent_key: {q_id: letter}}
        answer_table = {}
        for agent_key, output in (state_dict.get("agent_outputs") or {}).items():
            if output:
                tom_ans = output.get("tom_answers", [])
                if isinstance(tom_ans, list):
                    answer_table[agent_key] = {
                        a.get("id"): _extract_choice_letter(a.get("value", ""))
                        for a in tom_ans if a.get("id")
                    }
                elif isinstance(tom_ans, dict):
                    answer_table[agent_key] = {
                        "q1": _extract_choice_letter(tom_ans.get("q1_belief", ""))
                    }

        user_content = f"""
Current context file:
{json.dumps(state_dict, ensure_ascii=False, indent=2)}

Current debate_round: {debate_round}
Max rounds allowed: {self.max_rounds}

Normalized answer labels (letter only, for agreement check):
{json.dumps(answer_table, ensure_ascii=False)}

Your task:
- Compare the normalized labels above for agreement on each question id
- If all present agents share the same label for every question → agreement: true
- Otherwise → agreement: false, trigger debate

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
        state_dict = asdict(state)
        user_content = f"""
Agents failed to reach consensus after {self.max_rounds} debate rounds.

Scenario:
{state_dict.get('scenario', '')}

Questions:
{json.dumps(state_dict.get('questions', []), ensure_ascii=False, indent=2)}

Agent outputs:
{json.dumps(state_dict.get('agent_outputs', {}), ensure_ascii=False, indent=2)}

Analyze why the agents disagree and provide correction guidance.
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

    def _extract_final_answer(self, supervisor_result: dict, state: ToMState = None) -> ToMAnswers:
        ans_map = _parse_final_answer(supervisor_result.get("final_answer"))

        if not ans_map.get("q1") and state is not None:
            for output in (asdict(state).get("agent_outputs") or {}).values():
                if not output:
                    continue
                tom_ans = output.get("tom_answers")
                q1_val = _extract_choice_letter(get_answer_value(tom_ans, "q1"))
                if q1_val:
                    if isinstance(tom_ans, list):
                        for a in tom_ans:
                            qid = a.get("id")
                            if qid and not ans_map.get(qid):
                                ans_map[qid] = _extract_choice_letter(a.get("value", ""))
                    break

        return ToMAnswers(answers=[{"id": k, "value": v} for k, v in ans_map.items() if v])
