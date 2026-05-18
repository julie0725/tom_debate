"""
debate.py
"""
import asyncio
import copy
import json
import logging
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from core.context_file import ToMState, ToMAnswers, get_answer_value
from core.message_pool import MessagePool
from core.llm_client import call_llm

logger = logging.getLogger(__name__)


def _extract_choice_letter(text: str) -> str:
    if not text:
        return ""
    m = re.match(r'^([A-Z])(?:\.|\s|$)', text.strip())
    return m.group(1) if m else text.strip()



class DebateManager:
    def __init__(
        self,
        agents: dict,
        max_rounds: int,
        tiebreak_agent: int = 3,
        client=None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2000,
    ):
        self.agents = agents
        self.max_rounds = max_rounds
        self.tiebreak_agent = tiebreak_agent
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self._critique_prompt = self._load_prompt("debate_critique_prompt.txt")
        self._rebuttal_prompt = self._load_prompt("debate_rebuttal_prompt.txt")
        self.accumulated_flags: list = []

    def _load_prompt(self, filename: str) -> str:
        p = Path(__file__).parent.parent / "prompts" / filename
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # ── Main debate loop ──────────────────────────────────────────────────────

    async def run_debate(
        self,
        pool: MessagePool,
        supervisor_correction_fn=None,
        run_logger=None,
    ) -> ToMAnswers:

        self.accumulated_flags = []

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"[Debate] Round {round_num} / {self.max_rounds}")
            pool.update_debate_round(round_num)

            state = pool.get_state()
            outputs_before = copy.deepcopy(state.agent_outputs)

            critiques = await self._run_critique_phase(pool, round_num, run_logger)
            self._print_critique_phase(round_num, critiques)

            rebuttal_results = await self._run_rebuttal_phase(pool, round_num, critiques, run_logger)
            self._print_rebuttal_phase(round_num, rebuttal_results)
            self._accumulate_flags(critiques, rebuttal_results, outputs_before, round_num)

            state = pool.get_state()
            agreement, answer_map = self._check_agreement(state)
            self._print_round_result(round_num, {"agreement": agreement, "answer_map": answer_map})
            logger.info(f"[Debate] Round {round_num} agreement={agreement}")

            if run_logger:
                run_logger.log_debate_round(
                    round_num=round_num,
                    debate_context=state.debate_context,
                    agent_outputs_before=outputs_before,
                    agent_outputs_after=dict(state.agent_outputs),
                    supervisor_result={"agreement": agreement, "answer_map": answer_map},
                )
                run_logger.log_agent_outputs(state.agent_outputs, label=f"debate_round_{round_num:02d}")
                run_logger.log_context_file(asdict(state), label=f"debate_round_{round_num:02d}")

            if agreement:
                logger.info(f"[Debate] Consensus reached at round {round_num}")
                return self._extract_answer_from_state(state)

        logger.info("[Debate] Max rounds reached. Supervisor analyzing errors...")

        if supervisor_correction_fn:
            state = pool.get_state()
            correction = supervisor_correction_fn(state, self.accumulated_flags)
            pool.update_supervisor_correction(correction)

            if run_logger:
                run_logger.log_supervisor_correction(correction)

            pool.update_debate_context({})
            logger.info("[Debate] Re-reasoning from scratch...")

            await self._re_reason_fresh(pool)

            state = pool.get_state()
            agreement, answer_map = self._check_agreement(state)

            if run_logger:
                run_logger.log_agent_outputs(state.agent_outputs, label="fresh_reInfer")
                run_logger.log_context_file(asdict(state), label="after_correction_reInfer")

            if agreement:
                logger.info("[Debate] Consensus reached after correction.")
                return self._extract_answer_from_state(state)

        logger.info("[Debate] Applying majority vote.")
        state = pool.get_state()
        state.majority_vote_applied = True
        return self._majority_vote(state)

    # ── Phase 1: Critique ─────────────────────────────────────────────────────

    async def _run_critique_phase(
        self, pool: MessagePool, round_num: int, run_logger=None
    ) -> dict:
        """Each agent critiques the other two, citing specific timeline events."""
        state = pool.get_state()
        state_dict = asdict(state)
        agent_ids = list(self.agents.keys())

        async def critique_one(agent_id: int) -> tuple[int, dict]:
            agent_key = f"agent{agent_id}"
            user_content = (
                f"You are agent{agent_id}.\n\n"
                f"Scenario:\n{state_dict['scenario']}\n\n"
                f"Questions:\n{json.dumps(state_dict['questions'], ensure_ascii=False)}\n\n"
                f"All agents' current outputs:\n"
                f"{json.dumps(state_dict['agent_outputs'], ensure_ascii=False, indent=2)}\n\n"
                f"Critique the reasoning of every agent EXCEPT yourself. "
                f"Cite specific event numbers as evidence. "
                f"Set critique_of_{agent_key} to empty string."
            )
            raw = await asyncio.get_event_loop().run_in_executor(
                None, call_llm,
                self.client, self.model, self._critique_prompt, user_content, self.max_tokens,
            )
            cleaned = re.sub(r"```json|```", "", raw).strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(f"[Debate] Critique JSON parse error (agent{agent_id}): {raw[:120]}")
                parsed = {f"critique_of_agent{aid}": "" for aid in agent_ids}
                parsed["my_answer"] = ""
            parsed[f"critique_of_{agent_key}"] = ""   # enforce own field empty
            parsed["agent_id"] = agent_id
            return agent_id, parsed

        results = await asyncio.gather(*[critique_one(aid) for aid in agent_ids])
        critiques = {f"agent{aid}": out for aid, out in results}

        context = {
            "round": round_num,
            "phase": "critique",
            **{f"agent{aid}_critique": out for aid, out in results},
        }
        pool.update_debate_context(context)

        if run_logger:
            run_logger.log_context_file(asdict(pool.get_state()), label=f"critique_round_{round_num:02d}")

        logger.info(f"[Debate] Critique phase done (round {round_num})")
        return critiques

    # ── Phase 2: Rebuttal ─────────────────────────────────────────────────────

    async def _run_rebuttal_phase(
        self, pool: MessagePool, round_num: int, critiques: dict, run_logger=None
    ) -> None:
        """Each agent rebuts critiques directed at it and updates its answer."""
        state = pool.get_state()
        state_dict = asdict(state)
        agent_ids = list(self.agents.keys())

        async def rebuttal_one(agent_id: int) -> tuple[int, dict]:
            agent_key = f"agent{agent_id}"
            incoming = {
                critic_key: c_out.get(f"critique_of_{agent_key}", "")
                for critic_key, c_out in critiques.items()
                if critic_key != agent_key
                and c_out.get(f"critique_of_{agent_key}")
            }
            user_content = (
                f"You are agent{agent_id}.\n\n"
                f"Scenario:\n{state_dict['scenario']}\n\n"
                f"Questions:\n{json.dumps(state_dict['questions'], ensure_ascii=False)}\n\n"
                f"Your current output:\n"
                f"{json.dumps(state_dict['agent_outputs'].get(agent_key, {}), ensure_ascii=False, indent=2)}\n\n"
                f"Critiques directed at you:\n"
                f"{json.dumps(incoming, ensure_ascii=False, indent=2)}"
            )
            raw = await asyncio.get_event_loop().run_in_executor(
                None, call_llm,
                self.client, self.model, self._rebuttal_prompt, user_content, self.max_tokens,
            )
            cleaned = re.sub(r"```json|```", "", raw).strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(f"[Debate] Rebuttal JSON parse error (agent{agent_id}): {raw[:120]}")
                parsed = {"rebuttal": raw[:300], "my_answer": ""}
            parsed["agent_id"] = agent_id
            return agent_id, parsed

        results = await asyncio.gather(*[rebuttal_one(aid) for aid in agent_ids])

        # Merge revised my_answer back into each agent's tom_answers (q1)
        for agent_id, rebuttal_out in results:
            agent_key = f"agent{agent_id}"
            new_letter = _extract_choice_letter(rebuttal_out.get("my_answer", ""))
            if new_letter:
                current = dict(state.agent_outputs.get(agent_key) or {})
                tom = list(current.get("tom_answers") or [])
                updated = False
                for entry in tom:
                    if entry.get("id") == "q1":
                        entry["value"] = new_letter
                        updated = True
                        break
                if not updated:
                    tom = [{"id": "q1", "value": new_letter}] + tom
                pool.update_agent_output(agent_id, {**current, "tom_answers": tom})

        # Combined context carries both phases so the logger can render the full transcript
        context = {
            "round": round_num,
            "phase": "combined",
            **{f"{k}_critique": v for k, v in critiques.items()},
            **{f"agent{aid}_rebuttal": out for aid, out in results},
        }
        pool.update_debate_context(context)

        if run_logger:
            run_logger.log_context_file(asdict(pool.get_state()), label=f"rebuttal_round_{round_num:02d}")

        logger.info(f"[Debate] Rebuttal phase done (round {round_num})")
        return results

    # ── Console transcript printers ───────────────────────────────────────────

    @staticmethod
    def _print_critique_phase(round_num: int, critiques: dict) -> None:
        W = 40
        print(f"\n{'=' * W}")
        print(f"[Round {round_num}] CRITIQUE PHASE")
        print(f"{'=' * W}")
        for src_key in ["agent1", "agent2", "agent3"]:
            c_out = critiques.get(src_key, {})
            for tgt_key in ["agent1", "agent2", "agent3"]:
                if tgt_key == src_key:
                    continue
                crit_text = str(c_out.get(f"critique_of_{tgt_key}", ""))
                if crit_text:
                    trail = "..." if len(crit_text) > 150 else ""
                    src = src_key.replace("agent", "Agent")
                    tgt = tgt_key.replace("agent", "Agent")
                    print(f"{src} → {tgt}: \"{crit_text[:150]}{trail}\"")

    @staticmethod
    def _print_rebuttal_phase(round_num: int, rebuttal_results) -> None:
        W = 40
        print(f"\n{'=' * W}")
        print(f"[Round {round_num}] REBUTTAL PHASE")
        print(f"{'=' * W}")
        for agent_id, r_out in sorted(rebuttal_results, key=lambda x: x[0]):
            rebuttal_text = str(r_out.get("rebuttal", ""))
            answer = r_out.get("my_answer", "?")
            trail = "..." if len(rebuttal_text) > 150 else ""
            print(f"Agent{agent_id} rebuttal: \"{rebuttal_text[:150]}{trail}\"")
            print(f"  → answer: {answer}")

    @staticmethod
    def _print_round_result(round_num: int, result: dict) -> None:
        W = 40
        print(f"\n{'=' * W}")
        print(f"[Round {round_num}] RESULT")
        print(f"{'=' * W}")
        print(f"Agreement: {result.get('agreement', '?')}")
        if result.get("answer_map"):
            for qid, votes in result["answer_map"].items():
                print(f"  {qid}: {votes}")

    # ── Fresh re-reason (after correction) ───────────────────────────────────

    async def _re_reason_fresh(self, pool: MessagePool) -> None:
        state = pool.get_state()
        state_dict = asdict(state)

        async def re_reason_agent(agent_id, agent):
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, agent.reason, state_dict)
            return agent_id, output

        results = await asyncio.gather(*[re_reason_agent(aid, ag) for aid, ag in self.agents.items()])
        for agent_id, output in results:
            pool.update_agent_output(agent_id, output)

    # ── Majority vote ─────────────────────────────────────────────────────────

    def _majority_vote(self, state: ToMState) -> ToMAnswers:
        outputs = [
            state.agent_outputs.get(f"agent{i}")
            for i in [1, 2, 3]
            if state.agent_outputs.get(f"agent{i}") is not None
        ]

        q_ids: list[str] = []
        for out in outputs:
            if out:
                for a in (out.get("tom_answers") or []):
                    qid = a.get("id")
                    if qid and qid not in q_ids:
                        q_ids.append(qid)
        if not q_ids:
            q_ids = ["q1"]

        def vote_for(q_id: str) -> str:
            votes = [
                _extract_choice_letter(get_answer_value(out.get("tom_answers"), q_id))
                for out in outputs
                if out and out.get("tom_answers") is not None
            ]
            if not votes:
                return ""
            counter = Counter(votes)
            top = counter.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                return top[0][0]
            tiebreak = state.agent_outputs.get(f"agent{self.tiebreak_agent}")
            if tiebreak:
                tb_val = get_answer_value(tiebreak.get("tom_answers"), q_id)
                return _extract_choice_letter(tb_val) or top[0][0]
            return top[0][0]

        result = ToMAnswers(answers=[{"id": qid, "value": vote_for(qid)} for qid in q_ids])
        logger.info(f"[Debate] Majority vote result: {result}")
        return result

    # ── Flag accumulation ────────────────────────────────────────────────────

    def _accumulate_flags(
        self,
        critiques: dict,
        rebuttal_results: list,
        outputs_before: dict,
        round_num: int,
    ) -> None:
        """Record whether each agent accepted or ignored directed critiques."""
        for agent_id, rebuttal_out in rebuttal_results:
            agent_key = f"agent{agent_id}"
            new_answer = _extract_choice_letter(rebuttal_out.get("my_answer", ""))

            old_tom = (outputs_before.get(agent_key) or {}).get("tom_answers") or []
            old_answer = next(
                (_extract_choice_letter(a.get("value", "")) for a in old_tom if a.get("id") == "q1"),
                "",
            )

            for critic_key, critique_out in critiques.items():
                if critic_key == agent_key:
                    continue
                critique_text = critique_out.get(f"critique_of_{agent_key}", "")
                if not critique_text:
                    continue
                changed = bool(new_answer) and (new_answer != old_answer)
                self.accumulated_flags.append({
                    "round": round_num,
                    "critic": critic_key,
                    "target": agent_key,
                    "flag": "accepted_critique" if changed else "ignored_critique",
                    "old_answer": old_answer,
                    "new_answer": new_answer,
                })

    # ── Agreement check (Python, no LLM) ─────────────────────────────────────

    def _check_agreement(self, state: ToMState) -> tuple:
        """Rule-based consensus check — no LLM call."""
        answer_map = {}
        for agent_key, output in state.agent_outputs.items():
            if not output:
                continue
            for a in (output.get("tom_answers") or []):
                qid = a.get("id")
                val = _extract_choice_letter(a.get("value", ""))
                if qid and val:
                    answer_map.setdefault(qid, {})[agent_key] = val

        agreement = bool(answer_map) and all(
            len(set(votes.values())) == 1
            for votes in answer_map.values()
            if votes
        )
        return agreement, answer_map

    def _extract_answer_from_state(self, state: ToMState) -> ToMAnswers:
        """Extract final answer directly from agent outputs (used when agreement=True)."""
        for output in state.agent_outputs.values():
            if not output:
                continue
            tom_ans = output.get("tom_answers") or []
            if tom_ans:
                return ToMAnswers(answers=[
                    {"id": a["id"], "value": _extract_choice_letter(a["value"])}
                    for a in tom_ans if a.get("id") and a.get("value")
                ])
        return ToMAnswers()
