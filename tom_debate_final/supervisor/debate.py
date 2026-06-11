"""
debate.py
"""
import asyncio
import copy
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from core.context_file import ToMState, ToMAnswers, get_answer_value
from core.message_pool import MessagePool
from core.llm_client import call_llm


logger = logging.getLogger(__name__)

_AGENT_ROLES = {
    1: "You are Agent1 (Semantic Agent). Your role is to analyze context, judge which events are true or false, and infer character goals from observable facts.",
    2: "You are Agent2 (Ego Agent). Your role is to track each character's belief state step by step, following their epistemic access through the timeline.",
    3: "You are Agent3 (Observer Agent). Your role is to reason about higher-order beliefs — what characters believe about what other characters believe.",
}


def _extract_choice_letter(text: str) -> str:
    if not text:
        return ""
    m = re.match(r'^([A-Z])(?:\.|\s|$)', text.strip())
    return m.group(1) if m else text.strip()


def _evidence_correction_text(correction: dict, agent_key: str) -> str:
    """Return evidence-backed correction text for agent_key, or empty string."""
    if not correction:
        return ""
    val = correction.get(agent_key, "")
    if isinstance(val, dict):
        return val.get("text", "") if val.get("has_evidence") else ""
    return str(val)  # backward compat: old-style plain string


class DebateManager:
    def __init__(
        self,
        agents: dict,
        max_rounds: int,
        tiebreak_agent: int = 3,
        client=None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2000,
        temperature: float = 0.0,
        event_callback=None,
    ):
        self.agents = agents
        self.max_rounds = max_rounds
        self.tiebreak_agent = tiebreak_agent  # kept for interface compat, unused in P6
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._critique_prompt = self._load_prompt("debate_critique_prompt.txt")
        self._rebuttal_prompt = self._load_prompt("debate_rebuttal_prompt.txt")
        self._persona_prompts = {
            i: self._load_prompt(f"agent{i}_persona_prompt.txt") for i in [1, 2, 3]
        }
        self.accumulated_flags: list = []
        self.candidate_pool: dict[str, set] = {}  # P4: qid → set of distinct answer letters
        self.event_callback = event_callback or (lambda event, data: None)

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
        self.candidate_pool = {}

        # P4: seed candidate pool from initial outputs
        state = pool.get_state()
        self._accumulate_candidates(state.initial_agent_outputs or state.agent_outputs)

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"[Debate] Round {round_num} / {self.max_rounds}")
            pool.update_debate_round(round_num)

            state = pool.get_state()
            outputs_before = copy.deepcopy(state.agent_outputs)

            critiques = await self._run_critique_phase(pool, round_num, run_logger)
            self.event_callback("critique", {
                "round": round_num,
                "critiques": {
                    src: {tgt: out.get(f"critique_of_{tgt}", "")
                          for tgt in [f"agent{i}" for i in self.agents] if tgt != src}
                    for src, out in critiques.items()
                },
                "answers_before": {
                    f"agent{aid}": [{"id": a.get("id"), "value": a.get("value")}
                                    for a in (outputs_before.get(f"agent{aid}") or {}).get("tom_answers", [])]
                    for aid in self.agents
                },
            })
            self._print_critique_phase(round_num, critiques)

            rebuttal_results = await self._run_rebuttal_phase(pool, round_num, critiques, run_logger)
            state = pool.get_state()

            # P4: accumulate new answers into candidate pool after each rebuttal
            self._accumulate_candidates(state.agent_outputs)

            self.event_callback("rebuttal", {
                "round": round_num,
                "answers_after": {
                    f"agent{aid}": [{"id": a.get("id"), "value": a.get("value")}
                                    for a in (state.agent_outputs.get(f"agent{aid}") or {}).get("tom_answers", [])]
                    for aid in self.agents
                },
                "rebuttals": {f"agent{aid}": out.get("rebuttal", "") for aid, out in rebuttal_results},
            })
            agreement, answer_map = self._check_agreement(state)
            self.event_callback("consensus", {"round": round_num, "agreement": agreement, "answer_map": answer_map})
            self._print_rebuttal_phase(round_num, rebuttal_results, state.agent_outputs)
            self._accumulate_flags(critiques, rebuttal_results, outputs_before, round_num, state)

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

        logger.info("[Debate] Max rounds reached. Running candidate selection (P4)...")

        # P3: get evidence-gated correction
        correction = {}
        if supervisor_correction_fn:
            state = pool.get_state()
            correction = supervisor_correction_fn(state, self.accumulated_flags)
            pool.update_supervisor_correction(correction)
            if run_logger:
                run_logger.log_supervisor_correction(correction)

        # P4: select from candidate pool instead of free re-reasoning
        pool.update_debate_context({})
        await self._select_from_candidates(pool, correction)
        state = pool.get_state()
        agreement, answer_map = self._check_agreement(state)

        if run_logger:
            run_logger.log_agent_outputs(state.agent_outputs, label="candidate_selection")
            run_logger.log_context_file(asdict(state), label="after_candidate_selection")

        if agreement:
            logger.info("[Debate] Consensus reached after candidate selection.")
            candidate = self._extract_answer_from_state(state)
            return self._gated_finalize(candidate, state)  # P1: gate before returning

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

        # Blind critique: hide tom_answers to prevent sycophancy
        blind_outputs = {
            ak: {k: v for k, v in (out or {}).items() if k != "tom_answers"}
            for ak, out in state_dict["agent_outputs"].items()
        }

        # Targeted critique: focus on contested questions only
        _, answer_map = self._check_agreement(state)
        agreed_qs = [qid for qid, votes in answer_map.items() if len(set(votes.values())) == 1]
        contested_qs = [qid for qid, votes in answer_map.items() if len(set(votes.values())) > 1]
        if agreed_qs and contested_qs:
            focus_hint = (
                f"\nFocus your critique ONLY on {', '.join(contested_qs)} — "
                f"{', '.join(agreed_qs)} already has consensus and does not need critique."
            )
        else:
            focus_hint = ""

        async def critique_one(agent_id: int) -> tuple:
            agent_key = f"agent{agent_id}"
            user_content = (
                f"{_AGENT_ROLES.get(agent_id, f'You are agent{agent_id}.')}\n\n"
                f"Scenario:\n{state_dict['scenario']}\n\n"
                f"Questions:\n{json.dumps(state_dict['questions'], ensure_ascii=False)}\n\n"
                f"All agents' current reasoning (final answers hidden):\n"
                f"{json.dumps(blind_outputs, ensure_ascii=False, indent=2)}\n\n"
                f"Critique the reasoning of every agent EXCEPT yourself. "
                f"Cite specific event numbers and story sentences as evidence. "
                f"Set critique_of_{agent_key} to empty string."
                f"{focus_hint}"
            )
            persona = self._persona_prompts.get(agent_id, "")
            critique_system = (persona + "\n\n" + self._critique_prompt) if persona else self._critique_prompt
            raw = await asyncio.get_event_loop().run_in_executor(
                None, call_llm,
                self.client, self.model, critique_system, user_content, self.max_tokens, self.temperature,
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
    ) -> list:
        """Each agent rebuts all critiques directed at it and updates its answer."""
        state = pool.get_state()
        state_dict = asdict(state)
        agent_ids = list(self.agents.keys())

        async def rebuttal_one(agent_id: int) -> tuple:
            agent_key = f"agent{agent_id}"

            incoming = {
                critic_key: c_out.get(f"critique_of_{agent_key}", "")
                for critic_key, c_out in critiques.items()
                if critic_key != agent_key and c_out.get(f"critique_of_{agent_key}")
            }
            user_content = (
                f"{_AGENT_ROLES.get(agent_id, f'You are agent{agent_id}.')}\n\n"
                f"Scenario:\n{state_dict['scenario']}\n\n"
                f"Questions:\n{json.dumps(state_dict['questions'], ensure_ascii=False)}\n\n"
                f"Your current output:\n"
                f"{json.dumps(state_dict['agent_outputs'].get(agent_key, {}), ensure_ascii=False, indent=2)}\n\n"
                f"Critiques directed at you:\n"
                f"{json.dumps(incoming, ensure_ascii=False, indent=2)}"
            )
            persona = self._persona_prompts.get(agent_id, "")
            rebuttal_system = (persona + "\n\n" + self._rebuttal_prompt) if persona else self._rebuttal_prompt
            raw = await asyncio.get_event_loop().run_in_executor(
                None, call_llm,
                self.client, self.model, rebuttal_system, user_content, self.max_tokens, self.temperature,
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

        # Merge revised answers back into each agent's tom_answers (all questions)
        for agent_id, rebuttal_out in results:
            agent_key = f"agent{agent_id}"
            my_answers = rebuttal_out.get("my_answers") or []
            if not my_answers:
                letter = _extract_choice_letter(rebuttal_out.get("my_answer", ""))
                if letter:
                    my_answers = [{"id": "q1", "value": letter}]
            if my_answers:
                current = dict(state.agent_outputs.get(agent_key) or {})
                tom = list(current.get("tom_answers") or [])
                tom_map = {e.get("id"): e for e in tom}
                for new_ans in my_answers:
                    qid = new_ans.get("id")
                    val = _extract_choice_letter(new_ans.get("value", ""))
                    if qid and val:
                        if qid in tom_map:
                            tom_map[qid]["value"] = val
                        else:
                            tom_map[qid] = {"id": qid, "value": val}
                pool.update_agent_output(agent_id, {**current, "tom_answers": list(tom_map.values())})

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
    def _print_rebuttal_phase(round_num: int, rebuttal_results, agent_outputs: dict = None) -> None:
        W = 40
        print(f"\n{'=' * W}")
        print(f"[Round {round_num}] REBUTTAL PHASE")
        print(f"{'=' * W}")
        for agent_id, r_out in sorted(rebuttal_results, key=lambda x: x[0]):
            rebuttal_text = str(r_out.get("rebuttal", ""))
            agent_key = f"agent{agent_id}"
            merged = (agent_outputs or {}).get(agent_key) or {}
            answer = get_answer_value(merged.get("tom_answers"), "q1") or "?"
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

    # ── P4: Candidate pool ────────────────────────────────────────────────────

    def _accumulate_candidates(self, agent_outputs: dict) -> None:
        """Collect all distinct answer letters from agent_outputs into candidate_pool."""
        for out in (agent_outputs or {}).values():
            for a in (out or {}).get("tom_answers", []):
                qid = a.get("id")
                val = _extract_choice_letter(a.get("value", ""))
                if qid and val:
                    self.candidate_pool.setdefault(qid, set()).add(val)

    # ── P4: Candidate selection (replaces free re-reasoning) ─────────────────

    async def _select_from_candidates(self, pool: MessagePool, correction: dict) -> None:
        """Each agent selects from accumulated candidate answers instead of generating freely."""
        state = pool.get_state()
        state_dict = asdict(state)
        base_outputs = state.initial_agent_outputs or state.agent_outputs or {}

        qid_candidates: dict[str, list] = {
            qid: sorted(cands)
            for qid, cands in self.candidate_pool.items()
            if cands
        }

        # If every question already has only one candidate — skip LLM entirely
        if all(len(v) == 1 for v in qid_candidates.values()):
            for agent_id in self.agents:
                ak = f"agent{agent_id}"
                base = dict(base_outputs.get(ak) or {})
                tom_map = {a["id"]: a for a in (base.get("tom_answers") or [])}
                for qid, cands in qid_candidates.items():
                    tom_map[qid] = {"id": qid, "value": cands[0]}
                pool.update_agent_output(agent_id, {**base, "tom_answers": list(tom_map.values())})
            logger.info("[Debate/P4] Single candidate per question — skipped LLM selection.")
            return

        selection_system = (
            "You select the most logically consistent answer from a fixed candidate set. "
            "Output JSON only: {\"selections\": [{\"id\": \"q1\", \"value\": \"<letter>\"}, ...]}\n"
            "You MUST choose a value from the provided candidates. Do not invent new values."
        )

        async def select_one(agent_id: int) -> tuple:
            agent_key = f"agent{agent_id}"
            corr_text = _evidence_correction_text(correction, agent_key)
            base_out = base_outputs.get(agent_key) or {}

            user_content = (
                f"{_AGENT_ROLES.get(agent_id, '')}\n\n"
                f"Scenario:\n{state_dict['scenario']}\n\n"
                f"Questions:\n{json.dumps(state_dict['questions'], ensure_ascii=False)}\n\n"
                f"Your original reasoning:\n{base_out.get('reasoning', '')}\n\n"
                + (f"Supervisor correction (evidence-backed):\n{corr_text}\n\n" if corr_text else "")
                + f"Candidate answers (choose one per question — no other values allowed):\n"
                f"{json.dumps(qid_candidates, ensure_ascii=False, indent=2)}\n\n"
                f"Select the most logically consistent answer for each question."
            )

            raw = await asyncio.get_event_loop().run_in_executor(
                None, call_llm,
                self.client, self.model, selection_system, user_content, 512, self.temperature,
            )
            cleaned = re.sub(r"```json|```", "", raw).strip()
            try:
                parsed = json.loads(cleaned)
                selections = parsed.get("selections") or []
            except json.JSONDecodeError:
                logger.warning(f"[Debate/P4] Selection parse error (agent{agent_id}): {raw[:80]}")
                selections = []

            current = dict(base_out)
            tom_map = {a["id"]: a for a in (current.get("tom_answers") or [])}
            for sel in selections:
                qid = sel.get("id")
                val = _extract_choice_letter(sel.get("value", ""))
                allowed = qid_candidates.get(qid, [])
                if qid and val and (not allowed or val in allowed):
                    tom_map[qid] = {"id": qid, "value": val}

            return agent_id, {**current, "tom_answers": list(tom_map.values())}

        results = await asyncio.gather(*[select_one(aid) for aid in self.agents])
        for agent_id, output in results:
            pool.update_agent_output(agent_id, output)

    # ── P1: Majority overthrow gate ───────────────────────────────────────────

    def _gated_finalize(self, candidate: ToMAnswers, state: ToMState) -> ToMAnswers:
        """Revert candidate to initial majority if no event-cited evidence supports the change."""
        initial_majority = self._get_initial_majority(
            state.initial_agent_outputs or state.agent_outputs
        )

        gated = []
        for a in candidate.answers:
            qid = a.get("id")
            cand_val = a.get("value")
            init_val = initial_majority.get(qid)

            if not init_val or not cand_val or cand_val == init_val:
                gated.append(a)
                continue

            # Override attempt — require concrete event evidence from a supporting agent
            evidence = any(
                _extract_choice_letter(
                    get_answer_value((out or {}).get("tom_answers"), qid)
                ) == cand_val
                and self._has_event_reference((out or {}).get("reasoning", "") or "")
                for out in state.agent_outputs.values()
                if out
            )

            if evidence:
                gated.append(a)
            else:
                logger.info(
                    f"[Gate/P1] {qid}: candidate={cand_val} overrides majority={init_val} "
                    f"without event evidence → reverting"
                )
                gated.append({"id": qid, "value": init_val})

        return ToMAnswers(answers=gated)

    @staticmethod
    def _has_event_reference(text: str) -> bool:
        return bool(re.search(r'event\s*#?\d+|step\s*\d+|\(\d+\)', text, re.IGNORECASE))

    @staticmethod
    def _get_initial_majority(agent_outputs: dict) -> dict:
        """Simple plurality per question from given outputs."""
        per_qid: dict[str, dict] = {}
        for out in (agent_outputs or {}).values():
            for a in (out or {}).get("tom_answers", []):
                qid = a.get("id")
                val = _extract_choice_letter(a.get("value", ""))
                if qid and val:
                    per_qid.setdefault(qid, {})
                    per_qid[qid][val] = per_qid[qid].get(val, 0) + 1
        return {
            qid: max(counts, key=counts.get)
            for qid, counts in per_qid.items()
            if counts
        }

    # ── P6: Majority vote (deadlock-only weighting) ───────────────────────────

    def _majority_vote(self, state: ToMState) -> ToMAnswers:
        q_ids: list[str] = []
        for out in state.agent_outputs.values():
            for a in (out or {}).get("tom_answers", []):
                qid = a.get("id")
                if qid and qid not in q_ids:
                    q_ids.append(qid)
        if not q_ids:
            q_ids = ["q1"]

        # Stability prior from debate history: fewer flips = higher reliability
        flip_counts: dict[str, int] = {}
        for flag in self.accumulated_flags:
            if flag["flag"] == "accepted_critique":
                ak = flag["target"]
                flip_counts[ak] = flip_counts.get(ak, 0) + 1

        def vote_for(q_id: str) -> str:
            counts: dict[str, int] = {}
            for i in [1, 2, 3]:
                ak = f"agent{i}"
                out = state.agent_outputs.get(ak)
                val = _extract_choice_letter(
                    get_answer_value((out or {}).get("tom_answers"), q_id)
                )
                if val:
                    counts[val] = counts.get(val, 0) + 1

            if not counts:
                return "unknown"

            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            # Clear majority wins unconditionally — no weighting applied
            if len(top) == 1 or top[0][1] > top[1][1]:
                return top[0][0]

            # True tie only: break with stability prior
            weighted: dict[str, float] = {}
            for i in [1, 2, 3]:
                ak = f"agent{i}"
                out = state.agent_outputs.get(ak)
                val = _extract_choice_letter(
                    get_answer_value((out or {}).get("tom_answers"), q_id)
                )
                if val:
                    stability = 1.0 / (1 + flip_counts.get(ak, 0))
                    weighted[val] = weighted.get(val, 0.0) + stability

            return max(weighted, key=weighted.get) if weighted else top[0][0]

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
        state_after: ToMState = None,
    ) -> None:
        """Record whether each agent accepted or ignored directed critiques."""
        for agent_id, _ in rebuttal_results:
            agent_key = f"agent{agent_id}"
            merged_output = (state_after.agent_outputs.get(agent_key) if state_after else None) or {}
            new_answer = _extract_choice_letter(
                get_answer_value(merged_output.get("tom_answers"), "q1")
            )

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
        """Merge answers across all agents — first valid value wins per question ID."""
        qid_to_value: dict[str, str] = {}
        for output in state.agent_outputs.values():
            if not output:
                continue
            for a in (output.get("tom_answers") or []):
                qid = a.get("id")
                val = _extract_choice_letter(a.get("value", ""))
                if qid and val and qid not in qid_to_value:
                    qid_to_value[qid] = val
        if qid_to_value:
            return ToMAnswers(answers=[{"id": qid, "value": val} for qid, val in qid_to_value.items()])
        return ToMAnswers()
