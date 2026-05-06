"""
run_logger.py
-------------
실행 로그 시스템
- Context File 스냅샷 저장 (토론 전/라운드별/후)
- 각 에이전트 출력 상세 로그
- 토론 시 에이전트가 보는 내용 (debate_context) 로그
- 모든 로그를 outputs/logs/<dataset_id>/ 에 저장
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from core.context_file import get_answer_value

logger = logging.getLogger(__name__)


class RunLogger:
    """
    한 샘플 실행의 전체 로그를 관리
    - context_file_*.json : Context File 스냅샷
    - agent_outputs_*.json : 각 에이전트 출력
    - debate_round_*.json  : 토론 라운드별 상세 (에이전트가 보는 내용 포함)
    - summary.json         : 최종 요약
    """

    def __init__(self, output_dir: str, dataset_id: str):
        self.dataset_id = dataset_id or "unknown"
        safe_id = str(self.dataset_id).replace("/", "_").replace(":", "_")
        self.log_dir = Path(output_dir) / "logs" / safe_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for f in self.log_dir.glob("*"):
            if f.is_file():
                f.unlink()
        self.start_time = datetime.now().isoformat()
        self._round_logs = []

    # ── Context File 스냅샷 ───────────────────────────────────

    def log_context_file(self, state_dict: dict, label: str = "initial"):
        """
        Context File 현재 상태를 JSON으로 저장
        label: 'initial', 'after_initial_reasoning', 'debate_round_N', 'final'
        """
        path = self.log_dir / f"context_file_{label}.json"
        # 핵심 필드만 추출하여 가독성 향상
        snapshot = {
            "_label": label,
            "_timestamp": datetime.now().isoformat(),
            "scenario": state_dict.get("scenario", ""),
            "questions": state_dict.get("questions", {}),
            "status": state_dict.get("status", ""),
            "debate_round": state_dict.get("debate_round", 0),
            "debate_triggered": state_dict.get("debate_triggered", False),
            "agent_outputs": state_dict.get("agent_outputs", {}),
            "debate_context": state_dict.get("debate_context", {}),
            "supervisor_correction": state_dict.get("supervisor_correction"),
            "final_answer": state_dict.get("final_answer", {}),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        logger.info(f"[RunLogger] Context file snapshot saved: {path.name}")

    # ── 에이전트 출력 로그 ────────────────────────────────────

    def log_agent_outputs(self, agent_outputs: dict, label: str = "initial"):
        """
        3개 에이전트 출력값을 한눈에 보이는 형식으로 저장
        label: 'initial', 'debate_round_N', 'fresh_reInfer'
        """
        path = self.log_dir / f"agent_outputs_{label}.json"

        formatted = {
            "_label": label,
            "_timestamp": datetime.now().isoformat(),
        }

        for agent_key, output in agent_outputs.items():
            if output is None:
                formatted[agent_key] = None
                continue

            agent_id = output.get("agent_id", agent_key)

            # 에이전트 타입별 핵심 필드 추출
            tom_ans_raw = output.get("tom_answers", [])
            if agent_id == 1:  # Semantic Agent
                formatted[agent_key] = {
                    "_role": "Semantic Agent (진실/거짓 판단)",
                    "character_goal": output.get("character_goal", ""),
                    "truth_judgment": output.get("truth_judgment", {}),
                    "tom_answer": get_answer_value(tom_ans_raw, "q1"),
                    "tom_answers_full": tom_ans_raw,
                }
            elif agent_id == 2:  # Ego Agent
                formatted[agent_key] = {
                    "_role": "Ego Agent (Belief State 추적)",
                    "character_goal": output.get("character_goal", ""),
                    "update_log": output.get("update_log", []),
                    "belief_state": output.get("belief_state", ""),
                    "tom_answer": get_answer_value(tom_ans_raw, "q1"),
                    "tom_answers_full": tom_ans_raw,
                }
            elif agent_id == 3:  # Observer Agent
                formatted[agent_key] = {
                    "_role": "Observer Agent (고차원 추론)",
                    "update_log": output.get("update_log", []),
                    "belief_state": output.get("belief_state", []),
                    "tom_answer": get_answer_value(tom_ans_raw, "q1"),
                    "tom_answers_full": tom_ans_raw,
                }
            else:
                formatted[agent_key] = output

        with open(path, "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)

        # 콘솔에도 간략 출력
        self._print_agent_summary(formatted, label)
        logger.info(f"[RunLogger] Agent outputs saved: {path.name}")

    def _print_agent_summary(self, formatted: dict, label: str):
        """에이전트 출력 요약을 콘솔에 출력"""
        print(f"\n{'─'*60}")
        print(f"  AGENT OUTPUTS [{label}]")
        print(f"{'─'*60}")
        for key in ["agent1", "agent2", "agent3"]:
            out = formatted.get(key)
            if out is None:
                continue
            role = out.get("_role", key)
            tom_ans = out.get("tom_answer", "N/A")
            print(f"  [{key}] {role}")
            print(f"    → tom_answer: {tom_ans}")
            if "tom_answers_full" in out:
                full = out["tom_answers_full"]
                q2 = get_answer_value(full, "q2")
                q3 = get_answer_value(full, "q3")
                if q2:
                    print(f"    → q2_desire : {q2}")
                if q3:
                    print(f"    → q3_action : {q3}")
        print(f"{'─'*60}\n")

    # ── 토론 라운드 로그 ──────────────────────────────────────

    def log_debate_round(self, round_num: int, debate_context: dict, agent_outputs_before: dict, agent_outputs_after: dict, supervisor_result: dict):
        """
        토론 라운드 전체를 기록
        - debate_context: 각 에이전트가 이번 라운드에서 보는 '다른 에이전트의 출력'
        - agent_outputs_before/after: 재추론 전후 비교
        - supervisor_result: 감독관 판단
        """
        path = self.log_dir / f"debate_round_{round_num:02d}.json"

        round_log = {
            "_round": round_num,
            "_timestamp": datetime.now().isoformat(),

            # ① 각 에이전트가 토론 시 보는 내용 (핵심!)
            "what_agents_see": self._format_debate_context(debate_context),

            # ② 재추론 전 상태
            "agent_outputs_before_reInfer": self._extract_tom_answers(agent_outputs_before),

            # ③ 재추론 후 상태
            "agent_outputs_after_reInfer": self._extract_tom_answers(agent_outputs_after),

            # ④ 감독관 판단
            "supervisor_result": supervisor_result,
        }

        self._round_logs.append(round_log)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(round_log, f, ensure_ascii=False, indent=2)

        # 콘솔 출력
        self._print_debate_round(round_log)
        logger.info(f"[RunLogger] Debate round {round_num} saved: {path.name}")

    def _format_debate_context(self, debate_context: dict) -> dict:
        """
        debate_context를 '에이전트가 보는 내용' 형식으로 정리
        """
        if not debate_context:
            return {}

        formatted = {"_round": debate_context.get("round", "?")}
        for agent_key in ["agent1", "agent2", "agent3"]:
            ctx = debate_context.get(agent_key, {})
            other_outputs = ctx.get("other_outputs", {})

            # 각 에이전트가 보는 다른 에이전트의 tom_answer만 추출 (가독성)
            sees = {}
            for other_key, other_out in other_outputs.items():
                if other_out:
                    sees[other_key] = {
                        "tom_answer": get_answer_value(other_out.get("tom_answers", []), "q1") or "?",
                        "reasoning_summary": str(other_out.get("reasoning", ""))[:200],
                        "full_output": other_out,
                    }
            formatted[f"{agent_key}_sees"] = sees

        return formatted

    def _extract_tom_answers(self, agent_outputs: dict) -> dict:
        """에이전트 출력에서 tom_answers만 추출"""
        result = {}
        for key, out in agent_outputs.items():
            if out:
                result[key] = out.get("tom_answers", [])
        return result

    def _print_debate_round(self, round_log: dict):
        """토론 라운드 요약 콘솔 출력"""
        r = round_log["_round"]
        print(f"\n{'═'*60}")
        print(f"  DEBATE ROUND {r}")
        print(f"{'═'*60}")

        print("  [각 에이전트가 보는 다른 에이전트 답변]")
        what_sees = round_log.get("what_agents_see", {})
        for agent_key in ["agent1", "agent2", "agent3"]:
            sees = what_sees.get(f"{agent_key}_sees", {})
            if sees:
                others = {k: v["tom_answer"] for k, v in sees.items()}
                print(f"    {agent_key} sees → {others}")

        print("\n  [재추론 후 답변]")
        after = round_log.get("agent_outputs_after_reInfer", {})
        for key, ans in after.items():
            q1 = get_answer_value(ans, "q1") or "?"
            q2 = get_answer_value(ans, "q2") or "?"
            q3 = get_answer_value(ans, "q3") or "?"
            print(f"    {key}: q1={q1} | q2={q2} | q3={q3}")

        sup = round_log.get("supervisor_result", {})
        agree = sup.get("agreement", "?")
        print(f"\n  [감독관 판단] agreement={agree}")
        if sup.get("supervisor_reasoning"):
            print(f"    reasoning: {sup['supervisor_reasoning'][:200]}")
        print(f"{'═'*60}\n")

    # ── 감독관 수정 로그 ──────────────────────────────────────

    def log_supervisor_correction(self, correction: str):
        """max_rounds 초과 시 감독관 오류 분석 저장"""
        path = self.log_dir / "supervisor_correction.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            f.write(correction)
        print(f"\n[감독관 오류 분석]\n{correction[:400]}...\n")
        logger.info(f"[RunLogger] Supervisor correction saved: {path.name}")

    # ── 최종 요약 ─────────────────────────────────────────────

    def log_final_summary(self, state_dict: dict):
        """실행 완료 후 최종 요약 저장"""
        path = self.log_dir / "summary.json"

        final_answer = state_dict.get("final_answer", {})
        ground_truth = state_dict.get("ground_truth", {})

        # ground_truth가 dataclass일 수 있음
        if hasattr(ground_truth, "__dict__"):
            ground_truth = asdict(ground_truth)

        summary = {
            "dataset_id": self.dataset_id,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "status": state_dict.get("status", ""),
            "debate_triggered": state_dict.get("debate_triggered", False),
            "debate_rounds": state_dict.get("debate_round", 0),
            "majority_vote_applied": state_dict.get("majority_vote_applied", False),
            "final_answer": final_answer if isinstance(final_answer, dict) else asdict(final_answer),
            "ground_truth": ground_truth,
            "log_files": [f.name for f in self.log_dir.glob("*.json")] + [f.name for f in self.log_dir.glob("*.txt")],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE | dataset_id={self.dataset_id}")
        print(f"  status={summary['status']} | debate_rounds={summary['debate_rounds']}")
        fa = summary["final_answer"]
        fa_answers = fa.get("answers", []) if isinstance(fa, dict) else []
        q1 = get_answer_value(fa_answers, "q1")
        q2 = get_answer_value(fa_answers, "q2")
        q3 = get_answer_value(fa_answers, "q3")
        print(f"  final_answer: q1={q1} | q2={q2} | q3={q3}")
        print(f"  logs saved to: {self.log_dir}")
        print(f"{'='*60}\n")

        logger.info(f"[RunLogger] Summary saved: {path}")