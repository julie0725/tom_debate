"""
evaluator.py
------------
Big-ToM / Hi-ToM 정답과 최종 답변 비교
논문 실험용 정량 평가 스크립트
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _to_map(answers_dict: dict) -> dict:
    """Convert either new list schema or legacy dict to {q_id: value} map."""
    if answers_dict is None:
        return {}
    answers = answers_dict.get("answers")
    if isinstance(answers, list):
        return {a["id"]: (a.get("value") or "") for a in answers if a.get("id")}
    # Legacy: {"q1_belief": "A", "q2_desire": "B", "q3_action": "..."}
    return {
        "q1": answers_dict.get("q1_belief", ""),
        "q2": answers_dict.get("q2_desire", ""),
        "q3": answers_dict.get("q3_action", ""),
    }


class Evaluator:
    def __init__(self, output_dir: str = "outputs/"):
        self.output_dir = Path(output_dir)
        self.results = []

    def evaluate_single(self, final_answer: dict, ground_truth: dict) -> dict:
        fa = _to_map(final_answer)
        gt = _to_map(ground_truth)

        q1_correct = self._match(fa.get("q1"), gt.get("q1")) if gt.get("q1") else None
        q2_correct = self._match(fa.get("q2"), gt.get("q2")) if gt.get("q2") else None
        q3_correct = self._match_action(fa.get("q3"), gt.get("q3")) if gt.get("q3") else None

        present = [r for r in [q1_correct, q2_correct, q3_correct] if r is not None]
        return {
            "q1_correct": q1_correct,
            "q2_correct": q2_correct,
            "q3_correct": q3_correct,
            "all_correct": all(present) if present else False,
        }

    def evaluate_from_jsonl(self, jsonl_path: str = None, results_file: str = None, output_file: str = "evaluation_summary.json") -> dict:
        """results.jsonl 전체 평가 — 논문 Table 기준 집계"""
        if jsonl_path:
            path = Path(jsonl_path)
        elif results_file:
            path = self.output_dir / results_file
        else:
            path = self.output_dir / "results.jsonl"

        if not path.exists():
            logger.warning(f"No results file at {path}")
            return {}

        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        total = len(records)
        if total == 0:
            return {"total": 0}

        q1_acc, q2_acc, q3_acc, all_acc = 0, 0, 0, 0
        q1_count, q2_count, q3_count = 0, 0, 0
        debate_triggered_count = 0
        majority_vote_count = 0
        total_rounds = 0

        for r in records:
            fa = r.get("final_answer", {})
            gt = r.get("ground_truth", {})
            if gt:
                eval_result = self.evaluate_single(fa, gt)
                if eval_result["q1_correct"] is not None:
                    q1_count += 1
                    q1_acc += int(eval_result["q1_correct"])
                if eval_result["q2_correct"] is not None:
                    q2_count += 1
                    q2_acc += int(eval_result["q2_correct"])
                if eval_result["q3_correct"] is not None:
                    q3_count += 1
                    q3_acc += int(eval_result["q3_correct"])
                all_acc += int(eval_result["all_correct"])

            if r.get("debate_triggered"):
                debate_triggered_count += 1
            if r.get("majority_vote_applied"):
                majority_vote_count += 1
            total_rounds += r.get("debate_round", 0)

        # q1/q2/q3 개별 정확도, 모두 맞춘 비율, 토론 진입률, 다수결 적용률, 평균 토론 라운드 수
        # avg_debate_rounds_among_debated 추가: 토론 진입한 샘플에 한해서 평균 라운드 계산
        summary = {
            "total": total,
            "q1_belief_accuracy": round(q1_acc / q1_count, 4) if q1_count else None,
            "q2_desire_accuracy": round(q2_acc / q2_count, 4) if q2_count else None,
            "q3_action_accuracy": round(q3_acc / q3_count, 4) if q3_count else None,
            "joint_accuracy": round(all_acc / total, 4),
            "debate_trigger_rate": round(debate_triggered_count / total, 4),
            "majority_vote_rate": round(majority_vote_count / total, 4),
            "avg_debate_rounds": round(total_rounds / total, 4),
            "avg_debate_rounds_among_debated": round(total_rounds / debate_triggered_count, 4) if debate_triggered_count else None,
        }

        out_path = self.output_dir / output_file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"[Evaluator] Summary saved to {out_path}")
        self._print_summary(summary)
        return summary

    def _match(self, pred: str, gt: str) -> bool:
        if pred is None or gt is None:
            return False
        if not pred or not gt:
            return False
        return pred.strip().upper() == gt.strip().upper()

    def _match_action(self, pred: str, gt: str) -> bool:
        if pred is None or gt is None:
            return False
        if not pred or not gt:
            return False
        pred_lower = pred.lower()
        gt_lower = gt.lower()
        gt_keywords = [w for w in gt_lower.split() if len(w) > 3]
        if not gt_keywords:
            return pred_lower == gt_lower
        match_count = sum(1 for kw in gt_keywords if kw in pred_lower)
        return match_count / len(gt_keywords) >= 0.5

    def _print_summary(self, summary: dict) -> None:
        def fmt(v):
            return f"{v:.2%}" if v is not None else "N/A"

        print("\n" + "="*50)
        print("  EVALUATION SUMMARY")
        print("="*50)
        print(f"  Total samples       : {summary['total']}")
        print(f"  Q1 Belief accuracy  : {fmt(summary['q1_belief_accuracy'])}")
        print(f"  Q2 Desire accuracy  : {fmt(summary['q2_desire_accuracy'])}")
        print(f"  Q3 Action accuracy  : {fmt(summary['q3_action_accuracy'])}")
        print(f"  Joint accuracy      : {fmt(summary['joint_accuracy'])}")
        print(f"  Debate trigger rate : {summary['debate_trigger_rate']:.2%}")
        print(f"  Majority vote rate  : {summary['majority_vote_rate']:.2%}")
        print(f"  Avg debate rounds   : {summary['avg_debate_rounds']:.2f}")
        among = summary.get("avg_debate_rounds_among_debated") # 추가 :  토론 진입한 샘플에 한해서 평균 라운드 계산
        print(f"  Avg rounds (debated): {f'{among:.2f}' if among is not None else 'N/A'}")
        print("="*50 + "\n")
