"""
evaluator.py
------------
Big-ToM / Hi-ToM 정답과 최종 답변 비교
논문 실험용 정량 평가 스크립트
"""

import json
import logging
from pathlib import Path
from dataclasses import asdict

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, output_dir: str = "outputs/"):
        self.output_dir = Path(output_dir)
        self.results = []

    def evaluate_single(self, final_answer: dict, ground_truth: dict) -> dict:
        """
        단일 샘플 평가
        final_answer, ground_truth: {"q1_belief": ..., "q2_desire": ..., "q3_action": ...}
        """
        q1_correct = self._match(final_answer.get("q1_belief"), ground_truth.get("q1_belief"))
        q2_correct = self._match(final_answer.get("q2_desire"), ground_truth.get("q2_desire"))
        q3_correct = self._match_action(final_answer.get("q3_action"), ground_truth.get("q3_action"))

        return {
            "q1_correct": q1_correct,
            "q2_correct": q2_correct,
            "q3_correct": q3_correct,
            "all_correct": q1_correct and q2_correct and q3_correct
        }

    def evaluate_from_jsonl(self, jsonl_path: str = None, results_file: str = None, output_file: str = "evaluation_summary.json") -> dict:
        """
        results.jsonl 전체 평가
        논문 Table 기준 집계
        """
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
        debate_triggered_count = 0
        majority_vote_count = 0
        total_rounds = 0

        for r in records:
            fa = r.get("final_answer", {})
            gt = r.get("ground_truth", {})
            if gt:
                eval_result = self.evaluate_single(fa, gt)
                q1_acc += int(eval_result["q1_correct"])
                q2_acc += int(eval_result["q2_correct"])
                q3_acc += int(eval_result["q3_correct"])
                all_acc += int(eval_result["all_correct"])

            if r.get("debate_triggered"):
                debate_triggered_count += 1
            if r.get("majority_vote_applied"):
                majority_vote_count += 1
            total_rounds += r.get("debate_round", 0)

        summary = {
            "total": total,
            "q1_belief_accuracy": round(q1_acc / total, 4),
            "q2_desire_accuracy": round(q2_acc / total, 4),
            "q3_action_accuracy": round(q3_acc / total, 4),
            "joint_accuracy": round(all_acc / total, 4),
            "debate_trigger_rate": round(debate_triggered_count / total, 4),
            "majority_vote_rate": round(majority_vote_count / total, 4),
            "avg_debate_rounds": round(total_rounds / total, 4)
        }

        # 결과 저장
        out_path = self.output_dir / output_file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"[Evaluator] Summary saved to {out_path}")
        self._print_summary(summary)
        return summary

    def _match(self, pred: str, gt: str) -> bool:
        """q1, q2: 대소문자 무시 비교"""
        if pred is None or gt is None:
            return False
        return pred.strip().upper() == gt.strip().upper()

    def _match_action(self, pred: str, gt: str) -> bool:
        """
        q3 (Action): 개방형 질문
        현재는 핵심 키워드 포함 여부로 판단
        추후 LLM-based evaluation으로 교체 가능
        """
        if pred is None or gt is None:
            return False
        pred_lower = pred.lower()
        gt_lower = gt.lower()
        # gt의 핵심 단어들이 pred에 포함되는지 확인
        gt_keywords = [w for w in gt_lower.split() if len(w) > 3]
        if not gt_keywords:
            return pred_lower == gt_lower
        match_count = sum(1 for kw in gt_keywords if kw in pred_lower)
        return match_count / len(gt_keywords) >= 0.5

    def _print_summary(self, summary: dict) -> None:
        print("\n" + "="*50)
        print("  EVALUATION SUMMARY")
        print("="*50)
        print(f"  Total samples       : {summary['total']}")
        print(f"  Q1 Belief accuracy  : {summary['q1_belief_accuracy']:.2%}")
        print(f"  Q2 Desire accuracy  : {summary['q2_desire_accuracy']:.2%}")
        print(f"  Q3 Action accuracy  : {summary['q3_action_accuracy']:.2%}")
        print(f"  Joint accuracy      : {summary['joint_accuracy']:.2%}")
        print(f"  Debate trigger rate : {summary['debate_trigger_rate']:.2%}")
        print(f"  Majority vote rate  : {summary['majority_vote_rate']:.2%}")
        print(f"  Avg debate rounds   : {summary['avg_debate_rounds']:.2f}")
        print("="*50 + "\n")
