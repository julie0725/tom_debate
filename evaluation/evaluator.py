"""
evaluator.py
------------
Big-ToM / Hi-ToM 정답과 최종 답변 비교
논문 실험용 정량 평가 스크립트
"""

import csv
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
        q2_correct = self._match(fa.get("q2"), gt.get("q2")) if (gt.get("q2") and fa.get("q2")) else None
        q3_correct = self._match_action(fa.get("q3"), gt.get("q3")) if (gt.get("q3") and fa.get("q3")) else None

        present = [r for r in [q1_correct, q2_correct, q3_correct] if r is not None]
        return {
            "q1_correct": q1_correct,
            "q2_correct": q2_correct,
            "q3_correct": q3_correct,
            "all_correct": all(present) if present else False,
        }

    def evaluate_from_jsonl(self, jsonl_path: str = None, results_file: str = None, output_file: str = "evaluation_summary.json", dataset_name: str = None, condition: str = "PRISM", silent: bool = False) -> dict:
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
        total_elapsed = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0

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
            total_elapsed += r.get("elapsed_sec", 0)
            total_prompt_tokens += r.get("prompt_tokens", 0)
            total_completion_tokens += r.get("completion_tokens", 0)
            total_cost += r.get("estimated_cost_usd", 0)

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
            "avg_elapsed_sec": round(total_elapsed / total, 2) if total_elapsed else None,
            "total_elapsed_sec": round(total_elapsed, 2) if total_elapsed else None,
            "throughput_samples_per_hour": round(3600 / (total_elapsed / total), 1) if total_elapsed else None,
            "total_prompt_tokens": total_prompt_tokens if total_prompt_tokens else None,
            "total_completion_tokens": total_completion_tokens if total_completion_tokens else None,
            "total_cost_usd": round(total_cost, 6) if total_cost else None,
            "avg_cost_per_sample": round(total_cost / total, 6) if total_cost else None,
        }

        out_path = self.output_dir / output_file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if dataset_name:
            summary["dataset_name"] = dataset_name
        logger.info(f"[Evaluator] Summary saved to {out_path}")
        if not silent:
            self._print_final_metrics(summary, dataset_name=dataset_name, condition=condition)
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

    def _print_final_metrics(self, summary: dict, dataset_name: str = None, condition: str = "PRISM") -> None:
        W = 60
        total = summary.get("total") or 0
        trigger_rate = summary.get("debate_trigger_rate") or 0
        total_conflicts = int(round(trigger_rate * total))
        total_rounds = int(round((summary.get("avg_debate_rounds") or 0) * total))
        print("\n" + "=" * W)
        print("  FINAL METRICS")
        print("=" * W)
        print(f"  {'condition':<26}: {condition}")
        print(f"  {'dataset':<26}: {(dataset_name or '').lower()}")
        print(f"  {'total':<26}: {total}")
        print(f"  {'q1_accuracy':<26}: {summary.get('q1_belief_accuracy')}")
        print(f"  {'q2_accuracy':<26}: {summary.get('q2_desire_accuracy')}")
        print(f"  {'q3_accuracy':<26}: {summary.get('q3_action_accuracy')}")
        print(f"  {'joint_accuracy':<26}: {summary.get('joint_accuracy')}")
        print(f"  {'total_conflicts':<26}: {total_conflicts}")
        print(f"  {'total_rounds':<26}: {total_rounds}")
        print(f"  {'conflict_rate':<26}: {trigger_rate}")
        print(f"  {'debate_trigger_rate':<26}: {trigger_rate}")
        print(f"  {'avg_debate_rounds':<26}: {summary.get('avg_debate_rounds')}")
        print(f"  {'majority_vote_rate':<26}: {summary.get('majority_vote_rate')}")
        print(f"  {'avg_elapsed_sec':<26}: {summary.get('avg_elapsed_sec')}")
        print(f"  {'throughput_per_hour':<26}: {summary.get('throughput_samples_per_hour')}")
        print(f"  {'avg_cost_per_sample':<26}: {summary.get('avg_cost_per_sample')}")
        print(f"  {'total_cost_usd':<26}: {summary.get('total_cost_usd')}")
        print("=" * W)

    def save_samples_csv(self, jsonl_path: str, output_path: str, dataset_name: str, system_name: str) -> None:
        path = Path(jsonl_path)
        if not path.exists():
            return
        fieldnames = [
            "dataset_name", "system", "dataset_id",
            "elapsed_sec", "prompt_tokens", "completion_tokens", "estimated_cost_usd",
            "debate_round", "debate_triggered", "majority_vote_applied", "supervisor_used",
            "q1_correct", "q2_correct", "q3_correct", "all_correct",
        ]
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                fa = r.get("final_answer", {})
                gt = r.get("ground_truth")
                if gt:
                    ev = self.evaluate_single(fa, gt)
                    q1, q2, q3 = ev["q1_correct"], ev["q2_correct"], ev["q3_correct"]
                    all_c = ev["all_correct"]
                else:
                    q1 = q2 = q3 = all_c = None
                rows.append({
                    "dataset_name": dataset_name,
                    "system": system_name,
                    "dataset_id": r.get("dataset_id"),
                    "elapsed_sec": r.get("elapsed_sec"),
                    "prompt_tokens": r.get("prompt_tokens"),
                    "completion_tokens": r.get("completion_tokens"),
                    "estimated_cost_usd": r.get("estimated_cost_usd"),
                    "debate_round": r.get("debate_round", 0),
                    "debate_triggered": r.get("debate_triggered", False),
                    "majority_vote_applied": r.get("majority_vote_applied", False),
                    "supervisor_used": r.get("supervisor_used", False),
                    "q1_correct": q1,
                    "q2_correct": q2,
                    "q3_correct": q3,
                    "all_correct": all_c,
                })
        sanitized = [{k: ("" if v is None else v) for k, v in row.items()} for row in rows]
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if Path(output_path).stat().st_size == 0:
                writer.writeheader()
            writer.writerows(sanitized)
