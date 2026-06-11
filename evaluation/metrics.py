"""
metrics.py
----------
에이전트 간 의견 충돌(초기 추론 단계) 계산 유틸
"""

import re
from typing import Optional


def extract_choice_letter(text: str) -> str:
    if not text:
        return ""
    m = re.match(r"^([A-Z])(?:\.|\s|$)", str(text).strip())
    return m.group(1) if m else str(text).strip().upper()


def agent_answer_map(agent_outputs: dict, question_ids: Optional[list] = None) -> dict:
    """{agent_key: {q_id: normalized_label}} for present agents."""
    table = {}
    for agent_key, output in (agent_outputs or {}).items():
        if not output:
            continue
        tom_ans = output.get("tom_answers", [])
        if isinstance(tom_ans, list):
            table[agent_key] = {
                a.get("id"): extract_choice_letter(a.get("value", ""))
                for a in tom_ans
                if a.get("id")
            }
        elif isinstance(tom_ans, dict):
            table[agent_key] = {
                "q1": extract_choice_letter(tom_ans.get("q1_belief", ""))
            }
    if question_ids:
        for key in table:
            table[key] = {qid: table[key].get(qid, "") for qid in question_ids}
    return table


def count_conflicts(
    agent_outputs: dict,
    question_ids: Optional[list] = None,
    active_agents: Optional[list] = None,
) -> dict:
    """
    초기(또는 지정) agent_outputs에서 충돌 집계.

    Returns:
        conflict_count: 의견이 갈린 question 수 (해당 질문에서 2명 이상 서로 다른 답)
        had_conflict: conflict_count > 0
        per_question: {q_id: bool}
    """
    table = agent_answer_map(agent_outputs, question_ids)
    keys = active_agents or sorted(table.keys())
    keys = [k if k.startswith("agent") else f"agent{k}" for k in keys]
    keys = [k for k in keys if k in table and table[k]]

    if not question_ids:
        question_ids = sorted(
            {qid for m in table.values() for qid in m.keys()}
        )

    per_question = {}
    conflict_count = 0
    for qid in question_ids:
        labels = [
            table[k].get(qid, "")
            for k in keys
            if table.get(k, {}).get(qid, "") is not None
        ]
        labels = [l for l in labels if l]
        unique = set(labels)
        disagree = len(labels) >= 2 and len(unique) > 1
        per_question[qid] = disagree
        if disagree:
            conflict_count += 1

    return {
        "conflict_count": conflict_count,
        "had_conflict": conflict_count > 0,
        "per_question": per_question,
        "answer_table": table,
    }
