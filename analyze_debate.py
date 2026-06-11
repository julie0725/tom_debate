"""
Debate Analysis Script
Analyzes flip behavior and supervisor correction effects across 400 samples.
"""

import json
import os
import ast
import re
from collections import defaultdict

LOG_BASE = "d:/donghyun/tom_debate/outputs/results_full_system/bigtom/logs"

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def answers_to_dict(answer_list):
    """[{'id':'q1','value':'A'}, ...] → {'q1':'A', ...}  (last entry wins for dups)"""
    d = {}
    if not isinstance(answer_list, list):
        return d
    for item in answer_list:
        if isinstance(item, dict):
            qid = item.get("id", "")
            val = item.get("value", "")
            if qid and val:           # skip empty-string values
                d[qid] = val
    return d


def parse_sup_correction(txt):
    """
    Parse supervisor_correction.txt.
    Returns dict: {agent_id: {'text': ..., 'has_evidence': bool}}
    """
    result = {}
    # Split by [agentX] blocks
    blocks = re.split(r'\[(agent\d+)\]', txt)
    # blocks[0] = header, then pairs: agent_id, content, agent_id, content ...
    it = iter(blocks[1:])
    for agent_id, content in zip(it, it):
        content = content.strip()
        # Try to parse the dict literal
        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, dict):
                result[agent_id] = parsed
                continue
        except Exception:
            pass
        # Fallback: detect manually
        has_evidence = "'has_evidence': True" in content or '"has_evidence": true' in content
        text_match = re.search(r"'text'\s*:\s*'(.*?)'", content, re.DOTALL)
        if not text_match:
            text_match = re.search(r'"text"\s*:\s*"(.*?)"', content, re.DOTALL)
        text = text_match.group(1) if text_match else content
        result[agent_id] = {"text": text, "has_evidence": has_evidence}
    return result


# ─────────────────────────────────────────────
# Per-sample extraction
# ─────────────────────────────────────────────

def get_initial_answers(sample_dir):
    """Return {agent: {qid: answer}} from agent_outputs_initial.json"""
    p = os.path.join(sample_dir, "agent_outputs_initial.json")
    if not os.path.exists(p):
        return {}
    d = load_json(p)
    result = {}
    for agent in ["agent1", "agent2", "agent3"]:
        if agent not in d:
            continue
        ag_data = d[agent]
        if isinstance(ag_data, dict):
            answers = ag_data.get("tom_answers_full", [])
        else:
            answers = []
        result[agent] = answers_to_dict(answers)
    return result


def get_round_answers(sample_dir, round_num):
    """
    Return {agent: {qid: answer}} AFTER the debate round (from agent_outputs_after_reInfer
    inside debate_round_XX.json).
    Falls back to agent_outputs_debate_round_XX.json → tom_answers_full.
    """
    rstr = f"{round_num:02d}"
    # Primary: debate_round_XX.json → agent_outputs_after_reInfer
    p1 = os.path.join(sample_dir, f"debate_round_{rstr}.json")
    if os.path.exists(p1):
        d = load_json(p1)
        after = d.get("agent_outputs_after_reInfer", {})
        result = {}
        for agent in ["agent1", "agent2", "agent3"]:
            if agent in after:
                result[agent] = answers_to_dict(after[agent])
        if result:
            return result
    # Fallback: agent_outputs_debate_round_XX.json
    p2 = os.path.join(sample_dir, f"agent_outputs_debate_round_{rstr}.json")
    if os.path.exists(p2):
        d = load_json(p2)
        result = {}
        for agent in ["agent1", "agent2", "agent3"]:
            if agent in d and isinstance(d[agent], dict):
                result[agent] = answers_to_dict(d[agent].get("tom_answers_full", []))
        return result
    return {}


def get_ground_truth(sample_dir):
    """Return {qid: answer} from summary.json ground_truth."""
    p = os.path.join(sample_dir, "summary.json")
    if not os.path.exists(p):
        return {}
    d = load_json(p)
    gt_raw = d.get("ground_truth", {})
    if isinstance(gt_raw, dict) and "answers" in gt_raw:
        return answers_to_dict(gt_raw["answers"])
    return {}


def get_final_answer(sample_dir):
    """Return {qid: answer} from summary.json final_answer."""
    p = os.path.join(sample_dir, "summary.json")
    if not os.path.exists(p):
        return {}
    d = load_json(p)
    fa_raw = d.get("final_answer", {})
    if isinstance(fa_raw, dict) and "answers" in fa_raw:
        return answers_to_dict(fa_raw["answers"])
    return {}


# ─────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────

# Counters for flip analysis
# flip_counts[round][agent][direction] = count
flip_counts = defaultdict(lambda: defaultdict(lambda: {"correct_to_wrong": 0, "wrong_to_correct": 0}))

# Per-question flip tracking
qid_flip_counts = defaultdict(lambda: defaultdict(lambda: {"correct_to_wrong": 0, "wrong_to_correct": 0}))

# Round-level totals (ignoring agent)
round_flip_total = defaultdict(lambda: {"correct_to_wrong": 0, "wrong_to_correct": 0})

# Supervisor correction analysis
sup_all_no_error = 0      # all 3 agents have 'No logical errors found.'
sup_has_correction = 0    # at least one agent has real correction
sup_total = 0             # samples with supervisor_correction.txt

# Final accuracy split
acc_no_correction = {"correct": 0, "total": 0}
acc_has_correction = {"correct": 0, "total": 0}

# Per-agent flip total
agent_flip_total = defaultdict(lambda: {"correct_to_wrong": 0, "wrong_to_correct": 0})

skipped = 0
processed = 0

for sid in range(400):
    sample_dir = os.path.join(LOG_BASE, str(sid))
    if not os.path.exists(sample_dir):
        skipped += 1
        continue

    gt = get_ground_truth(sample_dir)
    if not gt:
        skipped += 1
        continue

    fa = get_final_answer(sample_dir)
    processed += 1

    # ── Flip analysis ──────────────────────────────────────────────────────
    # Collect per-agent answer at each checkpoint:
    #   round 0 = initial, round 1/2/3 = after debate round
    initial_answers = get_initial_answers(sample_dir)

    prev_answers = {ag: dict(anss) for ag, anss in initial_answers.items()}

    for rn in [1, 2, 3]:
        round_answers = get_round_answers(sample_dir, rn)
        if not round_answers:
            continue

        for agent in ["agent1", "agent2", "agent3"]:
            prev = prev_answers.get(agent, {})
            curr = round_answers.get(agent, {})
            all_qids = set(prev) | set(curr)

            for qid in all_qids:
                if qid not in gt:
                    continue
                prev_val = prev.get(qid)
                curr_val = curr.get(qid)
                if prev_val is None or curr_val is None:
                    continue
                if prev_val == curr_val:
                    continue
                # A flip happened
                prev_correct = prev_val == gt[qid]
                curr_correct = curr_val == gt[qid]
                if prev_correct and not curr_correct:
                    flip_counts[rn][agent]["correct_to_wrong"] += 1
                    round_flip_total[rn]["correct_to_wrong"] += 1
                    agent_flip_total[agent]["correct_to_wrong"] += 1
                    qid_flip_counts[rn][qid]["correct_to_wrong"] += 1
                elif not prev_correct and curr_correct:
                    flip_counts[rn][agent]["wrong_to_correct"] += 1
                    round_flip_total[rn]["wrong_to_correct"] += 1
                    agent_flip_total[agent]["wrong_to_correct"] += 1
                    qid_flip_counts[rn][qid]["wrong_to_correct"] += 1

            # Update prev for next round
            # Merge: keep previous answers for qids not updated this round
            merged = dict(prev)
            merged.update(curr)
            prev_answers[agent] = merged

    # ── Supervisor correction analysis ─────────────────────────────────────
    sup_path = os.path.join(sample_dir, "supervisor_correction.txt")
    if os.path.exists(sup_path):
        sup_total += 1
        try:
            with open(sup_path, encoding="utf-8") as f:
                sup_txt = f.read()
            parsed = parse_sup_correction(sup_txt)
        except Exception:
            parsed = {}

        # Check if all 3 agents have "No logical errors found."
        no_error_count = sum(
            1 for ag in ["agent1", "agent2", "agent3"]
            if parsed.get(ag, {}).get("has_evidence", False) is False
            and "No logical errors found." in parsed.get(ag, {}).get("text", "")
        )
        all_no_error = (no_error_count == 3 and len(parsed) == 3)
        has_correction = any(
            parsed.get(ag, {}).get("has_evidence", False) is True
            for ag in ["agent1", "agent2", "agent3"]
        )

        if all_no_error:
            sup_all_no_error += 1
        if has_correction:
            sup_has_correction += 1

        # Final accuracy for this sample
        correct_count = sum(1 for qid, ans in fa.items() if gt.get(qid) == ans)
        total_q = len(gt)
        is_fully_correct = (correct_count == total_q) if total_q > 0 else False

        if has_correction:
            acc_has_correction["correct"] += correct_count
            acc_has_correction["total"] += total_q
        else:
            acc_no_correction["correct"] += correct_count
            acc_no_correction["total"] += total_q


# ─────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────

SEP = "=" * 60

print(SEP)
print("DEBATE FLIP ANALYSIS")
print(SEP)
print(f"Processed samples: {processed} / 400  (skipped: {skipped})")
print()

print("[ Flip counts per round (all agents combined) ]")
print(f"{'Round':<8} {'Correct→Wrong':>15} {'Wrong→Correct':>15} {'Net change':>12}")
for rn in [1, 2, 3]:
    c2w = round_flip_total[rn]["correct_to_wrong"]
    w2c = round_flip_total[rn]["wrong_to_correct"]
    net = w2c - c2w
    print(f"Round {rn}  {c2w:>15,}  {w2c:>15,}  {net:>+12,}")
print()

total_c2w = sum(round_flip_total[r]["correct_to_wrong"] for r in [1,2,3])
total_w2c = sum(round_flip_total[r]["wrong_to_correct"] for r in [1,2,3])
print(f"TOTAL      {total_c2w:>15,}  {total_w2c:>15,}  {total_w2c - total_c2w:>+12,}")
print()

# Which round has most flips
max_c2w_round = max([1,2,3], key=lambda r: round_flip_total[r]["correct_to_wrong"])
max_w2c_round = max([1,2,3], key=lambda r: round_flip_total[r]["wrong_to_correct"])
max_total_round = max([1,2,3], key=lambda r: (
    round_flip_total[r]["correct_to_wrong"] + round_flip_total[r]["wrong_to_correct"]
))
print(f"Round with most Correct→Wrong flips : Round {max_c2w_round} ({round_flip_total[max_c2w_round]['correct_to_wrong']:,})")
print(f"Round with most Wrong→Correct flips : Round {max_w2c_round} ({round_flip_total[max_w2c_round]['wrong_to_correct']:,})")
print(f"Round with most total flips          : Round {max_total_round} ({round_flip_total[max_total_round]['correct_to_wrong'] + round_flip_total[max_total_round]['wrong_to_correct']:,})")
print()

print("[ Flip counts per agent (all rounds combined) ]")
print(f"{'Agent':<10} {'Correct→Wrong':>15} {'Wrong→Correct':>15} {'Net':>10}")
for ag in ["agent1", "agent2", "agent3"]:
    c2w = agent_flip_total[ag]["correct_to_wrong"]
    w2c = agent_flip_total[ag]["wrong_to_correct"]
    net = w2c - c2w
    print(f"{ag:<10} {c2w:>15,}  {w2c:>15,}  {net:>+10,}")
print()

print("[ Flip counts per agent per round ]")
header = f"{'':12}"
for rn in [1,2,3]:
    header += f"  R{rn} C→W   R{rn} W→C"
print(header)
for ag in ["agent1", "agent2", "agent3"]:
    row = f"{ag:<12}"
    for rn in [1,2,3]:
        c2w = flip_counts[rn][ag]["correct_to_wrong"]
        w2c = flip_counts[rn][ag]["wrong_to_correct"]
        row += f"  {c2w:>7}  {w2c:>7}"
    print(row)
print()

print("[ Flip counts by question ID (all agents, all rounds) ]")
all_qids_seen = set()
for rn in [1,2,3]:
    all_qids_seen |= set(qid_flip_counts[rn].keys())
for qid in sorted(all_qids_seen):
    c2w = sum(qid_flip_counts[rn][qid]["correct_to_wrong"] for rn in [1,2,3])
    w2c = sum(qid_flip_counts[rn][qid]["wrong_to_correct"] for rn in [1,2,3])
    print(f"  {qid}: Correct→Wrong = {c2w:,}, Wrong→Correct = {w2c:,}")
print()

print(SEP)
print("SUPERVISOR CORRECTION ANALYSIS")
print(SEP)
print(f"Samples with supervisor_correction.txt : {sup_total}")
print()
pct_no_error = 100 * sup_all_no_error / sup_total if sup_total else 0
pct_correction = 100 * sup_has_correction / sup_total if sup_total else 0
other = sup_total - sup_all_no_error - sup_has_correction
pct_other = 100 * other / sup_total if sup_total else 0

print(f"All 3 agents 'No logical errors found.' : {sup_all_no_error:>4}  ({pct_no_error:.1f}%)")
print(f"At least one agent has real correction  : {sup_has_correction:>4}  ({pct_correction:.1f}%)")
print(f"Other (partial no-error, missing data)  : {other:>4}  ({pct_other:.1f}%)")
print()

acc_no = 100 * acc_no_correction["correct"] / acc_no_correction["total"] if acc_no_correction["total"] else 0
acc_yes = 100 * acc_has_correction["correct"] / acc_has_correction["total"] if acc_has_correction["total"] else 0
print("[ Final answer accuracy by correction status ]")
print(f"No correction (all 'No logical errors found.')")
print(f"  Questions correct : {acc_no_correction['correct']:,} / {acc_no_correction['total']:,}  ({acc_no:.2f}%)")
print(f"Has correction (at least one agent flagged)")
print(f"  Questions correct : {acc_has_correction['correct']:,} / {acc_has_correction['total']:,}  ({acc_yes:.2f}%)")
print(f"  Accuracy difference : {acc_yes - acc_no:+.2f} pp (correction - no-correction)")
print()
print(SEP)
