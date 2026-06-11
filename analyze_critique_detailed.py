"""
Detailed analysis of critique patterns in correct→wrong flip cases.
Outputs actual critique texts + category classification.
Results written directly to critique_analysis_output.txt (UTF-8).
"""

import json
import re
from pathlib import Path
from collections import defaultdict

LOGS_DIR = Path(r"d:\donghyun\tom_debate\outputs\results_full_system\bigtom\logs")
OUT_PATH  = Path(r"d:\donghyun\tom_debate\critique_analysis_output.txt")

# -------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------

out = None  # file handle, opened in main

def w(text=""):
    out.write(text + "\n")


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_initial_agent_answers(sample_dir):
    """Returns {agent_key: {qid: value}} from agent_outputs_initial.json."""
    data = load_json(sample_dir / "agent_outputs_initial.json")
    if not data:
        return {}
    result = {}
    for agent_key in ["agent1", "agent2", "agent3"]:
        agent_data = data.get(agent_key)
        if not agent_data:
            continue
        per_q = {}
        for qa in agent_data.get("tom_answers_full", []):
            qid = qa.get("id", "")
            val = qa.get("value", "")
            if val and qid:
                per_q[qid] = val
        result[agent_key] = per_q
    return result


def majority_vote(per_agent_answers):
    """Compute majority vote across agents for each question."""
    votes = defaultdict(list)
    for answers in per_agent_answers.values():
        for qid, val in answers.items():
            if val:
                votes[qid].append(val)
    return {qid: max(set(vals), key=vals.count) for qid, vals in votes.items()}


def get_final_and_truth(sample_dir):
    summary = load_json(sample_dir / "summary.json")
    if not summary:
        return {}, {}
    final = {qa["id"]: qa["value"]
             for qa in summary.get("final_answer", {}).get("answers", [])}
    truth = {qa["id"]: qa["value"]
             for qa in summary.get("ground_truth", {}).get("answers", [])}
    return final, truth


def get_critiques_by_round(sample_dir):
    """Extract all critiques from all debate round files."""
    all_critiques = []
    for round_num in range(1, 10):
        path = sample_dir / f"debate_round_{round_num:02d}.json"
        if not path.exists():
            break
        data = load_json(path)
        if not data:
            continue
        for agent_key in ["agent1", "agent2", "agent3"]:
            agent = data.get("what_agents_see", {}).get(agent_key, {})
            for target_key, critique_text in agent.get("critiques_given", {}).items():
                all_critiques.append({
                    "round":          round_num,
                    "from_agent":     agent_key,
                    "to_agent":       target_key,
                    "critique":       critique_text,
                    "critique_answer": agent.get("critique_answer", ""),
                    "rebuttal":       agent.get("rebuttal", ""),
                })
    return all_critiques


# -------------------------------------------------------------------
# Category classifier
# -------------------------------------------------------------------

def classify_critique(text):
    t = text.lower()
    cats = []

    if any(w in t for w in ["witnessed", "saw", "observed", "did not witness",
                              "does not witness", "didn't witness",
                              "did not see", "didn't see", "not present"]):
        cats.append("WITNESS/OBSERVATION")

    if any(w in t for w in ["believes", "belief", "think", "thought", "assumed",
                              "assumption", "still believes", "incorrectly believes"]):
        cats.append("BELIEF_STATE")

    if any(w in t for w in ["desire", "goal", "want", "intention", "intend",
                              "motive", "purpose"]):
        cats.append("DESIRE/GOAL")

    if any(w in t for w in ["knows", "aware", "knowledge", "unaware",
                              "not know", "didn't know", "does not know",
                              "informed", "uninformed"]):
        cats.append("KNOWLEDGE/AWARENESS")

    if re.search(r"\[event\s*\d+\]", t):
        cats.append("EVENT_REFERENCE")

    if any(w in t for w in ["deceiv", "trick", "false belief", "mislead",
                              "swap", "replaced", "substitut",
                              "secretly", "without knowing"]):
        cats.append("DECEPTION/FALSE_BELIEF")

    if any(w in t for w in ["will do", "would do", "action", "behav",
                              "respond", "react", "going to", "likely to",
                              "will not", "would not"]):
        cats.append("ACTION_PREDICTION")

    if any(w in t for w in ["flawed", "incorrect", "error", "flaw", "wrong",
                              "overlooks", "contradict", "incorrect reasoning",
                              "fails to"]):
        cats.append("REASONING_ERROR_CLAIM")

    return cats if cats else ["OTHER"]


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------

def analyze():
    all_samples = sorted(
        [d for d in LOGS_DIR.iterdir() if d.is_dir()],
        key=lambda x: int(x.name)
    )

    flip_cases = []

    for sample_dir in all_samples:
        summary_path = sample_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if not summary or not summary.get("debate_triggered"):
            continue

        final_answers, ground_truth = get_final_and_truth(sample_dir)
        initial_majority = majority_vote(get_initial_agent_answers(sample_dir))

        flipped = []
        for qid, gt in ground_truth.items():
            init  = initial_majority.get(qid)
            final = final_answers.get(qid)
            if init == gt and final and final != gt and final != "unknown":
                flipped.append({"qid": qid, "initial": init, "final": final, "gt": gt})

        if not flipped:
            continue

        sup_path = sample_dir / "supervisor_correction.txt"
        supervisor_text = sup_path.read_text(encoding="utf-8") if sup_path.exists() else ""

        flip_cases.append({
            "sample_id":       sample_dir.name,
            "flipped":         flipped,
            "initial_majority": initial_majority,
            "final_answers":   final_answers,
            "ground_truth":    ground_truth,
            "critiques":       get_critiques_by_round(sample_dir),
            "supervisor_text": supervisor_text,
        })

    # ----------------------------------------------------------------
    # 1. Summary counts
    # ----------------------------------------------------------------
    w(f"Total correct→wrong flip cases: {len(flip_cases)}")
    w()

    # ----------------------------------------------------------------
    # 2. Category distribution
    # ----------------------------------------------------------------
    category_counter = defaultdict(int)
    all_items = []

    for case in flip_cases:
        for c in case["critiques"]:
            cats = classify_critique(c["critique"])
            for cat in cats:
                category_counter[cat] += 1
            all_items.append({**c,
                               "sample_id": case["sample_id"],
                               "flipped":   case["flipped"],
                               "categories": cats})

    w("=" * 70)
    w("CATEGORY DISTRIBUTION (all critiques in flip cases)")
    w("=" * 70)
    for cat, cnt in sorted(category_counter.items(), key=lambda x: -x[1]):
        w(f"  {cat:<35} : {cnt}")
    w()

    # ----------------------------------------------------------------
    # 3. Supervisor "No logical errors" despite flip
    # ----------------------------------------------------------------
    no_error = [c for c in flip_cases
                if "No logical errors found" in c["supervisor_text"]]
    w(f"Supervisor said 'No logical errors found' but flip occurred: {len(no_error)}")
    w()

    # ----------------------------------------------------------------
    # 4. Supervisor texts sample
    # ----------------------------------------------------------------
    w("=" * 70)
    w("SUPERVISOR TEXTS in flip cases (first 15)")
    w("=" * 70)
    for case in flip_cases[:15]:
        w()
        w(f"Sample {case['sample_id']} | flipped={case['flipped']}")
        w(f"  {case['supervisor_text'].strip()[:400]}")

    w()

    # ----------------------------------------------------------------
    # 5. Full critique texts — first 15 flip cases
    # ----------------------------------------------------------------
    w("=" * 70)
    w("ACTUAL CRITIQUE TEXTS — first 15 flip cases")
    w("=" * 70)

    for case in flip_cases[:15]:
        w()
        w("=" * 60)
        w(f"[Sample {case['sample_id']}]  flipped={case['flipped']}")
        w(f"  initial majority : {case['initial_majority']}")
        w(f"  final answers    : {case['final_answers']}")
        w(f"  ground truth     : {case['ground_truth']}")
        w()
        for c in case["critiques"]:
            cats = classify_critique(c["critique"])
            w(f"  Round {c['round']} | {c['from_agent']} → {c['to_agent']} | {cats}")
            w(f"  CRITIQUE : {c['critique']}")
            w(f"  REBUTTAL : {c['rebuttal'][:250] if c['rebuttal'] else '(none)'}")
            w()

    # ----------------------------------------------------------------
    # 6. Examples grouped by category (4 per category)
    # ----------------------------------------------------------------
    w()
    w("=" * 70)
    w("EXAMPLES GROUPED BY DOMINANT CATEGORY (up to 4 each)")
    w("=" * 70)

    target_cats = [
        "WITNESS/OBSERVATION",
        "DECEPTION/FALSE_BELIEF",
        "KNOWLEDGE/AWARENESS",
        "BELIEF_STATE",
        "ACTION_PREDICTION",
        "DESIRE/GOAL",
        "REASONING_ERROR_CLAIM",
    ]
    for tcat in target_cats:
        w()
        w(f"--- {tcat} ---")
        shown = 0
        for item in all_items:
            if tcat in item["categories"] and shown < 4:
                w(f"  [Sample {item['sample_id']} R{item['round']}]")
                w(f"  {item['critique']}")
                w()
                shown += 1
        if shown == 0:
            w("  (no examples)")

    return flip_cases


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        analyze()
    print(f"Written to {OUT_PATH}")
