"""
Analyze critique patterns in correct→wrong flip cases across debate-triggered samples.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

LOGS_DIR = Path(r"d:\donghyun\tom_debate\outputs\results_full_system\bigtom\logs")

def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def get_initial_answers(sample_dir):
    """Returns {question_id: majority_answer} from agent_outputs_initial.json"""
    data = load_json(sample_dir / "agent_outputs_initial.json")
    if not data:
        return {}
    answers = defaultdict(list)
    for agent_key in ["agent1", "agent2", "agent3"]:
        agent = data.get(agent_key)
        if not agent:
            continue
        for qa in agent.get("tom_answers_full", []):
            qid = qa.get("id", "")
            val = qa.get("value", "")
            if val:
                answers[qid].append(val)
    # majority vote
    majority = {}
    for qid, vals in answers.items():
        if vals:
            majority[qid] = max(set(vals), key=vals.count)
    return majority

def get_final_answers(sample_dir):
    """Returns {question_id: answer} from summary.json"""
    data = load_json(sample_dir / "summary.json")
    if not data:
        return {}, {}
    final = {}
    for qa in data.get("final_answer", {}).get("answers", []):
        final[qa["id"]] = qa["value"]
    ground_truth = {}
    for qa in data.get("ground_truth", {}).get("answers", []):
        ground_truth[qa["id"]] = qa["value"]
    return final, ground_truth

def get_all_critiques(sample_dir):
    """Extract all critique texts from all debate rounds."""
    critiques = []
    for round_num in range(1, 10):
        path = sample_dir / f"debate_round_{round_num:02d}.json"
        if not path.exists():
            break
        data = load_json(path)
        if not data:
            continue
        what_agents_see = data.get("what_agents_see", {})
        for agent_key in ["agent1", "agent2", "agent3"]:
            agent = what_agents_see.get(agent_key, {})
            for critic_key, critique_text in agent.get("critiques_given", {}).items():
                critiques.append({
                    "round": round_num,
                    "from_agent": agent_key,
                    "to_agent": critic_key,
                    "critique": critique_text,
                    "critique_answer": agent.get("critique_answer", ""),
                    "rebuttal": agent.get("rebuttal", ""),
                })
    return critiques

def analyze():
    all_samples = sorted([d for d in LOGS_DIR.iterdir() if d.is_dir()], key=lambda x: int(x.name))

    debate_triggered = []
    correct_to_wrong_flip = []
    no_flip = []

    for sample_dir in all_samples:
        summary_path = sample_dir / "summary.json"
        if not summary_path.exists():
            continue

        summary = load_json(summary_path)
        if not summary:
            continue

        if not summary.get("debate_triggered"):
            continue

        # Get ground truth
        ground_truth = {}
        for qa in summary.get("ground_truth", {}).get("answers", []):
            ground_truth[qa["id"]] = qa["value"]

        # Get final answers
        final_answers = {}
        for qa in summary.get("final_answer", {}).get("answers", []):
            final_answers[qa["id"]] = qa["value"]

        # Get initial answers (agent majority before debate)
        initial_answers = get_initial_answers(sample_dir)

        # Check if any question flipped from correct to wrong
        flipped_questions = []
        for qid, gt in ground_truth.items():
            initial = initial_answers.get(qid)
            final = final_answers.get(qid)
            if initial == gt and final != gt and final and final != "unknown":
                flipped_questions.append({
                    "qid": qid,
                    "initial": initial,
                    "final": final,
                    "ground_truth": gt
                })

        sample_info = {
            "sample_id": sample_dir.name,
            "ground_truth": ground_truth,
            "initial_answers": initial_answers,
            "final_answers": final_answers,
            "flipped_questions": flipped_questions,
        }

        debate_triggered.append(sample_info)

        if flipped_questions:
            # Extract critiques
            critiques = get_all_critiques(sample_dir)
            sample_info["critiques"] = critiques
            correct_to_wrong_flip.append(sample_info)
        else:
            no_flip.append(sample_info)

    print(f"Total debate-triggered samples: {len(debate_triggered)}")
    print(f"Correct→Wrong flip cases: {len(correct_to_wrong_flip)}")
    print(f"No flip (or wrong→correct): {len(no_flip)}")
    print()

    # Print detailed critique examples from flip cases
    print("=" * 80)
    print("CORRECT → WRONG FLIP CASES: CRITIQUE TEXTS")
    print("=" * 80)

    for i, case in enumerate(correct_to_wrong_flip[:30]):
        print(f"\n{'='*60}")
        print(f"Sample {case['sample_id']} | Flipped: {case['flipped_questions']}")
        print(f"Initial (correct): {case['initial_answers']}")
        print(f"Final (wrong):     {case['final_answers']}")
        print(f"Ground Truth:      {case['ground_truth']}")
        print(f"--- Critiques ---")
        for c in case.get("critiques", []):
            print(f"  [Round {c['round']}] {c['from_agent']} → {c['to_agent']}:")
            print(f"    CRITIQUE: {c['critique']}")
            print(f"    REBUTTAL: {c['rebuttal']}")
            print()

    return correct_to_wrong_flip

if __name__ == "__main__":
    cases = analyze()
