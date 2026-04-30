"""
preprocess.py
-------------
Big-ToM CSV → Hi-ToM 통일 형식 변환기

[핵심 목표]
Big-ToM의 자연어 문단 시나리오를 Hi-ToM의 번호 붙은 이벤트 로그 형식으로 변환.
에이전트들이 이벤트 번호 기준으로 추론할 수 있도록.

[Big-ToM 원본 구조]
Story (5문장 자연어):
  문장1: 배경   - "Noor is working as a barista..."
  문장2: 목표   - "Noor wants to make a cappuccino..."
  문장3: 초기행동 - "Noor grabs a milk pitcher..."
  문장4: 초기belief - "Noor believes the pitcher contains oatmilk."
  문장5: 사건발생 - "A coworker swaps the oat milk..."
Aware: "Noor sees her coworker swapping the milk."
Not Aware: "Noor does not see her coworker swapping the milk."

[Hi-ToM 목표 형식]
1 Noor is working as a barista at a busy coffee shop.
2 Noor wants to make a delicious cappuccino for a customer who asked for oat milk.
3 Noor grabbed a milk pitcher and filled it with oat milk.
4 Noor believes the milk pitcher contains oat milk.
5 A coworker swapped the oat milk in the pitcher with almond milk.
6 Noor saw her coworker swapping the milk.       ← true_belief
6 Noor did not see her coworker swapping the milk. ← false_belief

question_order 매핑:
  1 = belief 질문  (1차 ToM)
  2 = desire 질문  (2차 ToM)
  3 = action 질문  (3차 ToM)
"""

import json
import csv
import re
import argparse
from pathlib import Path


def split_sentences(text: str) -> list:
    """마침표/느낌표/물음표 기준으로 문장 분리"""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


def extract_character_name(sentence: str) -> str:
    """첫 문장에서 주인공 이름 추출"""
    pronouns = {"he", "she", "they", "it", "his", "her", "their", "the", "a", "an"}
    words = sentence.split()
    if words and words[0].lower() not in pronouns:
        name = re.match(r'^([A-Z][a-z]+)', words[0])
        if name:
            return name.group(1)
    return ""


def bigtom_to_hitom_story(row: list, condition: str) -> tuple:
    """
    Big-ToM 1행 + 조건 → Hi-ToM 번호 이벤트 로그 형식 story

    Big-ToM Story 5문장 구조:
      1 = 배경 (인물 소개)
      2 = 목표 (wants to...)
      3 = 초기 행동
      4 = 초기 belief (believes that...)
      5 = 사건 발생 (belief 변화 유발)
      6 = 목격 여부 (condition에 따라 분기)

    반환: (story_text, character_name, event_count)
    """
    story_raw = row[0].strip()
    aware_event = row[1].strip()
    not_aware_event = row[2].strip()

    sentences = split_sentences(story_raw)
    if len(sentences) != 5:
        raise ValueError(f"Expected 5 sentences, got {len(sentences)}")

    character = extract_character_name(sentences[0])

    # 1~5번: 원문 문장 그대로 번호만 붙임
    numbered_lines = []
    for i, sent in enumerate(sentences, 1):
        numbered_lines.append(f"{i} {sent}")

    # 6번: condition에 따라 분기
    perception = aware_event if condition == "true_belief" else not_aware_event
    numbered_lines.append(f"6 {perception}")

    story_text = "\n".join(numbered_lines) + "\n"
    return story_text, character, 6


def bigtom_row_to_samples(row: list, base_id: int) -> list:
    """
    Big-ToM 1행 → Hi-ToM 통일 형식 샘플 리스트

    choices: "A. <aware_answer>, B. <not_aware_answer>"
    answer:
      true_belief → "A"
      false_belief → "B"
    """
    belief_q = row[5].strip()
    desire_q = row[6].strip()
    action_q = row[7].strip()
    belief_ans_aware = row[8].strip()
    desire_ans_aware = row[9].strip()
    action_ans_aware = row[10].strip()
    belief_ans_not_aware = row[11].strip()
    desire_ans_not_aware = row[12].strip()
    action_ans_not_aware = row[13].strip()

    samples = []
    sid = base_id

    for condition in ["true_belief", "false_belief"]:
        story, character, event_count = bigtom_to_hitom_story(row, condition)

        if condition == "true_belief":
            correct_label = "A"
            b_ans, d_ans, a_ans = belief_ans_aware, desire_ans_aware, action_ans_aware
        else:
            correct_label = "B"
            b_ans, d_ans, a_ans = belief_ans_not_aware, desire_ans_not_aware, action_ans_not_aware

        # Belief 질문 (question_order=1)
        if belief_q:
            samples.append({
                "sample_id": sid,
                "dataset": "bigtom",
                "question_order": 1,
                "condition": condition,
                "character": character,
                "event_count": event_count,
                "story": story,
                "question": belief_q,
                "choices": f"A. {belief_ans_aware}, B. {belief_ans_not_aware}",
                "answer": correct_label,
                "answer_text": b_ans,
            })
            sid += 1

        # Desire 질문 (question_order=2)
        if desire_q and desire_ans_aware != desire_ans_not_aware:
            samples.append({
                "sample_id": sid,
                "dataset": "bigtom",
                "question_order": 2,
                "condition": condition,
                "character": character,
                "event_count": event_count,
                "story": story,
                "question": desire_q,
                "choices": f"A. {desire_ans_aware}, B. {desire_ans_not_aware}",
                "answer": correct_label,
                "answer_text": d_ans,
            })
            sid += 1

        # Action 질문 (question_order=3)
        if action_q:
            samples.append({
                "sample_id": sid,
                "dataset": "bigtom",
                "question_order": 3,
                "condition": condition,
                "character": character,
                "event_count": event_count,
                "story": story,
                "question": action_q,
                "choices": f"A. {action_ans_aware}, B. {action_ans_not_aware}",
                "answer": correct_label,
                "answer_text": a_ans,
            })
            sid += 1

    return samples


def convert_bigtom(csv_path: str, output_path: str) -> int:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) >= 17:
                rows.append(row)

    all_samples = []
    sid = 0
    errors = 0

    for i, row in enumerate(rows):
        try:
            samples = bigtom_row_to_samples(row, base_id=sid)
            all_samples.extend(samples)
            sid += len(samples)
        except Exception as e:
            print(f"[Warning] Row {i} skipped: {e}")
            errors += 1

    out = {"dataset": "bigtom", "total": len(all_samples), "errors": errors, "data": all_samples}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[Preprocess] Big-ToM: {len(rows)} rows → {len(all_samples)} samples → {output_path}")
    return len(all_samples)


def normalize_hitom(input_path: str, output_path: str) -> int:
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    normalized = []

    for i, item in enumerate(data):
        story = item.get("story", "")
        first_line = story.split("\n")[0] if story else ""
        characters = _extract_hitom_characters(first_line)
        event_count = len([l for l in story.split("\n") if l.strip() and l[0].isdigit()])

        normalized.append({
            "sample_id": item.get("sample_id", i),
            "dataset": "hitom",
            "question_order": item.get("question_order", 0),
            "condition": None,
            "character": characters,
            "event_count": event_count,
            "story": story,
            "question": item.get("question", ""),
            "choices": item.get("choices", ""),
            "answer": item.get("answer", ""),
            "answer_text": item.get("answer", ""),
        })

    out = {"dataset": "hitom", "total": len(normalized), "data": normalized}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[Preprocess] Hi-ToM: {len(normalized)} samples → {output_path}")
    return len(normalized)


def _extract_hitom_characters(first_line: str) -> list:
    match = re.search(r'^\d+\s+(.+?)\s+entered', first_line)
    if not match:
        return []
    names_str = match.group(1)
    names = re.split(r',\s*|\s+and\s+', names_str)
    return [n.strip() for n in names if n.strip()]


def preview_samples(json_path: str, n: int = 2):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("data", [])
    from collections import Counter
    order_dist = Counter(s["question_order"] for s in samples)

    print(f"\nDataset: {data.get('dataset')} | Total: {data.get('total')}")
    print(f"question_order 분포: {dict(sorted(order_dist.items()))}")
    if data.get("dataset") == "bigtom":
        cond_dist = Counter(s.get("condition") for s in samples)
        print(f"condition 분포: {dict(cond_dist)}")

    print(f"\n{'='*60}")
    for sample in samples[:n]:
        print(f"\n[sample_id={sample['sample_id']} | order={sample['question_order']} | condition={sample.get('condition')}]")
        print(f"character: {sample.get('character')}")
        print(f"story:\n{sample['story']}")
        print(f"question: {sample['question']}")
        print(f"choices: {sample['choices'][:100]}")
        print(f"answer: {sample['answer']} | answer_text: {sample.get('answer_text', '')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bigtom_csv", default="data/bigtom/bigtom.csv")
    parser.add_argument("--hitom_json", default="data/hitom/Hi-ToM_data.json")
    parser.add_argument("--bigtom_out", default="data/bigtom/bigtom_unified.json")
    parser.add_argument("--hitom_out", default="data/hitom/hitom_unified.json")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview_n", type=int, default=2)
    args = parser.parse_args()

    if Path(args.bigtom_csv).exists():
        convert_bigtom(args.bigtom_csv, args.bigtom_out)
        if args.preview:
            preview_samples(args.bigtom_out, n=args.preview_n)
    else:
        print(f"[Skip] Big-ToM CSV not found: {args.bigtom_csv}")

    if Path(args.hitom_json).exists():
        normalize_hitom(args.hitom_json, args.hitom_out)
        if args.preview:
            preview_samples(args.hitom_out, n=args.preview_n)
    else:
        print(f"[Skip] Hi-ToM JSON not found: {args.hitom_json}")