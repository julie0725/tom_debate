"""
extract_dataset.py
-------------------
Ablation 실험용 데이터셋 추출

생성:
    data/bigtom/bigtom.csv         ← bigtom_raw.csv의 앞 100행
    data/hitom/Hi-ToM_data.json    ← Hi-ToM_data_raw.json에서 층화 추출 60개

사용 전 준비:
    기존 데이터셋 파일명 변경 필요
    mv data/bigtom/bigtom.csv -> data/bigtom/bigtom_raw.csv
    mv data/hitom/Hi-ToM_data.json -> data/hitom/Hi-ToM_data_raw.json

사용:
  python scripts/extract_dataset.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42

BIGTOM_RAW = Path("data/bigtom/bigtom_raw.csv")
BIGTOM_OUT = Path("data/bigtom/bigtom.csv")
BIGTOM_N_ROWS = 100

HITOM_RAW = Path("data/hitom/Hi-ToM_data_raw.json")
HITOM_OUT = Path("data/hitom/Hi-ToM_data.json")
HITOM_PER_CELL = 2


def extract_bigtom() -> None:
    lines = BIGTOM_RAW.read_text(encoding="utf-8").splitlines(keepends=True)
    BIGTOM_OUT.write_text("".join(lines[:BIGTOM_N_ROWS]), encoding="utf-8")


def extract_hitom() -> None:
    raw = json.loads(HITOM_RAW.read_text(encoding="utf-8"))

    groups = defaultdict(list)
    for item in raw["data"]:
        key = (item["deception"], item["story_length"], item["question_order"])
        groups[key].append(item)

    rng = random.Random(SEED)
    sampled = []
    for key in sorted(groups.keys()):
        sampled.extend(rng.sample(groups[key], HITOM_PER_CELL))

    HITOM_OUT.write_text(
        json.dumps({"data": sampled}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        extract_bigtom()
        extract_hitom()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)