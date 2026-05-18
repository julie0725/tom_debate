"""
extract_dataset.py
-------------------
Ablation 실험용 데이터셋 추출

생성:
    data/bigtom/bigtom.csv         ← bigtom_raw.csv의 앞 100행
    data/hitom/Hi-ToM_data.json    ← Hi-ToM_data_raw.json에서 층화 추출 60개
                                    deception × story_length × question_order
                                    = 2 × 3 × 5 = 30칸, 각 칸에서 2개씩, seed=42)

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
    if not BIGTOM_RAW.exists():
        raise FileNotFoundError(
            f"BigToM raw file not found: {BIGTOM_RAW}\n"
            f"Did you rename the original to bigtom_raw.csv?"
        )

    lines = BIGTOM_RAW.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) < BIGTOM_N_ROWS:
        raise ValueError(
            f"BigToM raw has only {len(lines)} rows, need at least {BIGTOM_N_ROWS}"
        )

    BIGTOM_OUT.write_text("".join(lines[:BIGTOM_N_ROWS]), encoding="utf-8")


def extract_hitom() -> None:
    if not HITOM_RAW.exists():
        raise FileNotFoundError(
            f"Hi-ToM raw file not found: {HITOM_RAW}\n"
            f"Did you rename the original to Hi-ToM_data_raw.json?"
        )

    try:
        raw = json.loads(HITOM_RAW.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Hi-ToM raw is not valid JSON: {e}") from e

    if not isinstance(raw, dict) or "data" not in raw:
        raise ValueError(
            "Hi-ToM raw must have shape {'data': [...]}; got something else."
        )

    groups = defaultdict(list)
    for item in raw["data"]:
        key = (item["deception"], item["story_length"], item["question_order"])
        groups[key].append(item)

    rng = random.Random(SEED)
    sampled = []
    for key in sorted(groups.keys()):
        bucket = groups[key]
        if len(bucket) < HITOM_PER_CELL:
            raise ValueError(
                f"Hi-ToM cell {key} has only {len(bucket)} items, need {HITOM_PER_CELL}"
            )
        sampled.extend(rng.sample(bucket, HITOM_PER_CELL))

    HITOM_OUT.write_text(
        json.dumps({"data": sampled}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        extract_bigtom()
        extract_hitom()
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)