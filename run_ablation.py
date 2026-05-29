#!/usr/bin/env python
"""
run_ablation.py  ─  E / F / G 조건 단독 실행 + CSV 누적 저장
기존 코드 수정 없음.

사용법:
  python run_ablation.py --condition E
  python run_ablation.py --condition F --dataset bigtom
  python run_ablation.py --condition G
  python run_ablation.py --condition E --skip-sampling   # 이미 샘플링된 경우

--dataset: hitom (기본값) | bigtom
--skip-sampling: 샘플 파일이 이미 존재하면 재샘플링 건너뜀
"""
from dotenv import load_dotenv
load_dotenv()

import sys

def _configure_stdio() -> None:
    """Windows 기본 cp949 콘솔에서도 print 예외가 나지 않게 UTF-8로 설정."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

_configure_stdio()

import argparse
import copy
import csv
import json
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path

from user.ablation_ai_user import AblationAIUser
from evaluation.evaluator import Evaluator
from evaluation.experiments import infer_dataset_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── 데이터셋 경로 매핑 ────────────────────────────────────────────────────────
DATASET_PATHS = {
    "hitom":  "data/hitom/Hi-ToM_data.json",
    "bigtom": "data/bigtom/bigtom.csv",
}

# sampling_dataset.py가 기대하는 raw → sampled 경로
RAW_PATHS = {
    "hitom":  ("data/hitom/Hi-ToM_data_raw.json", "data/hitom/Hi-ToM_data.json"),
    "bigtom": ("data/bigtom/bigtom_raw.csv",       "data/bigtom/bigtom.csv"),
}

# ── Ablation 조건 정의 ────────────────────────────────────────────────────────
ABLATION_CONDITIONS = {
    "A": {
        "name": "baseline",
        "description": "기본 구조: 3-agent + 선택적 토론",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {"use_debate": True},
        },
    },
    "E": {
        "name": "camel_backend",
        "description": "CAMEL 백엔드 + 기존 3-agent/debate 구조 유지",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {"use_debate": True},
            "system": {"backend": "camel"},
        },
    },
    "F": {
        "name": "camel_baseline",
        "description": "pure CAMEL 단일 ChatAgent, 구조 없음",
        "overrides": {
            "system": {"backend": "camel_baseline"},
            "agents": {"use_agent1": False, "use_agent2": False, "use_agent3": False},
            "debate": {"use_debate": False},
        },
    },
    "G-1": {
        "name": "agent1_only",
        "description": "Agent1(진실판단)만 사용, 토론 없음",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": False, "use_agent3": False},
            "debate": {"use_debate": False},
        },
    },
    "G-2": {
        "name": "agent2_only",
        "description": "Agent2(신념추적)만 사용, 토론 없음",
        "overrides": {
            "agents": {"use_agent1": False, "use_agent2": True, "use_agent3": False},
            "debate": {"use_debate": False},
        },
    },
    "G-3": {
        "name": "agent3_only",
        "description": "Agent3(관점시뮬레이션)만 사용, 토론 없음",
        "overrides": {
            "agents": {"use_agent1": False, "use_agent2": False, "use_agent3": True},
            "debate": {"use_debate": False},
        },
    },
}

CSV_PATH = Path("outputs/ablation_results.csv")
CSV_COLUMNS = [
    "condition", "dataset", "total",
    "q1_accuracy", "q2_accuracy", "q3_accuracy", "joint_accuracy",
    "total_conflicts", "total_rounds", "conflict_rate",
    "debate_trigger_rate", "avg_debate_rounds",
    "timestamp",
]


# ── 샘플링 ────────────────────────────────────────────────────────────────────

def run_sampling(dataset: str, skip_if_exists: bool = False) -> None:
    """
    scripts/sampling_dataset.py 실행.
    raw 파일(hitom_raw / bigtom_raw)이 없으면 에러.
    sampled 파일이 이미 있고 skip_if_exists=True 이면 건너뜀.
    """
    raw_path, sampled_path = RAW_PATHS[dataset]

    if skip_if_exists and Path(sampled_path).exists():
        print(f"[Sampling] 기존 파일 사용: {sampled_path}")
        return

    if not Path(raw_path).exists():
        print(f"[ERROR] raw 파일 없음: {raw_path}")
        print(f"  원본 파일을 {raw_path} 로 이름 변경 후 재실행하세요.")
        sys.exit(1)

    print(f"[Sampling] {raw_path} → {sampled_path}")
    # sampling_dataset.py를 직접 import해서 실행
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "sampling_dataset",
        Path("scripts/sampling_dataset.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if dataset == "hitom":
        mod.extract_hitom()
    elif dataset == "bigtom":
        mod.extract_bigtom()

    print(f"[Sampling] 완료: {sampled_path}")


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(base_config: dict, condition: str) -> dict:
    cfg = copy.deepcopy(base_config)
    for section, overrides in ABLATION_CONDITIONS[condition]["overrides"].items():
        cfg.setdefault(section, {}).update(overrides)
    return cfg


def compute_conflict_metrics(jsonl_path: Path) -> dict:
    """
    기존 JSONL 필드(debate_triggered / debate_round / majority_vote_applied)에서
    라운드별 누적 충돌 지표 역산.

    공식:
      total_rounds    = 1 + debate_round   (debate_triggered=True)  else 1
      total_conflicts = debate_round + majority_vote_applied  (debate_triggered=True) else 0
    """
    total_conflicts = 0
    total_rounds = 0

    if not jsonl_path.exists():
        return {"total_conflicts": 0, "total_rounds": 0, "conflict_rate": 0.0}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("debate_triggered"):
                d_round  = r.get("debate_round", 0)
                total_rounds    += 1 + d_round
                total_conflicts += d_round
            else:
                total_rounds += 1

    rate = round(total_conflicts / total_rounds, 4) if total_rounds > 0 else 0.0
    return {"total_conflicts": total_conflicts, "total_rounds": total_rounds, "conflict_rate": rate}


def append_csv_row(row: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    print(f"\n[CSV] 저장 → {CSV_PATH}")


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def run_condition(
    condition: str,
    config_path: str,
    dataset: str,
    skip_sampling: bool = False,
    resume: bool = False,
) -> dict:
    # 1. 샘플링
    run_sampling(dataset, skip_if_exists=skip_sampling)

    dataset_path = DATASET_PATHS[dataset]
    spec = ABLATION_CONDITIONS[condition]

    print("\n" + "=" * 60)
    print(f"  Condition  : {condition}  ({spec['name']})")
    print(f"  {spec['description']}")
    print(f"  Dataset    : {dataset_path}  ({dataset})")
    print("=" * 60 + "\n")

    # 2. config 구성
    cfg = apply_overrides(load_config(config_path), condition)
    out_dir = Path("outputs") / "ablation" / f"condition_{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("evaluation", {})["output_dir"] = str(out_dir) + "/"

    # 3. 파이프라인 실행
    AblationAIUser(config=cfg).submit_from_dataset(dataset_path, resume=resume)

    results_file = cfg["evaluation"].get("results_file", "results.jsonl")

    # 4. 정확도 평가
    evaluator = Evaluator(output_dir=str(out_dir) + "/")
    summary = evaluator.evaluate_from_jsonl(
        results_file=results_file,
        output_file=f"evaluation_{condition}.json",
    )

    # 5. 충돌 지표 역산
    conflict = compute_conflict_metrics(out_dir / results_file)

    row = {
        "condition":           condition,
        "dataset":             dataset,
        "total":               summary.get("total", 0),
        "q1_accuracy":         summary.get("q1_belief_accuracy", ""),
        "q2_accuracy":         summary.get("q2_desire_accuracy", ""),
        "q3_accuracy":         summary.get("q3_action_accuracy", ""),
        "joint_accuracy":      summary.get("joint_accuracy", ""),
        "total_conflicts":     conflict["total_conflicts"],
        "total_rounds":        conflict["total_rounds"],
        "conflict_rate":       conflict["conflict_rate"],
        "debate_trigger_rate": summary.get("debate_trigger_rate", ""),
        "avg_debate_rounds":   summary.get("avg_debate_rounds", ""),
        "timestamp":           datetime.now().isoformat(timespec="seconds"),
    }
    append_csv_row(row)
    return row


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation E/F/G 단독 실행")
    parser.add_argument("--condition", required=True,
                        choices=["A", "E", "F", "G-1", "G-2", "G-3"])
    parser.add_argument("--dataset", default="hitom", choices=["hitom", "bigtom"])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-sampling", action="store_true",
                        help="샘플 파일이 이미 있으면 재샘플링 건너뜀")
    parser.add_argument("--resume", action="store_true",
                        help="기존 jsonl 결과를 유지하고 미완료 task만 이어서 실행")
    args = parser.parse_args()

    result = run_condition(
        condition=args.condition,
        config_path=args.config,
        dataset=args.dataset,
        skip_sampling=args.skip_sampling,
        resume=args.resume,
    )

    print("\n" + "=" * 60)
    print("  FINAL METRICS")
    print("=" * 60)
    for k, v in result.items():
        if k != "timestamp":
            print(f"  {k:<26}: {v}")
    print("=" * 60)
