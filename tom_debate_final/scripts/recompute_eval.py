"""Deduplicate results JSONL and recompute Hi-ToM metrics."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluator import Evaluator


def dedupe_jsonl(path: Path) -> list:
    by_id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_id[str(r["dataset_id"])] = r
    return list(by_id.values())


def main():
    base = Path("outputs/experiments/exp_0_baseline")
    src = base / "results_Hi-ToM_data.jsonl"
    records = dedupe_jsonl(src)
    deduped = base / "results_Hi-ToM_data_deduped.jsonl"
    with open(deduped, "w", encoding="utf-8") as f:
        for r in sorted(records, key=lambda x: int(x["dataset_id"]) if str(x["dataset_id"]).isdigit() else x["dataset_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ev = Evaluator(output_dir=str(base) + "/")
    summary = ev.evaluate_from_jsonl(
        jsonl_path=str(deduped),
        output_file="evaluation_exp_0_deduped.json",
        dataset_name="hitom",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
