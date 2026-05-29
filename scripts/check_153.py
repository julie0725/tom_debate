import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
jsonl = ROOT / "outputs/ablation/condition_A/results_Hi-ToM_data.jsonl"
lines = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
ids = [r["dataset_id"] for r in lines]
print("count", len(ids))
print("has 153", "153" in ids)

item = next(x for x in json.loads((ROOT / "data/hitom/Hi-ToM_data.json").read_text(encoding="utf-8"))["data"] if x["sample_id"] == 153)
print("sample 153 answer raw:", item["answer"])
print("question_order", item["question_order"])
