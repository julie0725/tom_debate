import json
from pathlib import Path

data = json.load(open("data/hitom/Hi-ToM_data.json", encoding="utf-8"))["data"]
expected = {str(x["sample_id"]) for x in data}

by_id = {}
path = Path("outputs/experiments/exp_0_baseline/results_Hi-ToM_data.jsonl")
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    by_id[str(r["dataset_id"])] = r

got = set(by_id.keys())
missing = sorted(expected - got, key=int)
print("expected", len(expected), "unique in results", len(got), "missing", len(missing))
if missing[:10]:
    print("first missing ids:", missing[:10])
wrong = [r for i, r in by_id.items() if i in expected]
empty_ans = sum(
    1
    for r in by_id.values()
    if not any(a.get("value") for a in (r.get("final_answer") or {}).get("answers", []))
)
print("empty final answers:", empty_ans)
