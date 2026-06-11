#!/usr/bin/env python
"""Verify hitom=60 samples, bigtom=100 rows / 200 tasks."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.adapters.proxy import Proxy

HITOM_EXPECT = 60
BIGTOM_ROWS_EXPECT = 100
BIGTOM_TASKS_EXPECT = 200


def main() -> int:
    hitom_path = ROOT / "data/hitom/Hi-ToM_data.json"
    bigtom_path = ROOT / "data/bigtom/bigtom.csv"

    hitom = json.loads(hitom_path.read_text(encoding="utf-8"))
    n_hitom = len(hitom["data"])

    lines = [ln for ln in bigtom_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n_bigtom_rows = len(lines)  # no header row in bigtom.csv

    proxy = Proxy({})
    n_hitom_tasks = len(list(proxy.get_tasks(str(hitom_path))))
    n_bigtom_tasks = len(list(proxy.get_tasks(str(bigtom_path))))

    ok = True
    checks = [
        ("hitom samples", n_hitom, HITOM_EXPECT),
        ("bigtom csv rows", n_bigtom_rows, BIGTOM_ROWS_EXPECT),
        ("hitom tasks", n_hitom_tasks, HITOM_EXPECT),
        ("bigtom tasks", n_bigtom_tasks, BIGTOM_TASKS_EXPECT),
    ]
    for name, got, expect in checks:
        status = "OK" if got == expect else "FAIL"
        print(f"[{status}] {name}: {got} (expected {expect})")
        if got != expect:
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
