"""
experiments.py
--------------
논문/캡스톤 실험 조건 정의 및 단일 실험 실행
"""

import copy
import json
import logging
import shutil
from pathlib import Path

from user.ai_user import AIUser
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)

# 실험 ID → config overrides + 설명
EXPERIMENTS = {
    "0": {
        "name": "baseline",
        "description": "기존 구조: agent1+2+3, 불일치 시 토론",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {
                "use_debate": True,
                "debate_mode": "on_disagreement",
            },
        },
    },
    "1-1": {
        "name": "majority_only",
        "description": "토론 없이 초기 3에이전트 답으로 다수결만",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {
                "use_debate": True,
                "debate_mode": "majority_only",
            },
        },
    },
    "1-2": {
        "name": "always_debate",
        "description": "불일치 여부와 관계없이 항상 토론",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {
                "use_debate": True,
                "debate_mode": "always",
            },
        },
    },
    "2": {
        "name": "minimal_prompts",
        "description": "에이전트 프롬프트 최소 출력 스키마",
        "overrides": {
            "agents": {
                "use_agent1": True,
                "use_agent2": True,
                "use_agent3": True,
                "minimal_output": True,
            },
            "debate": {"use_debate": True, "debate_mode": "on_disagreement"},
        },
    },
    "3": {
        "name": "tom_sota_backend",
        "description": "CAMEL-AI / ToM SOTA 백엔드 (설정 필요)",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": True},
            "debate": {"use_debate": True, "debate_mode": "on_disagreement"},
            "system": {"backend": "camel"},
        },
    },
    "4": {
        "name": "camel_baseline_compare",
        "description": "기본 CAMEL-AI 대비 성능 비교",
        "overrides": {
            "system": {"backend": "camel_baseline"},
        },
    },
    "5-1": {
        "name": "agents_2_3",
        "description": "agent2+3만, 토론",
        "overrides": {
            "agents": {"use_agent1": False, "use_agent2": True, "use_agent3": True},
            "debate": {"use_debate": True, "debate_mode": "on_disagreement"},
        },
    },
    "5-2": {
        "name": "agents_1_3",
        "description": "agent1+3만, 토론",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": False, "use_agent3": True},
            "debate": {"use_debate": True, "debate_mode": "on_disagreement"},
        },
    },
    "5-3": {
        "name": "agents_1_2",
        "description": "agent1+2만, 토론",
        "overrides": {
            "agents": {"use_agent1": True, "use_agent2": True, "use_agent3": False},
            "debate": {"use_debate": True, "debate_mode": "on_disagreement"},
        },
    },
}


def infer_dataset_name(dataset_path: str) -> str:
    path_lower = dataset_path.replace("\\", "/").lower()
    if "bigtom" in path_lower:
        return "bigtom"
    if "hitom" in path_lower or "hi-tom" in path_lower:
        return "hitom"
    return Path(dataset_path).parent.name


def apply_experiment_config(
    base_config: dict,
    experiment_id: str,
    dataset_path: str = None,
) -> dict:
    if experiment_id not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment_id}'. "
            f"Choose from: {', '.join(EXPERIMENTS.keys())}"
        )
    cfg = copy.deepcopy(base_config)
    spec = EXPERIMENTS[experiment_id]
    for section, overrides in spec["overrides"].items():
        if section not in cfg:
            cfg[section] = {}
        cfg[section].update(overrides)
    cfg.setdefault("evaluation", {})["experiment_id"] = experiment_id
    cfg["evaluation"]["experiment_name"] = spec["name"]
    if dataset_path:
        cfg["evaluation"]["dataset_name"] = infer_dataset_name(dataset_path)
        cfg["evaluation"]["dataset_path"] = dataset_path
    return cfg


class ExperimentRunner:
    def __init__(self, base_config: dict, dataset_path: str, output_root: str = "outputs/experiments/"):
        self.base_config = base_config
        self.dataset_path = dataset_path
        self.output_root = Path(output_root)

    def run(
        self,
        experiment_id: str,
        limit=None,
        resume: bool = True,
    ) -> dict:
        spec = EXPERIMENTS[experiment_id]
        cfg = apply_experiment_config(
            self.base_config, experiment_id, dataset_path=self.dataset_path
        )

        out_dir = self.output_root / f"exp_{experiment_id}_{spec['name']}"
        if out_dir.exists() and not resume:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg["evaluation"]["output_dir"] = str(out_dir) + "/"
        cfg["evaluation"]["resume"] = resume

        ds_name = cfg["evaluation"].get("dataset_name", "hitom")
        print("\n" + "=" * 60)
        print(f"  Experiment {experiment_id}: {spec['name']}")
        print(f"  {spec['description']}")
        print(f"  Dataset: {self.dataset_path} ({ds_name})")
        print(f"  Metrics: {'Q1 only' if ds_name == 'hitom' else 'Q1+Q2+Q3'}")
        print(f"  Output: {out_dir}")
        if limit:
            print(f"  Limit: {limit} samples")
        print(f"  Resume: {resume}")
        print("=" * 60 + "\n")

        ai_user = AIUser(config=cfg)
        ai_user.submit_from_dataset(self.dataset_path, limit=limit, resume=resume)

        results_file = cfg["evaluation"].get("results_file", "results.jsonl")
        summary_name = f"evaluation_exp_{experiment_id}.json"
        evaluator = Evaluator(output_dir=str(out_dir) + "/")
        ds_name = infer_dataset_name(self.dataset_path)
        summary = evaluator.evaluate_from_jsonl(
            results_file=results_file,
            output_file=summary_name,
            dataset_name=ds_name,
        )
        summary["experiment_id"] = experiment_id
        summary["experiment_name"] = spec["name"]
        summary["description"] = spec["description"]

        meta_path = out_dir / f"meta_exp_{experiment_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary
