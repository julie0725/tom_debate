"""
user/ai_user.py
----------
AI User: 시스템의 단일 입력 게이트웨이
- submit_from_text()  : 직접 텍스트 입력 또는 자연어 원문 → ToMTask → pipeline
- submit_from_dataset(): 데이터셋 경로 → adapter → ToMTask 반복 → pipeline
- _submit()           : 내부 공통 실행자 (ToMTask → ToMState)
"""

import asyncio
import logging
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Optional

from core.context_file import ToMState, ToMAnswers
from core.extractor import Extractor
from core.llm_client import TokenCounter, set_token_counter
from core.message_pool import MessagePool
from core.tom_task import ToMTask
from data.adapters.proxy import Proxy
from supervisor.supervisor import Supervisor

logger = logging.getLogger(__name__)


class AIUser:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get("evaluation", {}).get("output_dir", "outputs/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.proxy = Proxy(config)
        self.extractor = Extractor(config)
        self.progress_callback = lambda *_: None
        self._file_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def submit_from_text(
        self,
        raw_text: str = "",
        scenario: Optional[str] = None,
        question: Optional[str] = None,
        gold_answer: Optional[str] = None,
        dataset_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ToMState:
        """Run the pipeline from either explicit scenario+question or a raw natural language string.

        - scenario and question both given → build ToMTask directly (structured input)
        - only raw_text given → Proxy routes through TextAdapter (LLM extraction)
        """
        if scenario and question:
            task = ToMTask(
                context=scenario,
                question=question,
                gold_answer=gold_answer,
                dataset_id=dataset_id or "manual",
                metadata=metadata or {},
            )
        else:
            tasks = list(self.proxy.get_tasks(raw_text))
            if not tasks:
                raise ValueError("[AIUser] TextAdapter returned no tasks")
            task = tasks[0]
        return self._submit(task)

    def submit_from_dataset(
        self,
        path: str,
        limit: Optional[int] = None,
        sample_ids: Optional[list] = None,
    ) -> None:
        """Auto-detect adapter from path, load tasks, and run the pipeline for each."""
        dataset_stem = Path(path).stem
        results_file = f"results_{dataset_stem}.jsonl"
        if "evaluation" not in self.config:
            self.config["evaluation"] = {}
        self.config["evaluation"]["results_file"] = results_file

        results_path = self.output_dir / results_file
        if results_path.exists():
            results_path.unlink()

        self.md_dir = self.output_dir / "md"
        self.md_dir.mkdir(parents=True, exist_ok=True)
        for old_md in self.md_dir.glob("*.md"):
            old_md.unlink()

        tasks = list(self.proxy.get_tasks(path, limit))
        for task in tasks:
            task.metadata.setdefault("dataset_type", dataset_stem)

        if sample_ids is not None:
            id_set = set(map(str, sample_ids))
            tasks = [t for t in tasks if str(t.dataset_id) in id_set]
            logger.info(f"[AIUser] sample_ids 필터 적용 → {len(tasks)}개 샘플")

        if not tasks:
            logger.error(f"[AIUser] No samples loaded from: {path}")
            return

        workers = self.config.get("system", {}).get("parallel_workers", 5)
        total = len(tasks)
        print(f"data 개수 : {total}  (workers={workers})")

        def process(args):
            i, task = args
            logger.info(f"[AIUser] Processing {i}/{total} | id={task.dataset_id}")
            self._submit(task, idx=i, total=total)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process, (i + 1, task)): task
                for i, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[AIUser] Sample {task.dataset_id} failed: {e}")

    # ── Internal executor ─────────────────────────────────────────────────────

    def _submit(self, task: ToMTask, idx: int = None, total: int = None) -> ToMState:
        questions = [{"id": "q1", "text": task.question}]
        if task.metadata.get("q2"):
            questions.append({"id": "q2", "text": task.metadata["q2"]})
        if task.metadata.get("q3"):
            questions.append({"id": "q3", "text": task.metadata["q3"]})

        ground_truth = None
        if task.gold_answer is not None:
            gt_answers = [{"id": "q1", "value": task.gold_answer}]
            if task.metadata.get("gold_q2"):
                gt_answers.append({"id": "q2", "value": task.metadata["gold_q2"]})
            if task.metadata.get("gold_q3"):
                gt_answers.append({"id": "q3", "value": task.metadata["gold_q3"]})
            ground_truth = ToMAnswers(answers=gt_answers)

        common_state = self.extractor.extract(task)

        state = ToMState(
            scenario=task.context,
            questions=questions,
            dataset_id=task.dataset_id,
            ground_truth=ground_truth,
            reasoning_type=common_state.reasoning_type,
            characters=[c.name for c in common_state.characters],
            common_state=common_state.to_dict(),
        )

        pool = MessagePool()
        pool.publish(state, publisher="ai_user")
        logger.info(f"[AIUser] Published context file | dataset_id={task.dataset_id}")

        model = self.config.get("system", {}).get("model", "gpt-3.5-turbo")
        counter = TokenCounter()
        set_token_counter(counter)
        start = time.time()

        supervisor = Supervisor(pool=pool, config=self.config, progress_callback=self.progress_callback)
        supervisor.event_callback = getattr(self, "event_callback", None)
        final_state = asyncio.run(supervisor.run())

        elapsed = round(time.time() - start, 2)
        set_token_counter(None)
        prompt_tok, completion_tok = counter.get()
        cost = counter.cost(model)
        total_tok = prompt_tok + completion_tok

        if idx and total:
            print(f"  [{idx}/{total}] {task.dataset_id}  {elapsed:.1f}s | {total_tok:,} tokens | ~${cost:.4f}")

        self._save_result(final_state, elapsed, prompt_tok, completion_tok, cost)

        safe_id = str(task.dataset_id or "result").replace("/", "_").replace(":", "_")
        md_dir = getattr(self, "md_dir", self.output_dir)
        (md_dir / f"{safe_id}.md").write_text(pool.dump_markdown(), encoding="utf-8")

        logger.info(f"[AIUser] Pipeline complete | status={final_state.status}")
        return final_state

    def _save_result(self, state: ToMState, elapsed_sec: float = 0.0,
                     prompt_tokens: int = 0, completion_tokens: int = 0,
                     estimated_cost_usd: float = 0.0) -> None:
        results_file = self.config.get("evaluation", {}).get("results_file", "results.jsonl")
        log_path = self.output_dir / results_file
        record = {
            "timestamp": datetime.now().isoformat(),
            "dataset_id": state.dataset_id,
            "debate_round": state.debate_round,
            "debate_triggered": state.debate_triggered,
            "majority_vote_applied": state.majority_vote_applied,
            "final_answer": asdict(state.final_answer),
            "ground_truth": asdict(state.ground_truth) if state.ground_truth else None,
            "agent_outputs": state.agent_outputs,
            "elapsed_sec": elapsed_sec,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
        }
        with self._file_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
