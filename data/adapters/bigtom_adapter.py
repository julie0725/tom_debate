import json
from pathlib import Path
from typing import Iterator

from core.tom_task import ToMTask
from data.adapters.dataset_adapter import DatasetAdapter


class BigToMAdapter(DatasetAdapter):
    """Loads Big-ToM unified JSON (produced by data/preprocess.py).

    Expected sample keys: sample_id, story, question, choices, answer (letter "A"/"B"),
    question_order, condition, character, event_count.
    """

    def load(self) -> Iterator[ToMTask]:
        raw = json.loads(Path(self.path).read_text(encoding="utf-8"))
        samples = raw.get("data", raw) if isinstance(raw, dict) else raw
        for item in samples:
            choices = item.get("choices", "") or ""
            question = self._strip_options(item.get("question", ""))
            full_question = f"{question}\nChoices: {choices}" if choices else question
            gold = str(item.get("answer", "") or "").strip() or None
            yield ToMTask(
                context=item.get("story", ""),
                question=full_question,
                gold_answer=gold,
                dataset_id=str(item.get("sample_id", "")),
                metadata={
                    "question_order": item.get("question_order"),
                    "condition": item.get("condition"),
                    "character": item.get("character"),
                    "event_count": item.get("event_count"),
                },
            )
