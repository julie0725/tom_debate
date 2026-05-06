import json
import re
from pathlib import Path
from typing import Iterator

from core.tom_task import ToMTask
from data.adapters.dataset_adapter import DatasetAdapter


class HiToMAdapter(DatasetAdapter):
    """Loads Hi-ToM unified JSON (produced by data/preprocess.py).

    Expected sample keys: sample_id, story, question, choices, answer (text like "green_drawer"),
    question_order, condition, character, event_count.
    The text answer is reverse-mapped to its choice label via exact match.
    """

    def load(self) -> Iterator[ToMTask]:
        raw = json.loads(Path(self.path).read_text(encoding="utf-8"))
        samples = raw.get("data", raw) if isinstance(raw, dict) else raw
        for item in samples:
            choices = item.get("choices", "") or ""
            question = self._strip_options(item.get("question", ""))
            full_question = f"{question}\nChoices: {choices}" if choices else question
            ans = str(item.get("answer", "") or "").strip()
            gold = self._resolve_answer(ans, choices) if ans else None
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

    @staticmethod
    def _resolve_answer(answer_text: str, choices_str: str) -> str:
        """Reverse-map a text answer to its choice label using exact match.

        "green_drawer", "A. blue_drawer, B. green_crate, ..., K. green_drawer" → "K"
        Already-a-letter answers ("A", "B") are returned as-is.
        """
        if not choices_str or re.match(r'^[A-Z]$', answer_text.strip()):
            return answer_text
        for part in choices_str.split(','):
            part = part.strip()
            m = re.match(r'^([A-Z]+)\.\s*(.+)$', part)
            if m and m.group(2).strip() == answer_text.strip():
                return m.group(1)
        return answer_text
