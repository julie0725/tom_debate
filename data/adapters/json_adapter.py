import json
import re
from pathlib import Path
from typing import Iterator

from core.tom_task import ToMTask
from data.adapters.base import DatasetAdapter


class JsonAdapter(DatasetAdapter):
    """Loads event-based JSON datasets (e.g., Hi-ToM) directly.

    Handles original Hi-ToM_data.json without pre-processing.
    Extracts character names and event counts inline from story text.
    Text answers are reverse-mapped to their choice label via exact match.
    """

    def load(self) -> Iterator[ToMTask]:
        raw = json.loads(Path(self.path).read_text(encoding="utf-8"))
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        for i, item in enumerate(data):
            story = item.get("story", "")
            choices = item.get("choices", "") or ""
            question = self._strip_options(item.get("question", ""))
            full_question = f"{question}\nChoices: {choices}" if choices else question
            ans = str(item.get("answer", "") or "").strip()
            gold = self._resolve_answer(ans, choices) if ans else None
            char_raw = item.get("character") or self._extract_characters(story)
            characters = char_raw if isinstance(char_raw, list) else ([char_raw] if char_raw else [])
            event_count = item.get("event_count") or self._count_events(story)
            q_order = item.get("question_order") or 1
            _ORDER_TO_REASONING = {1: "1st-order", 2: "2nd-order", 3: "3rd-order"}
            reasoning_type = _ORDER_TO_REASONING.get(q_order, "1st-order")
            yield ToMTask(
                context=story,
                question=full_question,
                gold_answer=gold,
                dataset_id=str(item.get("sample_id", i)),
                metadata={
                    "question_order": q_order,
                    "condition": item.get("condition"),
                    "characters": characters,
                    "event_count": event_count,
                    "reasoning_type": reasoning_type,
                },
            )

    @staticmethod
    def _count_events(story: str) -> int:
        return len([ln for ln in story.split("\n") if ln.strip() and ln[0].isdigit()])

    @staticmethod
    def _extract_characters(story: str) -> list:
        first_line = story.split("\n")[0] if story else ""
        m = re.search(r'^\d+\s+(.+?)\s+entered', first_line)
        if not m:
            return []
        names = re.split(r',\s*|\s+and\s+', m.group(1))
        return [n.strip() for n in names if n.strip()]

    @staticmethod
    def _strip_options(text: str) -> str:
        stripped = re.sub(r'(?:\s+[A-Z][).]\s+\S+)+\s*$', '', text.strip())
        return stripped.strip() or text.strip()

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
