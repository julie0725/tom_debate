import csv
import logging
import re
from pathlib import Path
from typing import Iterator

from core.tom_task import ToMTask
from data.adapters.base import DatasetAdapter

logger = logging.getLogger(__name__)


class CsvAdapter(DatasetAdapter):
    """Loads Big-ToM semicolon-delimited CSV datasets directly.

    Each CSV row produces up to 6 ToMTask objects (2 conditions × up to 3 questions).
    CSV column layout (0-indexed):
      0  = story (5 sentences), 1 = aware_event, 2 = not_aware_event
      5  = belief_q, 6  = desire_q, 7  = action_q
      8  = belief_ans_aware,  9  = desire_ans_aware,  10 = action_ans_aware
      11 = belief_ans_not_aware, 12 = desire_ans_not_aware, 13 = action_ans_not_aware
    """

    def load(self) -> Iterator[ToMTask]:
        sid = 0
        with open(self.path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if len(row) < 17:
                    continue
                try:
                    tasks = self._row_to_tasks(row, base_id=sid)
                    for task in tasks:
                        yield task
                    sid += len(tasks)
                except Exception as e:
                    logger.warning(f"Row {sid} skipped: {e}")

    @staticmethod
    def _split_sentences(text: str) -> list:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sents if s.strip()]

    @staticmethod
    def _extract_character_name(sentence: str) -> str:
        pronouns = {"he", "she", "they", "it", "his", "her", "their", "the", "a", "an"}
        words = sentence.split()
        if words and words[0].lower() not in pronouns:
            name = re.match(r'^([A-Z][a-z]+)', words[0])
            if name:
                return name.group(1)
        return ""

    @classmethod
    def _to_event_story(cls, row: list, condition: str) -> tuple:
        """Convert a CSV row + condition → (numbered_story, character, event_count)."""
        sentences = cls._split_sentences(row[0].strip())
        if len(sentences) != 5:
            raise ValueError(f"Expected 5 sentences, got {len(sentences)}")
        character = cls._extract_character_name(sentences[0])
        lines = [f"{i} {s}" for i, s in enumerate(sentences, 1)]
        perception = row[1].strip() if condition == "true_belief" else row[2].strip()
        lines.append(f"6 {perception}")
        return "\n".join(lines) + "\n", character, 6

    @classmethod
    def _row_to_tasks(cls, row: list, base_id: int) -> list:
        belief_q = row[5].strip()
        desire_q = row[6].strip()
        action_q = row[7].strip()
        b_aware, d_aware, a_aware = row[8].strip(), row[9].strip(), row[10].strip()
        b_not, d_not, a_not = row[11].strip(), row[12].strip(), row[13].strip()

        tasks = []
        sid = base_id

        for condition in ("true_belief", "false_belief"):
            story, character, event_count = cls._to_event_story(row, condition)
            gold = "A" if condition == "true_belief" else "B"
            _ORDER_TO_REASONING = {1: "1st-order", 2: "2nd-order", 3: "3rd-order"}
            characters = [character] if character else []
            meta_base = {"condition": condition, "characters": characters, "event_count": event_count}

            if belief_q:
                tasks.append(ToMTask(
                    context=story,
                    question=f"{cls._strip_options(belief_q)}\nChoices: A. {b_aware}, B. {b_not}",
                    gold_answer=gold,
                    dataset_id=str(sid),
                    metadata={**meta_base, "question_order": 1, "reasoning_type": _ORDER_TO_REASONING[1]},
                ))
                sid += 1

            if desire_q and d_aware != d_not:
                tasks.append(ToMTask(
                    context=story,
                    question=f"{cls._strip_options(desire_q)}\nChoices: A. {d_aware}, B. {d_not}",
                    gold_answer=gold,
                    dataset_id=str(sid),
                    metadata={**meta_base, "question_order": 2, "reasoning_type": _ORDER_TO_REASONING[2]},
                ))
                sid += 1

            if action_q:
                tasks.append(ToMTask(
                    context=story,
                    question=f"{cls._strip_options(action_q)}\nChoices: A. {a_aware}, B. {a_not}",
                    gold_answer=gold,
                    dataset_id=str(sid),
                    metadata={**meta_base, "question_order": 3, "reasoning_type": _ORDER_TO_REASONING[3]},
                ))
                sid += 1

        return tasks

    @staticmethod
    def _strip_options(text: str) -> str:
        stripped = re.sub(r'(?:\s+[A-Z][).]\s+\S+)+\s*$', '', text.strip())
        return stripped.strip() or text.strip()
