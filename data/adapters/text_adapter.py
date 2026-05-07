import json
import logging
import re
from typing import Iterator

from core.llm_client import get_llm_client, call_llm
from core.tom_task import ToMTask
from data.adapters.base import TextDatasetAdapter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a ToM input parser.
Given free-form natural language input, extract:
- scenario: background story without the question
- question: the ToM question being asked
- characters: list of character names (count matters)
- reasoning_type:
    0th-order: world state ('where is X really?')
    1st-order: one character's belief ('what does A think?')
    2nd-order: A's knowledge of B's belief
    3rd-order: three-level nesting
Respond ONLY in valid JSON:
{
  "scenario": "...",
  "question": "...",
  "characters": ["..."],
  "reasoning_type": "1st-order"
}"""


class TextAdapter(TextDatasetAdapter):
    """Parses free-form natural language into a single ToMTask via LLM extraction."""

    def load(self) -> Iterator[ToMTask]:
        extracted = self._extract_via_llm()
        yield ToMTask(
            context=extracted.get("scenario", self.text),
            question=extracted.get("question", ""),
            gold_answer=None,
            dataset_id="user_input",
            metadata={
                "characters": extracted.get("characters", []),
                "reasoning_type": extracted.get("reasoning_type", "1st-order"),
                "raw_input": self.text,
            },
        )

    def _extract_via_llm(self) -> dict:
        sys_cfg = self.config.get("system", {})
        client = get_llm_client(
            provider=sys_cfg.get("provider", "openai"),
            base_url=sys_cfg.get("base_url"),
        )
        raw = call_llm(
            client=client,
            model=sys_cfg.get("model", "gpt-3.5-turbo"),
            system_prompt=_SYSTEM_PROMPT,
            user_content=self.text,
            max_tokens=sys_cfg.get("max_tokens", 2000),
        )
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"[TextAdapter] JSON parse error: {e}\nRaw: {raw[:200]}")
            return {
                "scenario": self.text,
                "question": "",
                "characters": [],
                "reasoning_type": "1st-order",
            }
