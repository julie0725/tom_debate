import json
import logging
import re
from pathlib import Path

from core.common_state import CommonToMState, ToMEvent, CharacterInfo, BeliefState
from core.llm_client import get_llm_client, call_llm
from core.tom_task import ToMTask

logger = logging.getLogger(__name__)


class Extractor:
    """
    Converts ToMTask (raw text) → CommonToMState via LLM.
    Results cached at outputs/cache/{dataset_id}.json.
    Cache hit → no LLM call. Cache miss → LLM call → save.
    """

    SYSTEM_PROMPT = """You are a ToM structure extractor. Given a story and question, extract:

1. events: list of objects
   - id: event number from story
   - text: event description
   - observed_by: list of character names present at this moment
     (track who entered/exited — once a character exits, they no longer observe)
   - type: "entry" | "exit" | "action" | "state" | "communication"

2. characters: list of objects
   - name: character name
   - exited_at: event id when they exited (null if never exited or still present)

3. belief_states: list of objects
   - agent: character name
   - proposition: what they believe about (e.g. "lettuce location")
   - value: what they believe (based ONLY on events they witnessed)
   - last_observed_event: last event id they were present for

4. goals: list of objects
   - agent: character name
   - goal: inferred goal

5. reasoning_type: one of
   - "0th-order": world state ("where is X really?")
   - "1st-order": one character's belief ("what does A think?")
   - "2nd-order": A knows what B thinks
   - "3rd-order": three-level nesting

CRITICAL RULES:
- observed_by must ONLY include characters present at that event
- belief_states.value must reflect what agent believes, NOT world truth
- Track entry/exit carefully: character observes only while inside the room

Respond ONLY in valid JSON. No markdown, no prose.
Schema:
{
  "events": [{"id": int, "text": str, "observed_by": [str], "type": str}],
  "characters": [{"name": str, "exited_at": int|null}],
  "belief_states": [{"agent": str, "proposition": str, "value": str, "last_observed_event": int}],
  "goals": [{"agent": str, "goal": str}],
  "reasoning_type": str
}"""

    def __init__(self, config: dict):
        self.config = config
        sys_cfg = config.get("system", {})
        self.model = sys_cfg.get("model", "gpt-3.5-turbo")
        self.max_tokens = sys_cfg.get("max_tokens", 2000)
        self.provider = sys_cfg.get("provider", "openai")
        self.base_url = sys_cfg.get("base_url", None)
        self.client = get_llm_client(provider=self.provider, base_url=self.base_url)
        self.cache_dir = Path("outputs/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = config.get("system", {}).get("temperature", 0.0)

    def extract(self, task: ToMTask) -> CommonToMState:
        cache_path = self.cache_dir / f"{task.dataset_id}.json"

        if cache_path.exists():
            logger.info(f"[Extractor] Cache hit: {task.dataset_id}")
            return self._load_cache(cache_path)

        logger.info(f"[Extractor] Extracting: {task.dataset_id}")
        result = self._extract_via_llm(task)
        self._save_cache(cache_path, result)
        return result

    def _extract_via_llm(self, task: ToMTask) -> CommonToMState:
        user_content = (
            f"Story:\n{task.context}\n\n"
            f"Question: {task.question}\n"
            f"Known characters: {task.metadata.get('characters', [])}"
        )
        raw = call_llm(
            client=self.client,
            model=self.model,
            system_prompt=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"[Extractor] JSON parse error: {e}\nRaw: {raw[:200]}")
            parsed = {
                "events": [], "characters": [], "belief_states": [], "goals": [],
                "reasoning_type": task.metadata.get("reasoning_type", "1st-order"),
            }

        return CommonToMState(
            events=[ToMEvent(**e) for e in parsed.get("events", [])],
            characters=[CharacterInfo(**c) for c in parsed.get("characters", [])],
            belief_states=[BeliefState(**b) for b in parsed.get("belief_states", [])],
            goals=parsed.get("goals", []),
            reasoning_type=parsed.get("reasoning_type", task.metadata.get("reasoning_type", "1st-order")),
            question=task.question,
            gold_answer=task.gold_answer,
            dataset_id=task.dataset_id,
            raw_story=task.context,
        )

    def _save_cache(self, path: Path, state: CommonToMState) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

    def _load_cache(self, path: Path) -> CommonToMState:
        with open(path, encoding="utf-8") as f:
            return CommonToMState.from_dict(json.load(f))
