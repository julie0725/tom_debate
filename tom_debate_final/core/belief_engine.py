"""
core/belief_engine.py  —  P5: Deterministic Belief Engine
Witness-based belief tracking with order-k recursive queries.

NOT yet wired into the debate flow.
Verify parse accuracy and engine accuracy independently before connecting.
Integration point: supervisor/debate.py _majority_vote or a new adjudication step —
  when agents disagree, call BeliefEngine.query() and inject the result as a signal.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Event:
    idx: int
    description: str
    witnesses: list[str] = field(default_factory=list)  # characters present
    object_: str = ""       # moved/affected object
    location: str = ""      # new location of object after this event
    action: str = ""        # move / see / tell / leave / enter


@dataclass
class ParsedStory:
    events: list[Event]
    characters: list[str]
    objects: list[str]


# ── LLM parse schema (for forced output) ─────────────────────────────────────
# Pass this to the LLM and force JSON output matching this shape.

PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {"type": "array", "items": {"type": "string"}},
        "objects": {"type": "array", "items": {"type": "string"}},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "description": {"type": "string"},
                    "witnesses": {"type": "array", "items": {"type": "string"}},
                    "object_": {"type": "string"},
                    "location": {"type": "string"},
                    "action": {"type": "string"},
                },
                "required": ["idx", "description", "witnesses"],
            },
        },
    },
    "required": ["characters", "objects", "events"],
}


# ── Deterministic engine ──────────────────────────────────────────────────────

class BeliefEngine:
    """
    Deterministic ToM belief tracker based on witness access.

    Example usage:
        engine = BeliefEngine()
        engine.load(parsed_story)
        # 1st-order: what does Sally believe about the ball's location?
        ans = engine.query("Sally", object_="ball")
        # 2nd-order: what does Ann believe Sally believes?
        ans = engine.query("Ann", "Sally", object_="ball")
    """

    def __init__(self):
        self.events: list[Event] = []
        self.characters: list[str] = []
        # belief_map[character][object_] = believed_location
        # Represents each character's current belief after processing all witnessed events
        self.belief_map: dict[str, dict[str, str]] = {}
        # witness_log[character] = list of event indices the character witnessed
        self.witness_log: dict[str, list[int]] = {}

    def load(self, story: ParsedStory) -> None:
        self.events = story.events
        self.characters = story.characters
        self.belief_map = {c: {} for c in story.characters}
        self.witness_log = {c: [] for c in story.characters}
        self._process_events()

    def _process_events(self) -> None:
        for event in self.events:
            for witness in event.witnesses:
                if witness not in self.belief_map:
                    self.belief_map[witness] = {}
                    self.witness_log[witness] = []
                self.witness_log[witness].append(event.idx)
                # Update witness's belief about object location
                if event.object_ and event.location:
                    self.belief_map[witness][event.object_] = event.location

    def query(self, *chain: str, object_: str) -> Optional[str]:
        """
        Order-k belief query via epistemic chain.

        query("A", object_="x")          → 1st-order: what A believes about x
        query("A", "B", object_="x")     → 2nd-order: what A believes B believes
        query("A", "B", "C", object_="x")→ 3rd-order
        """
        if not chain:
            return None
        if len(chain) == 1:
            return (self.belief_map.get(chain[0]) or {}).get(object_)

        outer, inner = chain[0], chain[1:]

        # Build inner[0]'s belief as seen from outer's perspective:
        # Only events that outer witnessed AND inner[0] also witnessed
        outer_witnessed = set(self.witness_log.get(outer, []))
        shared_events = [
            e for e in self.events
            if e.idx in outer_witnessed and inner[0] in e.witnesses
        ]

        # Construct inner[0]'s simulated belief from outer's viewpoint
        simulated: dict[str, str] = {}
        for event in shared_events:
            if event.object_ and event.location:
                simulated[event.object_] = event.location

        if len(inner) == 1:
            return simulated.get(object_)

        # Order 3+: recurse — outer knows inner[0]'s access; apply same logic
        # Build a synthetic BeliefEngine for inner[0] as modeled by outer
        sub = BeliefEngine()
        sub.events = [e for e in self.events if e.idx in outer_witnessed]
        sub.characters = list(inner)
        sub.belief_map = {inner[0]: simulated}
        sub.witness_log = {
            inner[0]: [e.idx for e in shared_events]
        }
        return sub.query(*inner, object_=object_)

    def action_prediction(self, character: str, object_: str, locations: list[str]) -> Optional[str]:
        """
        BigToM extension: where will `character` search for `object_`?
        Returns the believed location if it is one of the provided options.
        """
        believed = (self.belief_map.get(character) or {}).get(object_)
        if believed and believed in locations:
            return believed
        return believed  # return even if not in options; caller decides


# ── Parse helper ──────────────────────────────────────────────────────────────

def parse_story_with_llm(scenario: str, client, model: str, max_tokens: int = 1500) -> Optional[ParsedStory]:
    """
    Call LLM to parse a natural-language scenario into a structured ParsedStory.
    Returns None on parse failure.
    """
    from core.llm_client import call_llm
    import json

    system_prompt = (
        "Parse the ToM scenario into a structured JSON with this exact schema:\n"
        f"{json.dumps(PARSE_SCHEMA, indent=2)}\n\n"
        "Rules:\n"
        "- witnesses: characters physically present when the event occurs\n"
        "- object_: the item being moved or affected (empty if none)\n"
        "- location: new location of object after the event (empty if none)\n"
        "- action: one of move/see/tell/leave/enter/other\n"
        "Output ONLY the JSON object. No markdown fences."
    )

    raw = call_llm(client, model, system_prompt, f"Scenario:\n{scenario}", max_tokens, 0.0)
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    try:
        events = [
            Event(
                idx=e["idx"],
                description=e["description"],
                witnesses=e.get("witnesses", []),
                object_=e.get("object_", ""),
                location=e.get("location", ""),
                action=e.get("action", ""),
            )
            for e in data.get("events", [])
        ]
        return ParsedStory(
            events=events,
            characters=data.get("characters", []),
            objects=data.get("objects", []),
        )
    except (KeyError, TypeError):
        return None
