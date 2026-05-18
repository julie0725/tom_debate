from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ToMEvent:
    id: int
    text: str
    observed_by: list
    type: str  # "entry" | "exit" | "action" | "state" | "communication"


@dataclass
class CharacterInfo:
    name: str
    exited_at: Optional[int] = None  # event id when exited, None if still present


@dataclass
class BeliefState:
    agent: str
    proposition: str
    value: str
    last_observed_event: int


@dataclass
class CommonToMState:
    events: list          # list[ToMEvent]
    characters: list      # list[CharacterInfo]
    belief_states: list   # list[BeliefState]
    goals: list           # [{"agent": "...", "goal": "..."}]
    reasoning_type: str
    question: str
    gold_answer: Optional[str]
    dataset_id: Optional[str]
    raw_story: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CommonToMState":
        return cls(
            events=[ToMEvent(**e) for e in d.get("events", [])],
            characters=[CharacterInfo(**c) for c in d.get("characters", [])],
            belief_states=[BeliefState(**b) for b in d.get("belief_states", [])],
            goals=d.get("goals", []),
            reasoning_type=d.get("reasoning_type", "1st-order"),
            question=d.get("question", ""),
            gold_answer=d.get("gold_answer"),
            dataset_id=d.get("dataset_id"),
            raw_story=d.get("raw_story", ""),
        )
