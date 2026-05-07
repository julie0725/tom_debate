"""
context_file.py
---------------
ToMState: 시스템 전체에서 공유되는 데이터 구조 (context file의 Python 표현)
JSON 직렬화/역직렬화 지원
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


def get_answer_value(tom_answers, question_id: str) -> str:
    """Extract a single question's value from tom_answers in either format.

    Accepts list format [{"id": "q1", "value": "A"}, ...]
    or legacy dict format {"q1_belief": "A", "q2_desire": "", "q3_action": ""}.
    """
    if isinstance(tom_answers, list):
        for a in tom_answers:
            if a.get("id") == question_id:
                return a.get("value", "")
    elif isinstance(tom_answers, dict):
        _legacy = {"q1": "q1_belief", "q2": "q2_desire", "q3": "q3_action"}
        return tom_answers.get(_legacy.get(question_id, question_id), "")
    return ""


@dataclass
class ToMAnswers:
    """Dataset-independent answer container.

    answers: list of {"id": "q1", "value": "A"} dicts, one per question asked.
    """
    answers: list = field(default_factory=list)

    def get_value(self, question_id: str) -> Optional[str]:
        for a in self.answers:
            if a.get("id") == question_id:
                return a.get("value") or None
        return None

    def to_dict(self):
        return asdict(self)

    def is_complete(self) -> bool:
        return bool(self.answers) and all(a.get("value") for a in self.answers)

    def matches(self, other: "ToMAnswers") -> bool:
        for a in self.answers:
            if a.get("value") != other.get_value(a.get("id")):
                return False
        return True


@dataclass
class AgentOutput:
    """각 Agent의 출력값 (타입 힌트용; 런타임에는 raw dict로 저장됨)"""
    agent_id: int
    character_goal: Optional[str] = None
    truth_judgment: Optional[bool] = None
    update_log: list = field(default_factory=list)
    belief_state: Optional[str] = None
    tom_answers: list = field(default_factory=list)   # [{"id": "q1", "value": "A"}, ...]
    reasoning: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ToMState:
    """
    전역 메시지 풀에서 공유되는 Context File의 Python 표현
    모든 에이전트가 이 객체를 읽고, 감독관만 업데이트함
    """
    # 입력
    scenario: str = ""
    questions: list = field(default_factory=list)
    # [{"id": "q1", "text": "..."}, {"id": "q2", "text": "..."}, ...]

    # 에이전트 출력
    agent_outputs: dict = field(default_factory=lambda: {
        "agent1": None,
        "agent2": None,
        "agent3": None
    })

    # 토론 상태
    debate_round: int = 0
    status: str = "pending"
    debate_triggered: bool = False
    majority_vote_applied: bool = False

    debate_context: dict = field(default_factory=dict)
    supervisor_correction: Optional[str] = None

    # 최종 답변
    final_answer: ToMAnswers = field(default_factory=ToMAnswers)

    # 추론 메타데이터
    reasoning_type: Optional[str] = None   # "0th-order" | "1st-order" | "2nd-order" | "3rd-order"
    characters: list = field(default_factory=list)
    common_state: Optional[dict] = None    # CommonToMState.to_dict()

    # 메타데이터
    dataset_id: Optional[str] = None
    ground_truth: Optional[ToMAnswers] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# ToM Context File",
            f"**Status**: {self.status}",
            f"**Debate Round**: {self.debate_round}",
            "",
            "## Scenario",
            self.scenario,
            "",
            "## Questions",
        ]
        for q in self.questions:
            lines.append(f"- {q.get('id', '?')}: {q.get('text', '')}")
        lines += ["", "## Agent Outputs"]
        for agent_id, output in self.agent_outputs.items():
            if output:
                lines.append(f"### {agent_id}")
                lines.append(f"- Goal: {output.get('character_goal', 'N/A')}")
                lines.append(f"- Belief State: {output.get('belief_state', 'N/A')}")
                for a in (output.get("tom_answers") or []):
                    lines.append(f"- {a.get('id', '?')}: {a.get('value', 'N/A')}")
        if self.supervisor_correction:
            lines += ["", "## Supervisor Correction", self.supervisor_correction]
        lines += ["", "## Final Answer"]
        for a in self.final_answer.answers:
            lines.append(f"- {a.get('id', '?')}: {a.get('value', '')}")
        return "\n".join(lines)

    @classmethod
    def from_json(cls, json_str: str) -> "ToMState":
        data = json.loads(json_str)
        state = cls()
        state.scenario = data.get("scenario", "")
        state.debate_round = data.get("debate_round", 0)
        state.status = data.get("status", "pending")
        state.debate_triggered = data.get("debate_triggered", False)
        state.majority_vote_applied = data.get("majority_vote_applied", False)
        state.debate_context = data.get("debate_context", {})
        state.supervisor_correction = data.get("supervisor_correction", None)
        state.dataset_id = data.get("dataset_id")
        state.agent_outputs = data.get("agent_outputs", {})

        # questions: migrate old dict {"q1": "...", "q2": "...", "q3": "..."} → list
        raw_q = data.get("questions", [])
        if isinstance(raw_q, dict):
            state.questions = [{"id": k, "text": v} for k, v in raw_q.items() if v]
        else:
            state.questions = raw_q

        state.reasoning_type = data.get("reasoning_type", None)
        state.characters = data.get("characters", [])
        state.common_state = data.get("common_state", None)
        state.final_answer = _load_tom_answers(data.get("final_answer"))
        gt = data.get("ground_truth")
        state.ground_truth = _load_tom_answers(gt) if gt else None
        return state


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_tom_answers(raw) -> ToMAnswers:
    """Deserialise ToMAnswers from either the new list schema or the legacy dict schema."""
    if raw is None:
        return ToMAnswers()
    if isinstance(raw, dict) and "answers" in raw:
        return ToMAnswers(answers=raw["answers"])
    if isinstance(raw, dict):
        # Legacy: {"q1_belief": "A", "q2_desire": "B", "q3_action": "..."}
        mapping = [
            ("q1_belief", "q1"), ("q2_desire", "q2"), ("q3_action", "q3"),
        ]
        answers = [{"id": qid, "value": raw[old]} for old, qid in mapping if raw.get(old)]
        return ToMAnswers(answers=answers)
    if isinstance(raw, list):
        return ToMAnswers(answers=raw)
    return ToMAnswers()
