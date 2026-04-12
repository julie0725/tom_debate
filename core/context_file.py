"""
context_file.py
---------------
ToMState: 시스템 전체에서 공유되는 데이터 구조 (context file의 Python 표현)
JSON 직렬화/역직렬화 지원
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ToMAnswers:
    """Big-ToM 3가지 질문에 대한 답변"""
    q1_belief: Optional[str] = None    # Belief 질문 (이지선다)
    q2_desire: Optional[str] = None    # Desire 질문 (이지선다)
    q3_action: Optional[str] = None    # Action 질문 (개방형)

    def to_dict(self):
        return asdict(self)

    def is_complete(self) -> bool:
        return all([self.q1_belief, self.q2_desire, self.q3_action])

    def matches(self, other: "ToMAnswers") -> bool:
        """두 답변이 일치하는지 확인 (q3 action은 핵심 키워드 기준)"""
        q1_match = self.q1_belief == other.q1_belief
        q2_match = self.q2_desire == other.q2_desire
        # Action은 개방형이라 완전 일치보다 핵심 의미 비교 (일단 문자열 일치로 처리)
        q3_match = self.q3_action == other.q3_action
        return q1_match and q2_match and q3_match


@dataclass
class AgentOutput:
    """각 Agent의 출력값"""
    agent_id: int
    character_goal: Optional[str] = None
    truth_judgment: Optional[bool] = None       # Agent 1 전용
    update_log: list = field(default_factory=list)  # Agent 2, 3 전용
    belief_state: Optional[str] = None          # Agent 2, 3 전용
    tom_answers: ToMAnswers = field(default_factory=ToMAnswers)
    reasoning: Optional[str] = None             # 추론 과정 (토론 시 공유됨)

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class ToMState:
    """
    전역 메시지 풀에서 공유되는 Context File의 Python 표현
    모든 에이전트가 이 객체를 읽고, 감독관만 업데이트함
    """
    # 입력
    scenario: str = ""
    questions: dict = field(default_factory=lambda: {
        "q1": "",   # Belief 질문
        "q2": "",   # Desire 질문
        "q3": ""    # Action 질문
    })

    # 에이전트 출력
    agent_outputs: dict = field(default_factory=lambda: {
        "agent1": None,
        "agent2": None,
        "agent3": None
    })

    # 토론 상태
    debate_round: int = 0
    status: str = "pending"             # pending | debating | done
    debate_triggered: bool = False
    majority_vote_applied: bool = False

    # 최종 답변
    final_answer: ToMAnswers = field(default_factory=ToMAnswers)

    # 메타데이터 (논문 실험용)
    dataset_id: Optional[str] = None
    ground_truth: Optional[ToMAnswers] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        """디버깅용 MD 형식 출력"""
        lines = [
            "# ToM Context File",
            f"**Status**: {self.status}",
            f"**Debate Round**: {self.debate_round}",
            "",
            "## Scenario",
            self.scenario,
            "",
            "## Questions",
            f"- Q1 (Belief): {self.questions.get('q1', '')}",
            f"- Q2 (Desire): {self.questions.get('q2', '')}",
            f"- Q3 (Action): {self.questions.get('q3', '')}",
            "",
            "## Agent Outputs",
        ]
        for agent_id, output in self.agent_outputs.items():
            if output:
                lines.append(f"### {agent_id}")
                lines.append(f"- Goal: {output.get('character_goal', 'N/A')}")
                lines.append(f"- Belief State: {output.get('belief_state', 'N/A')}")
                answers = output.get('tom_answers', {})
                lines.append(f"- Q1: {answers.get('q1_belief', 'N/A')}")
                lines.append(f"- Q2: {answers.get('q2_desire', 'N/A')}")
                lines.append(f"- Q3: {answers.get('q3_action', 'N/A')}")
        lines.append("")
        lines.append("## Final Answer")
        lines.append(f"- Q1: {self.final_answer.q1_belief}")
        lines.append(f"- Q2: {self.final_answer.q2_desire}")
        lines.append(f"- Q3: {self.final_answer.q3_action}")
        return "\n".join(lines)

    @classmethod
    def from_json(cls, json_str: str) -> "ToMState":
        data = json.loads(json_str)
        state = cls()
        state.scenario = data.get("scenario", "")
        state.questions = data.get("questions", {})
        state.debate_round = data.get("debate_round", 0)
        state.status = data.get("status", "pending")
        state.debate_triggered = data.get("debate_triggered", False)
        state.majority_vote_applied = data.get("majority_vote_applied", False)
        state.dataset_id = data.get("dataset_id")
        state.agent_outputs = data.get("agent_outputs", {})
        fa = data.get("final_answer", {})
        state.final_answer = ToMAnswers(
            q1_belief=fa.get("q1_belief"),
            q2_desire=fa.get("q2_desire"),
            q3_action=fa.get("q3_action")
        )
        return state
