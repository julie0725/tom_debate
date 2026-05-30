# PRISM — 감독관 알고리즘 구현 및 진행 현황

> 종합설계1 8조 | 8~9주차 진행 보고

---

## 이 시스템의 차별점

### 1. 데이터셋 독립성 — 정답 없이 논리를 확인한다

기존 멀티에이전트 시스템은 **정답과 비교**해서 성능을 높임  
→ 정답이 없는 현실 문제에서는 작동 불가

PRISM은 감독관이 **시나리오와 정답을 보지 않음**  
→ 에이전트들의 추론 논리가 일관적인지만 판단  
→ 정답 없는 현실 문제에서도 작동

```
기존: 감독관이 정답 참고 → "A가 맞다, B가 틀렸다"
PRISM: 감독관이 논리만 확인 → "A의 추론이 이벤트 관찰 기록과 일치한다"
```

### 2. 구조화된 토론 — 무한 루프 방지

단순 반복 토론은 에이전트가 의견을 바꾸지 않으면 무한 루프  
→ PRISM은 **플래그 누적 + Lazy Supervisor** 구조로 해결

```
매 라운드: Python으로 합의 확인 (LLM 없음)
  → 합의 시: 즉시 종료
  → 불합의 시: 플래그 누적 (누가 비판을 무시했는가?)

MAX Round 초과:
  → Supervisor가 플래그 + 추론만 보고 오류 진단
  → 재추론 → 그래도 불합의 → 다수결
```

---

## 구현 내용

---

## 1. 감독관(Supervisor) 알고리즘

### Lazy Supervisor 개념

> **필요할 때만 LLM을 호출한다**

| 단계                         | 방식                    | LLM 호출  |
| ---------------------------- | ----------------------- | --------- |
| 매 라운드 합의 확인          | Python 문자열 비교      | ❌        |
| MAX rounds 초과 시 오류 진단 | LLM (flags + reasoning) | ✅ 딱 1회 |

기존 방식 대비 LLM 호출 수: `max_rounds × 에이전트 수 + 1` → `에이전트 수 + 1`

---

### Flag 누적 알고리즘

매 라운드 Rebuttal 후, 에이전트의 행동 패턴을 플래그로 기록

```
[Round N] Agent1이 Agent2를 비판
  → Agent2가 답변을 바꿨는가?
    Yes → accepted_critique  (비판을 수용)
    No  → ignored_critique   (비판을 무시)
```

누적된 플래그 예시:

```json
[
  {
    "round": 1,
    "critic": "agent1",
    "target": "agent2",
    "flag": "ignored_critique",
    "old_answer": "B",
    "new_answer": "B"
  },
  {
    "round": 2,
    "critic": "agent2",
    "target": "agent3",
    "flag": "accepted_critique",
    "old_answer": "A",
    "new_answer": "B"
  }
]
```

---

### 감독관 오류 진단 — 시나리오 차단

Supervisor는 **정답 유추 가능한 정보를 보지 않음**

| 전달 O                              | 전달 X                     |
| ----------------------------------- | -------------------------- |
| flags (누적 플래그)                 | scenario (원본 스토리)     |
| agent reasoning (추론 체인)         | questions (선택지 포함)    |
| agent belief_state (에이전트 해석)  | common_state.belief_states |
| events[] / characters[] (관찰 기록) | gold_answer                |

프롬프트 제약:

```
agent_outputs.belief_state is each agent's own interpreted belief, not ground truth.
Evaluate only whether it is logically derivable from events[].observed_by.
Do NOT confirm or infer the correct answer.
```

---

## 2. 핵심 구현 코드

---

### (1) 데이터셋은 어떻게 들어오는가 — Adapter + Proxy

**문제**: Big-ToM(CSV)과 Hi-ToM(JSON)은 포맷이 다름  
**해결**: Adapter Pattern으로 포맷 추상화

```
입력
  │
  ▼
Proxy.get_tasks(입력)
  ├─ 파일 경로? → detect_adapter() → CsvAdapter / JsonAdapter
  └─ 텍스트?   → TextAdapter (LLM으로 시나리오/질문 추출)
  │
  ▼
ToMTask (context, question, gold_answer, metadata)
  — 파이프라인은 이 형태만 받음 (데이터셋 무관)
```

Big-ToM CSV 처리 핵심:

```python
# CsvAdapter — 1행 = 2개 태스크 (true_belief / false_belief)
# belief/desire/action 질문을 하나의 태스크에 번들
meta["q2"] = f"{desire_q}\nChoices: A. {d_aware}, B. {d_not}"
meta["q3"] = f"{action_q}\nChoices: A. {a_aware}, B. {a_not}"
```

---

### (2) Context File이란 무엇인가 — ToMState

모든 에이전트와 감독관이 공유하는 **전역 상태 객체**

```python
@dataclass
class ToMState:
    scenario: str           # 시나리오 텍스트
    questions: list         # [{"id": "q1", "text": "..."}, ...]
    agent_outputs: dict     # {"agent1": {...}, "agent2": {...}, "agent3": {...}}
    debate_round: int
    debate_context: dict    # 라운드별 critique/rebuttal 내용
    supervisor_correction: str  # 감독관 오류 분석 결과
    final_answer: ToMAnswers
    common_state: dict      # Extractor가 추출한 구조화 상태
```

추가된 구조화 레이어 — **CommonToMState** (Extractor 추출):

```
events[]       — 이벤트 ID, 텍스트, 관찰자 목록, 유형
characters[]   — 인물명, 퇴장 시점
belief_states[]— 에이전트, 명제, 믿음값, 마지막 관찰 이벤트
reasoning_type — 0th / 1st / 2nd / 3rd-order
```

에이전트는 원본 텍스트 대신 이 구조를 primary source로 사용

---

### (3) 토론을 진행하는 코드 — DebateManager

```
run_debate()
  │
  for round in 1..max_rounds:
  │   ├─ Critique Phase  : 각 에이전트가 타 에이전트 추론 비판
  │   ├─ Rebuttal Phase  : 비판에 답변 후 자신의 답변 갱신
  │   ├─ _accumulate_flags()  ← 플래그 누적
  │   └─ _check_agreement()  ← Python 합의 확인 (LLM 없음)
  │        └─ 합의 → return final answer
  │
  MAX Round 초과
  ├─ supervisor_correction_fn(state, flags)  ← Lazy Supervisor
  ├─ 에이전트 재추론 (from scratch)
  ├─ _check_agreement()
  │    └─ 합의 → return final answer
  └─ _majority_vote()  ← 다수결 (동점 시 Observer 우선)
```

합의 확인 (Python, LLM 없음):

```python
def _check_agreement(self, state):
    answer_map = {}
    for agent_key, output in state.agent_outputs.items():
        for a in (output.get("tom_answers") or []):
            answer_map.setdefault(a["id"], {})[agent_key] = a["value"]

    agreement = all(
        len(set(votes.values())) == 1
        for votes in answer_map.values()
    )
    return agreement, answer_map
```

---

### (4) 감독관 오류 진단 보정 알고리즘

MAX rounds 초과 후 1회 호출. 시나리오 없이 논리만 판단:

```python
def _call_supervisor_correction(self, state, flags):
    # 필터링: 정답 유추 가능한 정보 제외
    filtered_outputs = {
        agent_key: {
            "reasoning": output["reasoning"],
            "belief_state": output["belief_state"],  # 에이전트 해석 (정답 아님)
            "tom_answers": output["tom_answers"],
        }
    }
    filtered_common = {
        "events": common["events"],          # 관찰 기록 (구조화)
        "characters": common["characters"],  # 진입/퇴장 타이밍
        # belief_states 제외 — 정답 힌트
    }
    # flags + 추론 + 관찰 기록만 전달
    # → Supervisor: "누가 논리적으로 일관적인가?" 판단
```

오류 진단 후 에이전트는 **처음부터 재추론** (correction을 guidance로 활용)

---

## 3. Ablation Study 확인 사항

### 실험 조건 (6가지)

| 조건            | Semantic | Ego | Observer | Debate |
| --------------- | -------- | --- | -------- | ------ |
| **full_system** | ✅       | ✅  | ✅       | ✅     |
| no_debate       | ✅       | ✅  | ✅       | ❌     |
| agent1_only     | ✅       | ❌  | ❌       | ❌     |
| agent2_only     | ❌       | ✅  | ❌       | ❌     |
| agent3_only     | ❌       | ❌  | ✅       | ❌     |
| no_agent3       | ✅       | ✅  | ❌       | ✅     |

### 확인하려는 것

**① 토론이 실제로 효과가 있는가?**

- `full_system` vs `no_debate` 비교
- 가설: 토론 있을 때 accuracy 향상, 특히 2nd/3rd-order

**② 어떤 에이전트가 가장 기여하는가?**

- `agent1_only` / `agent2_only` / `agent3_only` 단독 성능 비교
- 가설: Observer(agent3)가 고차원 문제에서 결정적 역할

**③ Observer 없으면 얼마나 떨어지는가?**

- `full_system` vs `no_agent3`
- 가설: 2nd-order 이상에서 유의미한 차이

**④ 측정 지표**

| 지표                | 설명                  |
| ------------------- | --------------------- |
| Q1 Belief accuracy  | belief 질문 정답률    |
| Q2 Desire accuracy  | desire 질문 정답률    |
| Q3 Action accuracy  | action 질문 정답률    |
| Joint accuracy      | 모든 질문 동시 정답률 |
| Debate trigger rate | 토론 발생 비율        |
| Avg debate rounds   | 평균 토론 라운드 수   |

---

## 현재까지 구현 완료

| 항목                                            | 상태             |
| ----------------------------------------------- | ---------------- |
| 3-Agent 병렬 추론                               | ✅               |
| Adapter Pattern (CSV / JSON / 자연어 텍스트)    | ✅               |
| Extractor + CommonToMState (구조화 추출 + 캐시) | ✅               |
| Critique / Rebuttal 토론 루프                   | ✅               |
| Python 기반 합의 확인 (LLM 없음)                | ✅               |
| Flag 누적 (ignored / accepted critique)         | ✅               |
| Lazy Supervisor (MAX rounds 초과 시만 LLM)      | ✅               |
| 시나리오 차단 오류 진단                         | ✅               |
| 다수결 (동점 시 Observer 우선)                  | ✅               |
| Ablation Study 러너                             | 🔲 업데이트 필요 |
| 전체 데이터셋 평가                              | 🔲 진행 예정     |
