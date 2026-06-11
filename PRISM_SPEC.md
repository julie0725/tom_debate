# PRISM — 프로젝트 명세서

> Theory of Mind 추론 특화 멀티에이전트 토론 프레임워크의 연구 배경, 설계 목표, 실험 결과 전반을 정리한 문서.
> Claude Code가 코드 수정·확장 판단에 활용할 수 있도록 작성.

---

## 개발 동기

인간은 사회적 상호작용에서 타인의 신념(Belief)·욕구(Desire)·의도(Intention)를 추론한다. 인지과학에서 이를 **Theory of Mind(ToM)** 라고 부르며, 협업·의사소통·교육·상담 등 광범위한 상황에서 핵심 역할을 한다.

인간은 영유아기부터 뇌 발달 과정에서 ToM을 자연 습득하지만, LLM은 별도 학습·추론 메커니즘 없이는 이 능력을 갖출 수 없다.

**LLM이 보이는 주요 한계:**

- **False Belief 과제 실패** — 타인이 자신과 다른 신념을 가질 수 있다는 사실을 의사결정에 반영하지 못해, 4세 아동도 통과하는 기본 ToM 과제에서도 불안정한 성능을 보임
- **표현 변화에 취약** — 시나리오 구조가 조금만 바뀌어도 성능 급락 → 표면적 패턴에 의존하고 있을 가능성
- **자기중심적 편향** — 단일 LLM은 자신이 아는 정보를 등장인물 관점에 투영하며, 다수 인물·기만·오해 등 복잡한 사회 상황에서 잘못된 믿음을 교정하지 못함

→ **해결 방향**: 단일 LLM을 보완하는 멀티에이전트 시스템. 쌍방향 추론으로 편향 최소화, 추가 교정으로 오류 보정.

---

## 목적 및 필요성

기존 ToM 연구는 단일 모델 평가나 프롬프트 개선에 집중. Chain-of-Thought·Self-Reflection·Debate 같은 기법도 일반 추론 성능 개선에 초점을 둘 뿐, ToM의 자기중심적 편향을 직접 해결하지는 못한다.

**실 환경에서 요구되는 것:**
- 인간 개입 없이 스스로 추론 오류를 발견·수정하는 능력
- 서로 다른 관점의 에이전트가 상호 검증·토론으로 오류를 교정하는 협업 구조

**사회적 배경 — AI Transformation(AX) 사회:**
AI가 개인 의사결정 보조·기업 업무 협업의 핵심 주체로 변화 중. AI가 사용자의 신념·의도·욕망을 잘못 추론할 경우 부적절한 조언과 잘못된 의사결정 지원으로 이어짐.

> 사례: 2023년 Air Canada 챗봇이 실제로 존재하지 않는 환불 정책을 안내 → 법적 분쟁 발생

→ 단일 AI 답변에 의존하지 않는 **교차 검증(Cross-check)·상호 견제·오류 교정 시스템** 필요.

---

## 개발 목표

**구조화된 토론으로 ToM 추론에 특화된 멀티에이전트 토론 프레임워크** 설계·구현.

"구조화된 토론" = 세 가지 다른 역할의 에이전트 + 토론 + Supervisor의 보정으로, **데이터셋 정답에 의존하는 것이 아닌** 논리성에 기반한 토론 방식.

- **Semantic Agent** — 맥락적 관점에서 시나리오의 현재 상황 추적
- **Ego Agent** — 등장인물의 내부 관점 추론
- **Observer Agent** — 제3자 관점에서 추론 결과 검토
- **Supervisor Agent** — 세 에이전트 관리·감독·보정 (하네스 엔지니어링 차용)

목표:
1. 단일 LLM의 자기중심적 편향 완화
2. ToM 추론 정확도·안전성 향상
3. Big-ToM·Hi-ToM 벤치마크로 정량 검증 및 기존 방식과 비교

---

## 선행 기술

| 구분 | 프레임워크 | 모델 |
|------|-----------|------|
| 범용 에이전트 | LangGraph | GPT-3.5-turbo |
| 범용 에이전트 | CAMEL-AI (AI User + AI Assistant 역할 분담) | GPT-3.5-turbo |
| ToM 특화 | Multi-agent Debate | GPT-3.5-turbo / GPT-3.5-0301 (현재 deprecated) |
| ToM 특화 | AutoGen (인간 개입 가능) | GPT-3.5-turbo |

---

## 핵심 아이디어

1. **역할 분리** — 서로 다른 페르소나를 가진 3명의 토론 에이전트 + 1명의 Supervisor
2. **토론 메커니즘** — 초기 추론 불일치 시 Critique → Rebuttal → Consensus 3단계 자동 트리거
3. **Supervisor 보정** — 에이전트 출력 기반으로 오류 분석 및 보정

---

## 핵심 구성 요소

### 비(非)에이전트

| 구성 요소 | 역할 |
|-----------|------|
| AI User | 입출력 담당 인터페이스 |

### 에이전트

| 에이전트 | 역할 |
|---------|------|
| Semantic Agent | 맥락 정보 출력. 현재 상황을 이벤트·진실 관점에서 분석 |
| Ego Agent | 인물 내부 관점 정보 출력. 등장인물의 belief state 추적 |
| Observer Agent | 타인 관점에서 인물 정보 출력. 고차원 ToM 추론 |
| Supervisor | 토론 출력 로그 기반 논리성 오류 보정 |

---

## 시스템 아키텍처

### 전체 파이프라인

```
사용자/데이터셋 입력
  │
  ▼
AI User 인터페이스
  └─ 시나리오 정형화 → Common ToM State 추출 → Context File 생성
     (정답 masked)
  │
  ▼
Supervisor
  └─ 세 에이전트에게 지시(command)
  │
  ├─ Agent1 (Semantic) ─┐
  ├─ Agent2 (Ego)       ├─ 병렬 추론(infer)
  └─ Agent3 (Observer) ─┘
  │
  ▼
Debate Trigger: 세 답변 일치 여부 확인
  ├─ 일치 → 최종 결과 도출
  └─ 불일치 → 토론 진입
```

### 토론 아키텍처

```
Debate Trigger 활성화
  │
  ▼
[Round N] Critique Phase — 각 에이전트가 타 에이전트 추론 비판
  │
  ▼
[Round N] Rebuttal Phase — 비판에 답변 후 답변 갱신
  │
  ▼
합의 확인 (Python 규칙 기반, LLM 없음)
  ├─ 합의 → 최종 답 도출
  └─ 미합의
       ├─ max_rounds 미도달 → 다음 Round
       └─ max_rounds 도달 → Supervisor 호출
            ├─ 토론 과정 Review + 맥락 파일 보정
            ├─ 세 에이전트 재추론
            └─ 고차원 다수결 합의 → 최종 답 도출
```

### 공유 저장소(Message Pool) 아키텍처

```
AI User → 시나리오 정보 → Common ToM State 추출 → Message Pool 저장

토론 에이전트(Semantic, Ego, Observer) + Supervisor
  ├─ Message Pool에서 맥락 정보 읽기(subscribe)
  └─ 추론 결과 저장(save)

Supervisor 논리 보정 필요 시:
  └─ correction 추가 작성 → 에이전트가 재추론
```

---

## 컨텍스트 파일 (Context File)

에이전트들이 공유하는 문맥 정보 = Common ToM State. 총 2단계 전처리 후 Message Pool에 저장.

### 1단계: 형태 통일 (Adapter 패턴 → ToMTask)

```
raw_story: "1 Avery, Charlotte ... entered living.\n2 Avery moved lettuce to green_bathtub.\n3 Owen moved lettuce to green_drawer."
question:  "Where is the lettuce really? Choices A. blue_drawer ... K. green_drawer"
gold_answer: "K"
dataset_id: 1
metadata: {question_order, characters, event_count}
```

### 2단계: 내용 공통 변수 추출 (Extractor → Common ToM State)

```
events:        {id, text, observed_by, type}           # 진입/퇴장 추적으로 observed_by 결정
belief_states: {agent, proposition, value, last_observed_event}  # 목격 기반 믿음값
characters:    {name, exited_at}                       # 퇴장 이벤트 번호로 관측 범위 결정
goals:         {agent, goal}
reasoning_type: "1st-order"
```

### 최종 저장

`outputs/cache/{dataset_id}.json` — 동일 샘플 재실행 시 LLM 호출 없이 재사용.

---

## 트리거

| 트리거 | 조건 | 방식 |
|--------|------|------|
| 토론 트리거 | 세 에이전트 단답형 `tom_answer` 중 하나 이상 불일치 | DebateManager 스케줄러, Python 문자열 비교 |
| 교정 트리거 | max_rounds까지 토론 진행 후 합의 미달 | Supervisor LLM 1회 호출 |

---

## Supervisor 교정 알고리즘

### 현재 구현: Flag Accumulation + Lazy Supervisor Call

- LLM 호출 없이 Python 규칙으로 매 라운드 합의 확인
- max_rounds 초과 시에만 Supervisor LLM 1회 호출
- Infer → Critique → Rebuttal 후 Python이 자동으로 Flag 누적:
  - `accepted_critique` — 비판받고 답변 변경
  - `ignored_critique` — 비판받고 답변 유지

**Supervisor에게 전달되는 정보:**

| 전달 O | 전달 X (정답 유추 방지) |
|--------|----------------------|
| `flags` (수용/무시 누적) | `scenario` (원본 스토리) |
| `agent_outputs.reasoning` | `questions` (정답 힌트 포함) |
| `agent_outputs.belief_state` | `common_state.belief_states` |
| `common_state.events[]` | `common_state.goals` |
| `common_state.characters[]` | `gold_answer` |

**Supervisor 역할 (현재 → 목표):**

| 항목 | 현재 | 목표 방향 |
|------|------|----------|
| 역할 | 정답을 추론 후 교정 | 논리 오류만 지적 |
| 출력 | "올바른 추론 방향은 이것이다" | "이 추론이 이 관찰 사실과 모순된다" |
| 에이전트 영향 | 정답 방향으로 수렴 강제 | 추론 과정 자체를 재점검하도록 유도 |

### 알려진 문제 — Hi-ToM 성능 역전

`no_supervisor` 조건이 `full_system`보다 Hi-ToM 정확도가 높아지는 성능 역전 발생. 원인 분석 및 알고리즘 개선 진행 중 (→ 아래 Ablation Study 5번 참고).

---

## 실험 결과 분석 (선행 기술 비교)

### 사용 데이터셋

| 데이터셋 | 샘플 수 | 특징 |
|---------|--------|------|
| Big-ToM | 400 | 1차원, True/False Belief 포함 |
| Hi-ToM | 1,200 | 다수 인물, 기만(deception) 포함, 고차원 ToM |

### 평가 척도

| 지표 | 정의 |
|------|------|
| Q1 Belief accuracy | 선택지 일치 (대소문자 무관) |
| Q2 Desire accuracy | 선택지 일치 |
| Q3 Action accuracy | 키워드 매칭 (gt 키워드 50% 이상 포함) |
| Joint accuracy | Q1+Q2+Q3 모두 정답인 샘플 비율 |
| Conflicts (debate trigger rate) | 초기 불일치로 토론에 진입한 샘플 비율 |
| Avg Rounds | 전체 샘플 기준 평균 토론 라운드 (토론 없는 경우 포함) |
| Avg Rounds among debate | 토론 진입 샘플에 한한 평균 라운드 |
| Majority vote rate | 다수결 투표 적용 비율 |
| avg_elapsed_sec | 총 실험 시간 / 전체 샘플 수 |
| throughput_per_hour | 3600 / avg_elapsed_sec |
| avg_cost_per_sample | 총 비용 / 샘플 수 |
| total_cost_usd | (입력 토큰 × 단가) + (출력 토큰 × 단가) 합산 |

### 추론 정확도

| 시스템 | BigToM Joint | Hi-ToM Q1(Belief) | 특이사항 |
|--------|-------------|-------------------|---------|
| Multiagent Debate | 44.25% | 35.75% | Q1 73.00%, Q2 52.75%, Q3 55.50% |
| AutoGen (Conversation) | 72.25% | 26.08% (Joint) | Hi-ToM에서 급락. GroupChatRoundRobin 특성상 첫 발언 오답 시 이후 동조 |
| CAMEL (Role-Playing) | 46.75% | 37.42% (Joint) | Q1 63%, Q2 58.9%, Q3 61.5% |
| LangGraph Orchestration | (추후 기재) | — | — |

### 처리 시간

| 시스템 | Big-ToM 400샘플 | Hi-ToM 1,200샘플 | 특이사항 |
|--------|----------------|-----------------|---------|
| Multiagent Debate | 1시간 미만 | 1시간 30분 미만 | gpt-3.5-turbo + temperature=0 |
| AutoGen | 빠름 | 빠름 | 토론 라운드 없음 → 오류 교정 기회 없음 |
| CAMEL | (추후 기재) | — | — |
| LangGraph | ~1시간 48분 | ~5시간 15분 | 샘플당 평균 15~16초. conflict_rate 0%임에도 매우 느림 |

### 비용

| 시스템 | Big-ToM | Hi-ToM | 합계 |
|--------|--------|--------|------|
| Multiagent Debate | $0.59 | $2.20 | $2.79 |
| AutoGen | $0.64 | $1.87 | $2.51 |
| CAMEL | $3.11 | $1.51 | $4.62 |
| LangGraph | $0.89 | $3.54 | $4.43 |

---

## Ablation Study

구성 요소 개별 기여도를 확인하기 위한 제거 실험.

- **에이전트 1개 제거** → 전체 시스템에 기여하는가?
- **에이전트 2개 제거 (1개만 유지)** → 해당 에이전트 역할 자체가 유효한가?

### 1. 토론 에이전트 1개 제거

- 세 에이전트는 각각 다른 ToM 차원을 담당, 잉여 없음 확인
- Observer 제거 시 충돌 증가: Hi-ToM 22.5%, Big-ToM 30.33% → Observer가 의견 통합에 결정적
- Hi-ToM에서 Observer 역할이 더 두드러짐. Big-ToM(1차원)에서는 Ego의 역할도 중요

### 2. 토론 에이전트 2개 제거 (단일 에이전트 실험)

- Big-ToM: Observer 단독 88% Joint Accuracy로 가장 우수 (제3자 관점이 1차원 믿음 추론에 유리)
- Hi-ToM: Semantic=58%, Observer=58%, Ego=53% → 세 에이전트 팀(61.67%)보다 모두 낮음
- 고차원 ToM일수록 단일 에이전트는 한계. 관점 교환이나 모든 역할을 아우르는 프롬프트 필요

### 3. 토론 최대 횟수(max_rounds) 통제

짝수 라운드는 tie-break 문제로 홀수(0, 1, 3, 5)로 실험.

| max_rounds | Big-ToM | Hi-ToM | 비고 |
|-----------|--------|--------|------|
| 0 | 기준 | 기준 | 토론 없음 |
| 1 | +9.0% | +6.7% | 유의미한 향상 |
| 3 | +2.5% | +6.7% | **최적** |
| 5 | +0.5% | -5.0% | Hi-ToM 성능 급락 |

- **결론**: max_rounds=3이 최적. 5라운드 이상에서 Ego Agent가 타 에이전트에 의존하며 자기결단력 상실 → 의존 편향 발생

### 4. 페르소나 제거

- 에이전트 역할(페르소나) 제거 시 Big-ToM Joint 87% → 47%로 절반 폭락
- 의미 관점(Semantic)·자아 관점(Ego)·타자 관점(Observer) 페르소나가 성능 결정 핵심 요인

### 5. Supervisor 제거 — 성능 역전 문제 ⚠️

`no_supervisor`가 `full_system`보다 Hi-ToM 추론 정확도가 **높게** 나오는 성능 역전 발생.

**현재 분석된 원인:**

| 원인 | 설명 |
|------|------|
| Anchoring Bias | 재추론 시 이전 `agent_outputs` (3라운드 토론 후 답변)이 state_dict에 그대로 포함 → 진정한 from-scratch 재추론 불가 |
| 정보 역설 | Supervisor는 scenario·belief_states 없이 events[]·characters[]만으로 복잡한 Hi-ToM 오류 진단 → 잘못된 correction 생성 가능 |
| False Consensus | 잘못된 correction → 세 에이전트 모두 같은 틀린 답으로 수렴 → agreement=True + 오답 |
| Flag 신뢰도 한계 | accepted/ignored만으로 어떤 에이전트가 틀렸는지 판단 불가 |

**검토 중인 수학적 개선 방향:**

| 방향 | 방법 | 비고 |
|------|------|------|
| Anchoring 제거 | 재추론 시 agent_outputs 초기화 | 코드 1줄, 즉시 적용 가능 |
| 안정성 가중 투표 | `reliability_i = 1 - |stability_i - 0.5| * 2` 로 가중 투표 | flag 재활용, 추가 비용 없음 |
| Entropy 기반 개입 | `H = -Σ p_k·log₂(p_k)` 감소 추세면 Supervisor 보류 | 과도한 개입 차단 |
| 코사인 유사도 | reasoning 임베딩 벡터 중심성으로 신뢰도 산출 | 임베딩 API 비용 발생 |
| 선택적 재추론 | suspicion score 기반 최소 에이전트만 재추론 | 전체 재추론 대신 표적 교정 |
