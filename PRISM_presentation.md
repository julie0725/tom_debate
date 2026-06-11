# PRISM: ToM 추론에 특화된 멀티에이전트 토론 프레임워크

> 종합설계1 8조 중간 발표자료

---

## 목차

1. 문제 상황
2. 연구 소개
3. 시스템 아키텍처 설계 및 설명
4. 데이터셋
5. 기여
6. 향후 계획 및 최종 결과물 기획

---

## 01. 문제 상황

### Theory of Mind (ToM)

> **타인의 생각, 감정, 의도를 추론하는 인간의 능력**

- → 사회적 상호작용의 핵심 기반
- → LLM의 ToM 평가가 중요 과제로 부상

### LLM의 Theory of Mind 추론 한계

- **단일 모델의 구조적 한계 — 자기중심적 편향(Egocentric Bias)**

**핵심 질문:**

> 멀티 에이전트 토론으로 단일 모델 대비 ToM 추론 성능을 개선할 수 있을까?

---

## 02. 연구 소개

### 연구 주제

> **ToM 추론에 특화된 멀티에이전트 토론 프레임워크**

### 핵심 아이디어

| #   | 아이디어          | 설명                                                                      |
| --- | ----------------- | ------------------------------------------------------------------------- |
| 1   | **역할 분리**     | 3개의 Agent 각각 구조화된 추론 전담 / 페르소나 혼동 구조적 방지           |
| 2   | **토론 메커니즘** | 1차 편향 교정 / 답변 불일치 시 추론 공유 및 답변 업데이트                 |
| 3   | **감독관 조율**   | 명령 지시 / 2차 편향 교정 / 하네스 엔지니어링 기법 차용 Context File 보정 |

---

## 03. 시스템 아키텍처 설계 및 설명

### 0) 구성 요소

| 컴포넌트                  | 역할                                 |
| ------------------------- | ------------------------------------ |
| **AI User**               | 태스크 입력                          |
| **Semantic Ego Observer** | 3개의 전문 추론 에이전트             |
| **Supervisor**            | 전체 프로세스 조율 및 최종 답변 결정 |

---

### 1) Overall Architecture

```
AI User → Context File → SuperVisor --Command--> [Semantic / Ego / Observer]
                                                        ↓ Infer (각각)
                                                   Debate Trigger
                                                  /              \
                                         w/o Debate           w/ Debate
                                              ↓                    ↓
                                         Final Answer ←──────────────
```

---

### 2) Debate Architecture

```
Debate Trigger (Yes)
    ↓
[Debate & Update]  ←──────────────────────────────────────
  Ego ↔ Observer ↔ Semantic                               |
    ↓                                                      |
Supervisor → Consensus? ──Yes──→ Final Answer             |
                ↓ No                                       |
           MAX Round? ──No──────────────────────────────────
                ↓ Yes
         Review & File Refinement
                ↓
         [Re-Infer & Update]
          Ego / Observer / Semantic
                ↓
         High-D Majority Consensus → Final Answer
```

**토론 진행 방식:**

| 단계              | 내용                                                      |
| ----------------- | --------------------------------------------------------- |
| 1. 출력값 공유    | 각 Agent 추론 근거 상호 공유 → ToM 답변 업데이트          |
| 2. 재비교         | 감독관이 업데이트된 답변 재비교                           |
| 3. 종료 판단      | 만장일치 → 토론 종료 \| 불일치 → 감독관 오류 분석 & 수정  |
| 4. 수정 후 재추론 | 각 Agent는 수정된 Context File 보고 보정, 재추론 1회 진행 |
| 5. 최종 결정      | 일치 → 종료 \| 불일치 → 다수결 판정 (차수 높은 순)        |

> **Message Pool에서 Context File 저장&수정되는 방식으로 관리**

---

### 3) Message Pool

```
              AI User
                ↓ Publish
         ┌─────────────┐
         │ Message Pool │
         │ Context File │
         └─────────────┘
    ↕Subscribe/Save  ↕Subscribe/Save  ↕Subscribe/Save  ↕Read/Write
     Semantic           Ego            Observer        Supervisor
```

---

### Agent 역할 상세 정의

#### Semantic Agent

- **진실 / 거짓 판단**
- 시나리오 전체 맥락 파악

**출력값:**

- `character_goal` (text)
- `truth_judgment` (true or false)
- `tom_answers` — q1_belief, q2_desire, q3_action

#### Ego Agent

- **인물 내부 상태 출력**
- Belief State 단계별 업데이트

**출력값:**

- `update_log` (text)
- `belief_state` (text)
- `tom_answers` — q1_belief, q2_desire, q3_action

#### Observer Agent

- **고차원 추론**
- 정보 비대칭 분석

**출력값:**

- `update_log` (text)
- `belief_state` (text)
- `tom_answers` — q1_belief, q2_desire, q3_action

#### Supervisor Agent

- 전체 프로세스 조율
- 시스템 전체 흐름 통제
- Context File 기반 수행 명령
- Agent 출력 수집 및 비교
- 토론 진행 관리 및 최종 답변 결정

---

### 토론 Trigger 조건

각 에이전트는 **단답형 정답 출력** → 단어의 일치 여부로 판단

```json
{
  "tom_answer": "K. red_bucket"
}
```

| 조건                | 결과                                                   |
| ------------------- | ------------------------------------------------------ |
| **전부 일치**       | triggered: false / 토론 진행 X / AI User에게 답변 전달 |
| **하나라도 불일치** | triggered: true / 토론 진행 O                          |

---

### 에이전트들이 토론 시 공유하는 내용

**Semantic Agent 출력 예시:**

```json
{
  "truth_judgement": {
    "16": "거짓 (발화 시점 기준, 상추는 red_bucket에 있으나 red_pantry라고 주장)"
  },
  "tom_answer": "K. red_bucket"
}
```

**Ego Agent 출력 예시:**

```json
{
  "update_log": {
    "4": {
      "character": {
        "Sophia": { "belief_state": "red_box (고정)" }
      }
    }
  },
  "tom_answer": "K. red_bucket"
}
```

**Observer Agent 출력 예시:**

```json
{
  "update_log": {
    "1-2": {
      "Sophia": {
        "Isla": { "goal": "상추 위치 추적", "belief": "red_bucket" },
        "Emily": {
          "Sophia": { "goal": "상추 위치 추적", "belief": "red_bucket" }
        }
      }
    }
  },
  "tom_answer": "K. red_bucket"
}
```

---

## 04. 데이터셋

### Big-ToM Dataset

> LLM의 ToM 능력을 체계적으로 평가하기 위한 대규모 데이터셋  
> 다양한 사회적 시나리오 + 시나리오 당 3종류 질문 포함

| 항목                 | 내용                                                                                                                                               |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **데이터 생성 방식** | Casual Template 프레임워크 / ToM 시나리오 → 인과 그래프 표현 / 변수 조작하여 통제 조건 생성 / 템플릿 변수는 LLM이 자동 채움 → 다양한 시나리오 생성 |
| **질문 구성**        | Belief 질문 (믿음) / Desire 질문 (욕구) / Action 질문 (행동)                                                                                       |

---

### Hi-ToM Dataset

> 더 높은 **차수**의 ToM 평가를 위한 객관식 질문 벤치마크  
> Deceptive communication protocol 사용  
> 기만/전략이 필요한 복잡한 상황의 학습에 도움  
> _(차수: 특정 질문의 추론에서 필요한 사람 수)_

| 항목          | 내용                                                                                                                                               |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **상세 내용** | 인물들이 물체를 옮기거나 가만히 두는 행동 / 각 인물 자신의 행동 말할 수 있음(침묵 가능), 거짓섞기(기만) 가능 / 특정 인물이 생각하는 물체 위치 예측 |
| **활용 방법** | 여러 인물의 의견 종합하는 능력 → Multi-Agent에 적합 / 모든 데이터로 모델 학습 → 모델이 거짓말/판별 능력 추가 위해 사용                             |

---

## 05. 기여

| 기여                                                             | 설명                                              |
| ---------------------------------------------------------------- | ------------------------------------------------- |
| **체계적 평가 방법론 확립**                                      | ToM 추론 성능 측정을 위한 표준화된 평가 프로세스  |
| **자기중심적 편향(Egocentric Bias)의 구조적 완화 메커니즘 제시** | 역할 분리 + 토론 + 감독관 조율로 편향 구조적 감소 |
| **ToM 특화 멀티에이전트 프레임워크의 최초 제안**                 | ToM 추론에 특화된 멀티에이전트 아키텍처           |

---

## 06. 향후 계획 및 최종 결과물 기획

- **감독관 알고리즘 구체화 예정**
- **Ablation Study로 평가 진행 → 시스템 성능 확인 예정**

### 주간 계획

| 주차      | 상세 내용                                    |
| --------- | -------------------------------------------- |
| 1주차     | 팀 빌딩                                      |
| 2주차     | 주제 선정 및 아이디어 도출                   |
| 3주차     | ToM 데이터셋 분석 스터디                     |
| 4주차     | ToM 기반 멀티 에이전트 협업 방식 논문 세미나 |
| 5주차     | 알고리즘 제안 및 구체화                      |
| 6주차     | 모델 구축                                    |
| **7주차** | **중간 발표**                                |
| 8주차     | 감독관 알고리즘 구체화                       |
| 9주차     | 감독관 알고리즘 적용 모델 구축 및 평가       |
| 10주차    | Ablation 평가                                |
| 11주차    | Ablation 평가                                |
| 12주차    | 최종 모델 구축                               |
| 13주차    | 보고서 작성                                  |
| 14주차    | 보고서 작성 및 최종 발표 준비                |
| 15주차    | 최종 발표                                    |
