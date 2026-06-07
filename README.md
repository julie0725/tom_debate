# PRISM — Multi-Agent Theory of Mind Debate System

Theory of Mind 추론을 위한 멀티에이전트 토론 프레임워크

---

## 설치

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_api_key" > .env
```

**의존성**: `openai==1.76.0`, `pyyaml==6.0.2`, `python-dotenv`

---

## 실행

# 기본 실행 : 데이터셋 전체 실행 — BigToM + HiToM 동시 실행

```bash
python main.py --mode full_system #전체
```

# 개별 실행 : 데이터셋 선택 가능

```bash
python main.py --mode bigtom # BigToM 단독
python main.py --mode hitom  # HiToM 단독
```

# 테스트 샘플 수 설정 가능

```bash
python main.py --mode bigtom --limit 10
```

# 추가 명령어

```bash
# 단일 자연어 입력 (터미널에서 직접 입력)
python main.py --mode single
```

```bash
# 저장된 결과만 평가
python main.py --mode eval
```

## 실행 결과 확인

```
full_system → outputs/results_full_system/{bigtom|hitom}/
bigtom → outputs/results_bigtom/
hitom → outputs/results_hitom/
```

> 전처리 스크립트 불필요 — Adapter가 원본 파일을 직접 읽음

---

## 프로젝트 구조

```
tom_debate/
├── main.py                              # 진입점 (single/batch/eval)
├── config/
│   └── config.yaml                      # 모델·토론·에이전트 설정
├── core/
│   ├── context_file.py                  # ToMState / ToMAnswers 데이터 구조
│   ├── common_state.py                  # CommonToMState (구조화 추출 결과)
│   ├── extractor.py                     # LLM 기반 구조 추출 + 파일 캐시
│   ├── tom_task.py                      # ToMTask (데이터 계층 ↔ 파이프라인 계약)
│   ├── llm_client.py                    # LLM provider 추상화 (OpenAI / Gemini / Custom)
│   ├── message_pool.py                  # 전역 Pub-Sub 메시지 풀 (thread-safe)
│   └── run_logger.py                    # 실행 로그 시스템
├── data/
│   ├── adapters/
│   │   ├── __init__.py                  # detect_adapter() (확장자 기반 자동 감지)
│   │   ├── base.py                      # DatasetAdapter / TextDatasetAdapter (추상 베이스)
│   │   ├── csv_adapter.py               # CsvAdapter — Big-ToM CSV 직접 로드
│   │   ├── json_adapter.py              # JsonAdapter — Hi-ToM JSON 직접 로드
│   │   ├── text_adapter.py              # TextAdapter — 자연어 원문 → ToMTask (LLM 추출)
│   │   └── proxy.py                     # Proxy — 파일 경로 vs 텍스트 자동 라우팅
│   ├── bigtom/
│   │   └── bigtom.csv                   # Big-ToM 원본 (세미콜론 구분)
│   └── hitom/
│       └── Hi-ToM_data.json             # Hi-ToM 원본
├── agents/
│   ├── base_agent.py                    # 공통 베이스 (LLM 호출, 프롬프트, 질문 섹션 주입)
│   ├── agent1_context.py                # Semantic Agent — 맥락 분석 / 진실·거짓 판단
│   ├── agent2_character.py              # Ego Agent — 인물 관점 / belief state 추적
│   └── agent3_perspective.py            # Observer Agent — Mode A/B 고차원 ToM
├── supervisor/
│   ├── supervisor.py                    # Supervisor — 병렬 추론 / Python 합의 체크 / 토론 trigger
│   └── debate.py                        # DebateManager — 토론 루프 / flag 누적 / Lazy Correction
├── user/
│   └── ai_user.py                       # AI User — 단일 입력 게이트웨이
├── prompts/
│   ├── agent1_prompt.txt
│   ├── agent2_prompt.txt
│   ├── agent3_prompt.txt                # Mode A (Internal Observer) / Mode B (Nested Simulator)
│   ├── debate_critique_prompt.txt
│   ├── debate_rebuttal_prompt.txt
│   └── supervisor_correction_prompt.txt
├── evaluation/
│   ├── evaluator.py                     # 정량 평가 (per-question accuracy, joint accuracy)
│   └── ablation.py                      # 6가지 ablation 조건 자동 실험
└── outputs/
    ├── results_<dataset>.jsonl          # 실험 결과 누적
    ├── evaluation_<dataset>.json        # 평가 요약
    ├── cache/<dataset_id>.json          # Extractor LLM 호출 캐시
    └── logs/<dataset_id>/               # 샘플별 상세 로그 (context 스냅샷, 토론 라운드)
```

---

## 시스템 아키텍처

### 전체 파이프라인

```
입력 (파일 경로 or 자연어 텍스트)
  │
  ▼
Proxy
  ├─ 파일 경로 → detect_adapter() → CsvAdapter / JsonAdapter
  └─ 텍스트 → TextAdapter (LLM 추출)
  │
  ▼
ToMTask (context, question, gold_answer, metadata)
  │
  ▼
Extractor  →  CommonToMState (events, characters, belief_states, goals)
  │              └─ outputs/cache/<id>.json 에 캐시 (재실행 시 LLM 호출 없음)
  ▼
ToMState → MessagePool.publish()
  │
  ▼
Supervisor
  ├─ Agent1 / Agent2 / Agent3 병렬 추론 (asyncio)
  │
  ├─ [Python] _check_agreement()
  │    ├─ 전원 일치 → final answer 확정
  │    └─ 불일치 → Debate Loop
  │
  └─ Debate Loop (max_rounds)
       ├─ [Round N] Critique Phase — 각 에이전트가 타 에이전트 추론 비판
       ├─ [Round N] Rebuttal Phase — 비판에 답변 후 답변 갱신
       ├─ [Python] _check_agreement() — 합의 확인 (LLM 없음)
       ├─ flag 누적: ignored_critique / accepted_critique
       │
       ├─ 합의 → final answer 확정
       └─ MAX rounds 초과 → [Lazy Supervisor Correction]
            ├─ flags + agent reasoning + events/characters 전달
            │   (scenario / questions / belief_states 제외 — 정답 유추 방지)
            ├─ 에이전트 재추론 (from scratch)
            ├─ 재합의 확인
            └─ 최후 수단: 다수결 (동점 시 Observer 우선)
```

---

## 핵심 설계

### 1. Adapter Pattern

데이터셋 포맷에 상관없이 파이프라인은 `ToMTask` 하나만 받음.

| Adapter       | 입력                | 특징                                               |
| ------------- | ------------------- | -------------------------------------------------- |
| `CsvAdapter`  | Big-ToM `.csv`      | 행당 2개 태스크 (true/false_belief), q1/q2/q3 번들 |
| `JsonAdapter` | Hi-ToM `.json`      | 직접 로드, q_order → reasoning_type 변환           |
| `TextAdapter` | 자연어 원문         | LLM으로 scenario/question/characters 추출          |
| `Proxy`       | 파일 경로 or 텍스트 | 자동 라우팅 — 파이프라인은 Proxy만 알면 됨         |

```python
# 모든 입력 경로가 동일하게 처리됨
proxy.get_tasks("data/bigtom/bigtom.csv")   # 파일
proxy.get_tasks("Sally moved the ball...")   # 자연어
```

### 2. Extractor + CommonToMState

매 `ToMTask`마다 LLM이 구조화된 ToM 상태를 추출. 에이전트는 원본 텍스트 대신 이 구조를 사용.

```
CommonToMState
  events[]       — id, text, observed_by[], type
  characters[]   — name, exited_at
  belief_states[]— agent, proposition, value, last_observed_event
  goals[]        — agent, goal
  reasoning_type — 0th / 1st / 2nd / 3rd-order
```

캐시: `outputs/cache/<dataset_id>.json` — 동일 시나리오 재실행 시 LLM 호출 없음

### 3. 에이전트 (PRISM)

세 에이전트는 동일한 `ToMState`를 보고 독립적으로 추론.

| 에이전트                    | 역할                             | 핵심 출력                                             |
| --------------------------- | -------------------------------- | ----------------------------------------------------- |
| **Semantic Agent** (Agent1) | 이벤트 진실/거짓 판단, 목표 추론 | `truth_judgment`, `tom_answers`                       |
| **Ego Agent** (Agent2)      | 인물별 belief state 추적         | `update_log`, `belief_state`, `tom_answers`           |
| **Observer Agent** (Agent3) | Mode A/B 고차원 추론             | `internal_layers` / `simulation_chain`, `tom_answers` |

**Observer Agent Mode 분기:**

- **Mode A (Internal Observer)**: `len(characters) == 1` 또는 `1st-order` — 내부 심리 레이어 분석 (rational / emotional / instinctive)
- **Mode B (Nested Perspective Simulator)**: `len(characters) >= 2` AND `2nd-order+` — 중첩 관점 체인 시뮬레이션

### 4. Lazy Supervisor + Flag Accumulation

매 라운드 합의 확인은 Python 규칙 기반 (LLM 없음). Supervisor LLM은 **MAX rounds 초과 시에만** 한 번 호출.

**Flag 누적** (매 Rebuttal 후):

```
critic → target 쌍마다:
  - 비판 받고 답변 바꿈  → "accepted_critique"
  - 비판 받고 답변 유지  → "ignored_critique"
```

**Supervisor Correction — 시나리오 차단:**

| 전달 O                       | 전달 X                       |
| ---------------------------- | ---------------------------- |
| `flags` (누적 플래그)        | `scenario` (원본 스토리)     |
| `agent_outputs.reasoning`    | `questions` (정답 힌트 포함) |
| `agent_outputs.belief_state` | `common_state.belief_states` |
| `common_state.events[]`      | `common_state.goals`         |
| `common_state.characters[]`  | `gold_answer`                |

> Supervisor는 시나리오를 모르고 논리적 일관성만 판단 → 정답 없는 현실 문제에서도 작동

**Supervisor Correction 역할 변화:**

|                 | 현재                          | 변경                                |
| --------------- | ----------------------------- | ----------------------------------- |
| supervisor 역할 | 정답을 추론 후 교정           | 논리 오류만 지적                    |
| 출력 내용       | "올바른 추론 방향은 이것이다" | "이 추론이 이 관찰 사실과 모순된다" |
| 에이전트 영향   | 정답 방향으로 수렴 강제       | 추론 과정 자체를 재점검하도록 유도  |

---

## 평가

```
Q1 Belief accuracy  — 선택지 일치 (대소문자 무관)
Q2 Desire accuracy  — 선택지 일치
Q3 Action accuracy  — 키워드 매칭 (gt 키워드 50% 이상 포함)
Joint accuracy      — ground_truth에 있는 질문 모두 정답 시
```

질문이 없는 항목(예: Big-ToM에서 desire 조건 동일 시 q2 없음)은 `N/A`로 표시되며 joint accuracy 계산에서 제외.

---

## 데이터셋 비교

| 항목           | Big-ToM                         | Hi-ToM             |
| -------------- | ------------------------------- | ------------------ |
| 입력 파일      | `bigtom.csv` (세미콜론 구분)    | `Hi-ToM_data.json` |
| 질문 수/샘플   | 최대 3개 (belief/desire/action) | 1개                |
| reasoning_type | 1st-order                       | 0th ~ 3rd-order    |
| 선택지         | A/B 이지선다                    | A~O 다지선다       |
| 등장인물       | 1명                             | 다수 (5명+)        |
| 조건           | true_belief / false_belief      | 없음               |

---

## 모델 설정

`config/config.yaml`에서 `provider`만 바꾸면 전환 완료:

| Provider | 환경변수         | 모델 예시                 |
| -------- | ---------------- | ------------------------- |
| `openai` | `OPENAI_API_KEY` | `gpt-4o`, `gpt-3.5-turbo` |
| `gemini` | `GEMINI_API_KEY` | `gemini-1.5-flash`        |
| `custom` | `CUSTOM_API_KEY` | Ollama 등 OpenAI 호환     |

---

## Ablation 조건

| 조건        | Semantic | Ego | Observer | Debate |
| ----------- | -------- | --- | -------- | ------ |
| full_system | O        | O   | O        | O      |
| no_debate   | O        | O   | O        | X      |
| agent1_only | O        | X   | X        | X      |
| agent2_only | X        | O   | X        | X      |
| agent3_only | X        | X   | O        | X      |
| no_agent3   | O        | O   | X        | O      |

결과: `outputs/ablation/<조건명>/`, 비교: `outputs/ablation/ablation_comparison.json`

---

## 설계 결정 사항

| 항목                       | 결정                           | 이유                                           |
| -------------------------- | ------------------------------ | ---------------------------------------------- |
| 합의 체크                  | Python 규칙 기반               | 문자열 일치 비교에 LLM 불필요                  |
| Lazy Supervisor            | MAX rounds 초과 시만 LLM       | 불필요한 LLM 호출 최소화                       |
| Supervisor — 시나리오 차단 | flags + reasoning만 전달       | 정답 유추 방지 → 정답 없는 현실 문제 적용 가능 |
| Flag 누적                  | ignored / accepted critique    | Supervisor가 디베이트 패턴 분석 가능           |
| Extractor 캐시             | `outputs/cache/`               | 동일 시나리오 반복 실험 시 LLM 비용 절감       |
| Observer Mode A/B          | characters 수 + reasoning_type | 단일 인물 내면 분석 vs 다인물 관점 체인 분리   |
| Agent 동시 실행            | asyncio + run_in_executor      | 순수 Python 병렬성, LangGraph 미사용           |
| 동점 tiebreak              | Observer 우선                  | 고차원 추론 전담                               |
| 결과 저장                  | jsonl append                   | 실험 중단 후 재시작해도 누적 가능              |
