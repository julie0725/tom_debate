# ToM Multi-Agent Debate System

Theory of Mind 추론을 위한 멀티에이전트 토론 프레임워크

---

## 설치

```bash
pip install -r requirements.txt
# .env 파일에 API 키 설정
echo "OPENAI_API_KEY=your_api_key" > .env
```

**의존성**: `openai==1.76.0`, `pyyaml==6.0.2`, `python-dotenv`

---

## 실행

```bash
# 전처리 먼저 실행 (최초 1회)
python data/preprocess.py

# 단일 샘플 테스트 (Sally-Anne 예시)
python main.py --mode single

# Hi-ToM 전체 실행
python main.py --mode batch --dataset data/hitom/hitom_unified.json

# Big-ToM 전체 실행
python main.py --mode batch --dataset data/bigtom/bigtom_unified.json

# Ablation study
python main.py --mode ablation --dataset data/hitom/hitom_unified.json

# 저장된 결과만 평가
python main.py --mode eval
```

---

## 프로젝트 구조

```
tom_debate/
├── main.py                             # 진입점 (single/batch/ablation/eval)
├── config/
│   └── config.yaml                     # 모델·토론·에이전트 설정
├── core/
│   ├── context_file.py                 # ToMState / ToMAnswers 데이터 구조
│   ├── llm_client.py                   # LLM provider 추상화 (OpenAI / Gemini / Custom)
│   ├── message_pool.py                 # 전역 Pub-Sub 메시지 풀 (thread-safe)
│   └── run_logger.py                   # 실행 로그 시스템 (스냅샷·토론 라운드·요약)
├── agents/
│   ├── base_agent.py                   # 공통 베이스 (LLM 호출, 프롬프트 로드)
│   ├── agent1_context.py               # Semantic Agent: 맥락 분석 (진실/거짓 판단)
│   ├── agent2_character.py             # Ego Agent: 인물 관점 (belief state 추적)
│   └── agent3_perspective.py           # Observer Agent: 고차원 ToM (2nd-order 이상)
├── supervisor/
│   ├── supervisor.py                   # 감독관 (병렬 실행, 일치 판단, 토론 trigger)
│   └── debate.py                       # 토론 루프 (max_rounds, correction, 다수결)
├── user/
│   └── ai_user.py                      # AI User (context file 생성 및 publish)
├── prompts/
│   ├── agent1_prompt.txt               # Semantic Agent 시스템 프롬프트
│   ├── agent2_prompt.txt               # Ego Agent 시스템 프롬프트
│   ├── agent3_prompt.txt               # Observer Agent 시스템 프롬프트
│   ├── supervisor_prompt.txt           # 감독관 판단 프롬프트
│   └── supervisor_correction_prompt.txt # 감독관 오류 분석 프롬프트
├── evaluation/
│   ├── evaluator.py                    # 정량 평가
│   └── ablation.py                     # 6가지 ablation 조건 자동 실험
├── data/
│   ├── preprocess.py                   # 전처리 스크립트 (Big-ToM/Hi-ToM → unified)
│   ├── hitom/
│   │   ├── Hi-ToM_data.json            # Hi-ToM 원본 (1200 samples)
│   │   └── hitom_unified.json          # 전처리 완료본 (실험 입력)
│   └── bigtom/
│       ├── bigtom.csv                  # Big-ToM 원본 (200 rows, `;` 구분)
│       └── bigtom_unified.json         # 전처리 완료본 (실험 입력)
└── outputs/
    ├── results.jsonl                   # 실험 결과 누적
    ├── evaluation_summary.json         # 평가 요약
    └── logs/<dataset_id>/              # 샘플별 상세 로그
```

---

## 데이터 전처리

두 데이터셋을 **동일한 unified 형식**으로 변환하여 파이프라인에 입력합니다.

```bash
python data/preprocess.py                  # 양쪽 모두 변환
python data/preprocess.py --preview        # 변환 결과 미리보기
```

### Unified 형식 (공통 스키마)

```json
{
  "sample_id": 0,
  "dataset": "hitom | bigtom",
  "question_order": 1,
  "condition": "true_belief | false_belief | null",
  "character": "Avery",
  "event_count": 6,
  "story": "1 Avery entered the room.\n2 ...",
  "question": "Where does Avery think the lettuce is?",
  "choices": "A. green_drawer, B. blue_pantry, ...",
  "answer": "K",
  "answer_text": "green_drawer"
}
```

### Big-ToM 변환 (`bigtom.csv` → `bigtom_unified.json`)

Big-ToM 원본은 5문장 자연어 단락 구조입니다. Hi-ToM과 동일한 **번호 붙은 이벤트 로그** 형식으로 변환합니다.

| 원본 구조              | 변환 결과                |
| ---------------------- | ------------------------ |
| 문장 1: 배경           | `1 <배경 문장>`          |
| 문장 2: 목표           | `2 <목표 문장>`          |
| 문장 3: 초기 행동      | `3 <행동 문장>`          |
| 문장 4: 초기 belief    | `4 <belief 문장>`        |
| 문장 5: 사건 발생      | `5 <사건 문장>`          |
| Aware 여부 (condition) | `6 <목격함 / 목격 못함>` |

- `true_belief` 조건: 6번 이벤트 = 목격함 → 정답 = "A"
- `false_belief` 조건: 6번 이벤트 = 목격 못함 → 정답 = "B"
- 1개 원본 row → 최대 6개 샘플 (2 condition × 3 question_order)
- **question_order 매핑**: `1=belief`, `2=desire`, `3=action`

```
원본 200 rows → 1014 samples
question_order: {1: 400, 2: 214, 3: 400}
condition: {true_belief: 507, false_belief: 507}
```

### Hi-ToM 정규화 (`Hi-ToM_data.json` → `hitom_unified.json`)

Hi-ToM은 이미 번호 이벤트 로그 형식이므로 필드명을 unified 스키마에 맞게 정규화합니다.

- `story` 첫 줄에서 등장인물 목록 자동 추출 (`character` 필드)
- 이벤트 줄 수 카운트 (`event_count` 필드)
- `condition = null` (Hi-ToM은 true/false_belief 조건 구분 없음)

```
원본 1200 samples → 정규화 1200 samples
question_order: {0: 240, 1: 240, 2: 240, 3: 240, 4: 240}
deception: {True: 600, False: 600}
story_length: {1: 400, 2: 400, 3: 400}
```

### 두 데이터셋 비교

| 항목           | Big-ToM                          | Hi-ToM                  |
| -------------- | -------------------------------- | ----------------------- |
| 시나리오 형식  | 자연어 단락 → 이벤트 로그로 변환 | 이벤트 로그 (원본 유지) |
| 질문 수/샘플   | 3개 (belief/desire/action)       | 1개                     |
| question_order | 1, 2, 3                          | 0, 1, 2, 3, 4           |
| 선택지         | A/B 이지선다                     | A~O 다지선다            |
| deception      | 없음                             | 있음 (50%)              |
| 등장인물       | 1명 (단일 character)             | 5명 (다수)              |

---

## 전체 파이프라인 흐름

```
AIUser
  └─ ToMState 생성 → MessagePool.publish()
       └─ Supervisor
            ├─ Semantic / Ego / Observer 병렬 추론 (asyncio)
            ├─ 감독관: tom_answer 일치 판단
            │    ├─ 전원 일치 → 최종 답변 확정
            │    └─ 불일치 → Debate Loop
            │         ├─ MessagePool에 debate_context 저장 → 재추론
            │         ├─ max_rounds 초과 → 감독관 오류 분석(correction) → 재추론
            │         └─ 최후 수단: 다수결 (동점 시 Observer 우선)
            └─ outputs/results.jsonl 저장 + outputs/logs/<id>/ 상세 로그
```

---

## 에이전트 출력 형식 (PRISM 기준)

토론 시 에이전트가 공유하는 핵심 출력:

**Semantic Agent**

```json
{
  "truth_judgment": {
    "5": "FALSE (coworker swapped milk without Noor seeing)"
  },
  "tom_answer": "A"
}
```

**Ego Agent**

```json
{
  "update_log": {
    "3": { "Noor": { "belief_state": "pitcher contains oat milk" } }
  },
  "tom_answer": "A"
}
```

**Observer Agent**

```json
{
  "belief_state": [
    { "target": "Noor", "1st_order": [{ "belief": "oat milk" }] }
  ],
  "tom_answer": "A"
}
```

토론 trigger 조건: `tom_answer` 불일치 시 → debate 시작

---

## 모델 설정

`config/config.yaml`에서 `provider`만 바꾸면 모델 전환 완료:

| Provider | 환경변수         | 모델 예시                 |
| -------- | ---------------- | ------------------------- |
| `openai` | `OPENAI_API_KEY` | `gpt-4o`, `gpt-3.5-turbo` |
| `gemini` | `GEMINI_API_KEY` | `gemini-1.5-flash`        |
| `custom` | `CUSTOM_API_KEY` | Ollama 등 OpenAI 호환     |

---

## Ablation 조건

| 조건        | Semantic | Ego | Observer | Debate |
| ----------- | -------- | --- | -------- | ------ |
| full_system | ✅       | ✅  | ✅       | ✅     |
| no_debate   | ✅       | ✅  | ✅       | ❌     |
| agent1_only | ✅       | ❌  | ❌       | ❌     |
| agent2_only | ❌       | ✅  | ❌       | ❌     |
| agent3_only | ❌       | ❌  | ✅       | ❌     |
| no_agent3   | ✅       | ✅  | ❌       | ✅     |

결과: `outputs/ablation/<조건명>/results.jsonl`, `outputs/ablation/ablation_comparison.json`

---

## 설계 결정 사항

| 항목                | 결정                                   | 이유                                    |
| ------------------- | -------------------------------------- | --------------------------------------- |
| max_rounds          | 3                                      | 비용/시간 vs 성능 균형                  |
| 동점 tiebreak       | Observer 우선                          | 고차원 추론 전담                        |
| Agent 동시 실행     | asyncio + run_in_executor              | 순수 Python 병렬성, LangGraph 미사용    |
| 토론 trigger        | tom_answer 단답 불일치                 | 선택지 레이블 일치 여부로 판단          |
| Q3 Action 평가      | 키워드 매칭 (50% 이상)                 | 개방형 답변, 추후 LLM-judge로 교체 가능 |
| 결과 저장           | jsonl append                           | 실험 중단 후 재시작해도 누적 가능       |
| debate_context 공유 | MessagePool 경유                       | 에이전트 간 결합도 최소화               |
| 감독관 correction   | max_rounds 초과 시 오류 분석 후 재추론 | 다수결 전 마지막 교정 기회              |
