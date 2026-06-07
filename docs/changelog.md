# Changelog

---

## [12] _accumulate_flags — my_answer 단수 필드 버그 수정

### 기존 문제

- `_accumulate_flags`가 `rebuttal_out.get("my_answer", "")` (단수 필드)로 답변 변경 여부를 판단
- BigToM은 q1/q2/q3를 `my_answers` (복수 리스트)로 반환 → `my_answer` 항상 빈 문자열
- `changed = bool("") and ...` → 항상 False → 답 바꿨든 안 바꿨든 **무조건 `ignored_critique`**
- supervisor correction에 "전원 모든 비판 무시" 신호가 전달됨
- `_print_rebuttal_phase`도 같은 단수 필드를 읽어 콘솔에 answer가 항상 `?`로 출력됨

### 수정

- `_accumulate_flags`: `rebuttal_out["my_answer"]` 대신 병합 완료된 `state_after.agent_outputs[agent_key].tom_answers` q1 값으로 `new_answer` 산출
- `_print_rebuttal_phase`: 동일하게 `agent_outputs`에서 병합된 q1 값을 출력
- 두 함수 모두 `state` / `agent_outputs`를 인자로 추가 전달

### Files changed

- `supervisor/debate.py`

---

## [11] _split_majority_minority — 전체 질문 기준으로 수정

### 기존 문제

- q1 vote만 기준으로 majority/minority 분류
- q1 합의 + q2/q3 불일치 케이스에서 minority=[] → fallback으로 전원 전체 critique 수신

### 수정

- 모든 질문(q1/q2/q3)에서 불일치 검사
- 하나라도 minority 답변을 가진 에이전트 → minority로 분류
- 모든 질문에서 majority 답변 → majority로 분류

### Files changed

- `supervisor/debate.py`

---

## [10] debate 루프 break 제거 — max_rounds 정상 동작

### 기존 문제

- 라운드 종료 후 minority 존재하면 `break`로 전체 루프 탈출
- `max_rounds=3`으로 설정해도 항상 1라운드만 실행되고 supervisor correction으로 넘어감

### 수정

- `if minority_keys: break` 블록 제거
- 합의 미달 시 다음 라운드로 자연스럽게 진행
- minority 보호 설계(합산 critique 전달, rebuttal 1회)는 rebuttal 단계에서 이미 보장되므로 break 불필요

### Files changed

- `supervisor/debate.py`

---

## [9] 실행 모드 재구성 + 출력 파일 폴더 정리

### 기존 문제

- `--mode batch`와 `--mode full_batch`가 중복으로 존재해 혼란
- md 파일이 output 루트에 누적되어 실행마다 파일이 쌓임
- 데이터셋별 단독 실행 모드 없음

### 수정한 사항

**main.py**
- `--mode full_batch` → `--mode full_system` (BigToM + HiToM 전체 실행)
- `--mode batch` 제거
- `--mode bigtom` 추가 → `outputs/results_bigtom/`
- `--mode hitom` 추가 → `outputs/results_hitom/`
- `full_system` 실행 시 각 데이터셋별 서브폴더: `outputs/results_full_system/{dataset}/`
- json, jsonl, csv 모두 해당 출력 폴더에 번들링
- `DATASET_CONFIGS` 상수 추가 (데이터셋 경로/이름 중앙 관리)

**user/ai_user.py**
- `submit_from_dataset`: `{output_dir}/md/` 서브폴더 생성, 실행 전 이전 md 파일 정리
- `_submit`: md 파일을 `{output_dir}/md/{dataset_id}.md`로 저장 (기존: 루트에 저장)

### 출력 구조

```
outputs/
  results_full_system/
    bigtom/
      md/               ← md 파일 (샘플별)
      results_*.jsonl
      prism_samples.csv
      evaluation_*.json
    hitom/
      md/
      ...
    prism_results.csv   ← 전체 요약
  results_bigtom/
    md/
    ...
  results_hitom/
    md/
    ...
```

### Files changed

- `main.py`
- `user/ai_user.py`

---

## [8] 전 모드 FINAL METRICS 출력 통일 + CSV None 수정

### 기존 문제

- `batch` 모드: `silent=True`로 평가 결과 출력 없음
- `eval` 모드: EVALUATION SUMMARY 형식 (다른 포맷)
- ablation 모드: condition/dataset 없이 evaluate_from_jsonl 호출
- CSV에서 None 값이 `"None"` 문자열로 저장됨 (빈 문자열이어야 함)

### 수정한 사항

**evaluator.py**
- `_print_summary` → `_print_final_metrics`로 교체 (FINAL METRICS 형식)
- `evaluate_from_jsonl`에 `condition` 파라미터 추가 (default: "PRISM")

**main.py**
- `run_batch`: `silent=True` 제거 → FINAL METRICS 출력
- CSV 저장 시 None → 빈 문자열로 변환 (`_none_to_empty`)

**ablation runners** (`no_debate`, `no_agent`, `no_supervisor`, `max_rounds`)
- `evaluate_from_jsonl` 호출 시 `condition=condition["name"]`, `dataset_name=dataset_name` 전달

### Files changed

- `evaluation/evaluator.py`
- `main.py`
- `evaluation/no_debate_ablation.py`
- `evaluation/no_agent_ablation.py`
- `evaluation/no_supervisor_ablation.py`
- `evaluation/max_rounds_ablation.py`

---

## [7] 페르소나 분리 — critique/rebuttal 전 단계 페르소나 유지

### 기존 문제

- critique/rebuttal 호출 시 3 에이전트가 동일한 generic system prompt 사용 → 페르소나 소실
- 토론 단계에서 Agent1(사건 판단), Agent2(믿음 추적), Agent3(관찰자) 역할 구분 없이 동일한 "ToM reasoner"로 동작
- LangGraph 대비 diversity 부재 → critique 관점이 획일화됨

### 수정한 사항

**페르소나 프롬프트 분리** (`prompts/agent{1/2/3}_persona_prompt.txt` 신규)
- 각 에이전트의 역할 정의 + 추론 방법론 규칙 (R_EVIDENCE, R_STRUCT 포함)
- 출력 schema(R8/R9) 제외 → critique/rebuttal 출력 형식과 충돌 방지

**초기 추론 프롬프트 분리** (`prompts/agent{1/2/3}_initial_infer_prompt.txt` 신규)
- 입력 형식 + 출력 schema (R8/R9) 만 포함
- 초기 추론 시 `persona + initial_infer` 조합으로 호출

**base_agent `_load_prompt()` 업데이트** (`agents/base_agent.py`)
- `agent{N}_persona_prompt.txt` + `agent{N}_initial_infer_prompt.txt` 자동 결합
- fallback: 기존 `agent{N}_prompt.txt`

**debate.py 페르소나 주입** (`supervisor/debate.py`)
- `__init__`에서 3개 persona prompt 로드
- critique: `persona[agent_id] + debate_critique_prompt`
- rebuttal: `persona[agent_id] + debate_rebuttal_prompt`

**debate 프롬프트 첫 줄 제거** (`prompts/debate_critique_prompt.txt`, `debate_rebuttal_prompt.txt`)
- generic "You are an expert Theory of Mind reasoner..." 제거 → persona prompt로 대체

### Files changed

- `prompts/agent1_persona_prompt.txt` — 신규
- `prompts/agent2_persona_prompt.txt` — 신규
- `prompts/agent3_persona_prompt.txt` — 신규
- `prompts/agent1_initial_infer_prompt.txt` — 신규
- `prompts/agent2_initial_infer_prompt.txt` — 신규
- `prompts/agent3_initial_infer_prompt.txt` — 신규
- `agents/base_agent.py` — `_load_prompt()` persona+initial_infer 결합
- `supervisor/debate.py` — persona prompt 로드 및 critique/rebuttal 주입
- `prompts/debate_critique_prompt.txt` — 첫 줄 제거
- `prompts/debate_rebuttal_prompt.txt` — 첫 줄 제거

---

## [6] 미반영 항목 보완 (agent 프롬프트 증거 인용 / 라운드 간 발언 순서 고정)

### 기존 문제

- agent1/2/3 초기 추론 프롬프트에 증거 인용 의무 없음 → 근거 없는 추론 허용
- max_rounds=3이면 소수 에이전트가 rebuttal 후에도 다음 라운드에서 다수의 critique를 또 받음 → 수적 압박 반복

### 수정한 사항

**agent1/2/3 초기 추론 프롬프트 R_EVIDENCE 추가** (`prompts/agent1/2/3_prompt.txt`)
- 모든 주장에 스토리 이벤트 번호 또는 문장 직접 인용 의무화

**라운드 간 발언 순서 고정** (`supervisor/debate.py`)
- 소수 에이전트가 rebuttal 완료 후, `_split_majority_minority()`로 소수 존재 확인 시 즉시 debate 루프 종료
- 소수가 항상 마지막 발언으로 끝나는 구조 보장 (합의 미달 시에도)

### Files changed

- `prompts/agent1_prompt.txt` — R_EVIDENCE 추가
- `prompts/agent2_prompt.txt` — R_EVIDENCE 추가
- `prompts/agent3_prompt.txt` — R_EVIDENCE 추가
- `supervisor/debate.py` — 소수 rebuttal 후 debate 루프 break

---

## [5] Debate 품질 개선 (Blind Critique / 다수 critique 합치기 / 가중치 투표 / 증거 인용)

### 기존 문제

- critique 시 다른 에이전트의 최종 답변이 노출되어 논리 검증 대신 답변 동조(sycophancy) 발생
- 소수 에이전트가 다수 에이전트들의 개별 압박을 동시에 수신 → 포지션이 아닌 수적 압박에 굴복
- majority vote에서 쉽게 답을 바꾼 에이전트와 일관된 에이전트가 동일 가중치로 처리됨
- 답변 변경 시 어떤 근거로 바꿨는지 명시 의무 없음 → 근거 없는 동조 허용

### 수정한 사항

**Blind Critique** (`supervisor/debate.py`)
- critique 시 `tom_answers` 제거한 blind_outputs만 전달 → 답변 숨김
- 에이전트가 논리를 공격하게 유도, 답변 동조 차단

**다수 critique 합치기 + 발언 순서 고정** (`supervisor/debate.py`)
- `_split_majority_minority()` 추가: q1 기준 다수/소수 에이전트 분류
- 소수 에이전트: 다수 측 critique를 "Majority position" 단일 메시지로 합산 수신
- 다수 에이전트: 소수 측 critique만 수신 → 발언 순서 자동 고정

**가중치 투표** (`supervisor/debate.py`)
- 초기 답변 유지 에이전트: 가중치 1.0
- 토론 중 답변 변경 에이전트: 가중치 0.5
- 쉽게 흔들린 에이전트의 투표 영향력 감소

**증거 인용 의무화** (`prompts/debate_critique_prompt.txt`, `prompts/debate_rebuttal_prompt.txt`)
- critique: 스토리 문장 직접 인용 + event 번호 함께 명시 의무화
- rebuttal: 답변 변경 시 어떤 근거가 추론의 어느 부분을 반박했는지 명시 의무화

### Files changed

- `supervisor/debate.py` — Blind Critique, majority/minority 분류, 가중치 투표
- `prompts/debate_critique_prompt.txt` — 답변 숨김 안내 + 증거 인용 규칙
- `prompts/debate_rebuttal_prompt.txt` — 답변 변경 시 근거 명시 규칙

---

## [4] 초기 출력 freeze + Supervisor 전체 맥락 확장

### 기존 문제

- 토론 중 `agent_outputs`가 계속 덮어써져 초기 독립 추론 기록이 소실됨
- `_re_reason_fresh()` 재추론 시 오염된 토론 결과와 debate_context가 그대로 전달되어 Anchoring 발생
- Supervisor correction이 `events[]`, `characters[]`만 보고 판단 → 시나리오 원문, belief_states, reasoning_type 없이 교정하여 hallucination 유발
- Supervisor 지침에 고차 ToM(2nd-order+) 기준 없음 → "직접 관찰 증거 없음"으로 정답을 오답 처리

### 수정한 사항

**에이전트별 초기 출력 freeze** (`core/context_file.py`, `supervisor/supervisor.py`, `supervisor/debate.py`)
- `ToMState`에 `initial_agent_outputs` 필드 추가
- 초기 추론 직후 `state.initial_agent_outputs`에 복사본 저장 (토론이 덮어써도 보존)
- `_re_reason_fresh()`: 오염된 `agent_outputs` 대신 `initial_agent_outputs` 사용, `debate_context` 초기화

**Supervisor 전체 맥락 제공** (`supervisor/supervisor.py`, `prompts/supervisor_correction_prompt.txt`)
- Supervisor correction에 `scenario`(원문), `questions`, `reasoning_type`, `belief_states`, `goals` 추가
- Supervisor 프롬프트에 고차 ToM 기준 명시: 2nd-order+ 추론에서 간접 추론으로 얻은 믿음도 유효함

### Files changed

- `core/context_file.py` — `initial_agent_outputs` 필드 추가
- `supervisor/supervisor.py` — 초기 출력 freeze + supervisor correction 맥락 확장
- `supervisor/debate.py` — `_re_reason_fresh()`에서 initial 출력 + 빈 debate_context 사용
- `prompts/supervisor_correction_prompt.txt` — 고차 ToM epistemic access 지침 추가

---

## [3] 독립 버그 3종 수정 (캐시 오염 / q1만 업데이트 / 빈값)

### What changed

**1. 캐시 오염 수정** (`core/extractor.py`, `user/ai_user.py`)
- 캐시 경로를 `outputs/cache/{dataset_id}.json` → `outputs/cache/{dataset_type}/{dataset_id}.json`으로 변경
- BigToM/HiToM 등 서로 다른 데이터셋이 같은 숫자 ID를 가질 때 캐시 파일이 충돌하던 문제 해결
- `submit_from_dataset()`에서 각 task의 metadata에 `dataset_type`(파일명 stem)을 자동 주입

**2. rebuttal이 q1만 업데이트하던 버그 수정** (`supervisor/debate.py`, `prompts/debate_rebuttal_prompt.txt`)
- rebuttal 프롬프트 출력 형식을 `my_answer` (단일 문자) → `my_answers` (전체 질문 리스트)로 변경
- 파싱 로직을 `tom_map` 딕셔너리 기반으로 변경해 q1/q2/q3 전체 업데이트
- 기존 `my_answer` 필드는 fallback으로 유지 (하위 호환)

**3. 빈값 → unknown 처리** (`supervisor/debate.py`)
- `_majority_vote()`: 빈 문자열 투표 필터링, 유효 투표 없으면 `"unknown"` 반환
- `_extract_answer_from_state()`: `_extract_choice_letter()` 결과가 빈값이면 `"unknown"` 반환

### Files changed

- `core/extractor.py` — cache 경로 namespace 추가
- `user/ai_user.py` — dataset_type을 task.metadata에 주입
- `prompts/debate_rebuttal_prompt.txt` — my_answers 리스트 형식으로 변경
- `supervisor/debate.py` — rebuttal 파싱 전체 질문 업데이트 + 빈값 unknown 처리

---

## [2] PRISM_SPEC.md 추가

### What changed

프로젝트 연구 배경·설계 목표·실험 결과 전반을 담은 명세서 `PRISM_SPEC.md`를 루트에 추가.
Claude Code가 코드 수정·확장 판단 시 참조할 수 있도록 개발 동기, 선행 기술, 시스템 아키텍처, 컨텍스트 파일 구조, 트리거, Supervisor 교정 알고리즘, Ablation Study 결과(Hi-ToM 성능 역전 문제 포함)를 정리.

### Files changed

**`PRISM_SPEC.md`** (신규 생성)

---

## [1] Answer Format Unification — Letter-Only Output

### What changed

All three agents previously produced answers in inconsistent formats (e.g., `"K. green_drawer"`, `"K"`, `"green_drawer"`). This caused the supervisor's agreement check to fail on semantically identical answers, triggering unnecessary debate rounds up to the maximum and wasting LLM calls.

Agents now output only the single choice-label letter (e.g., `"A"`, `"K"`) with no period, no answer text, and no extra characters. The supervisor compares these normalized letters directly.

### Files changed

**`prompts/agent1_prompt.txt`, `prompts/agent2_prompt.txt`, `prompts/agent3_prompt.txt`**

- Rule R8 updated: `"output ONLY the single choice letter (e.g., 'K'). Never include the answer text, a period, or any other characters."`
- The `response` field in the output schema updated to: `"<single letter only, e.g. \"K\"> — MUST be ONLY the choice label letter. No period, no text."`

**`prompts/supervisor_prompt.txt`**

- Agreement-check section updated: `"Agent answers are single choice letters (e.g., 'K', 'A', 'B'). Compare letters directly and exactly. 'K' == 'K' → AGREE. 'K' != 'L' → DISAGREE."`

**`supervisor/supervisor.py`**

- Added module-level `_extract_choice_letter()` (regex `r'^([A-Z])(?:\.|\s|$)'`) to strip any residual prefix before passing answers to the supervisor LLM.
- `_call_supervisor()` builds a normalized `answer_table` (letter-only) and includes it in the LLM prompt alongside the full context file.

**`supervisor/debate.py`**

- Added the same `_extract_choice_letter()` helper.
- `_majority_vote()` normalizes each agent's answer through `_extract_choice_letter()` before counting with `Counter`, so votes on `"K"` and `"K. green_drawer"` are not split.

**`main.py`**

- Added `find_choice_letter(answer_text, choices_str)` using **exact match** (not substring) to reverse-map Hi-ToM ground-truth text answers (e.g., `"green_drawer"`) back to their choice label (e.g., `"K"`) for accurate scoring.
- `run_batch()` injects the choices into the question string (`q1 = f"{question}\nChoices: {choices}"`) so the LLM can resolve the letter from the provided list.

### Why

Hi-ToM answer texts are diverse object names. Substring matching caused false positives (e.g., `"drawer"` matching `"green_drawer"` and `"blue_drawer"`). Exact match ensures ground-truth labels are unambiguous. Enforcing letter-only output in agent prompts removes the normalization burden from downstream code and makes agreement checks reliable.

---

## [2] `final_answer=None` Bug Fix — Fallback to Agent Outputs

### What changed

When all agents agreed and the supervisor returned `agreement: true`, the `final_answer` field in the saved result was `None` for some samples. This happened because the supervisor LLM occasionally returns an empty or null `final_answer` object even when it correctly identifies agreement — the LLM treats answer extraction as optional.

A deterministic fallback was added in both extraction paths: if the supervisor result contains no `final_answer`, the code reads the answer directly from the first non-empty agent output.

### Files changed

**`supervisor/supervisor.py` — `_extract_final_answer()`**

Before (no fallback):

```python
def _extract_final_answer(self, supervisor_result: dict) -> ToMAnswers:
    fa = supervisor_result.get("final_answer") or {}
    return ToMAnswers(
        q1_belief=fa.get("q1_belief"),
        q2_desire=fa.get("q2_desire"),
        q3_action=fa.get("q3_action")
    )
```

After (with fallback):

```python
def _extract_final_answer(self, supervisor_result: dict, state: ToMState = None) -> ToMAnswers:
    fa = supervisor_result.get("final_answer") or {}
    q1 = fa.get("q1_belief") or None
    q2 = fa.get("q2_desire") or None
    q3 = fa.get("q3_action") or None

    if not q1 and state is not None:
        for output in (asdict(state).get("agent_outputs") or {}).values():
            if output and output.get("tom_answers", {}).get("q1_belief"):
                answers = output["tom_answers"]
                q1 = q1 or _extract_choice_letter(answers.get("q1_belief", "")) or None
                q2 = q2 or _extract_choice_letter(answers.get("q2_desire", "")) or None
                q3 = q3 or answers.get("q3_action", "") or None
                break

    return ToMAnswers(q1_belief=q1, q2_desire=q2, q3_action=q3)
```

The call site in `Supervisor.run()` was updated to pass `state=state`.

**`supervisor/debate.py` — `_extract_answer()`**

The debate manager has its own answer-extraction method used when consensus is reached during a debate round. The same fallback was added here, and both call sites inside `run_debate()` were updated to pass `state`:

```python
# Before
return self._extract_answer(result)

# After
return self._extract_answer(result, state)
```

### Why

The initial fix only covered the non-debate path (`Supervisor.run()` → `_extract_final_answer()`). A separate code path in `debate.py` (`_extract_answer()`) handles consensus reached during debate rounds and was not covered. Both paths needed the same fix. The fallback is safe because by the time it is reached, all agents have agreed — any agent's output is the correct answer.

---

## [3] Per-Dataset Evaluation Files — No Cross-Run Accumulation

### What changed

Previously all runs appended to a single `outputs/results.jsonl` and wrote a single `outputs/evaluation_summary.json`. Running Hi-ToM after Big-ToM (or rerunning the same dataset) mixed results, making accuracy metrics meaningless.

Results are now written to dataset-specific files, and the results file is deleted at the start of each `run_batch()` call to ensure a clean slate.

| Dataset path                  | Results file           | Evaluation file          |
| ----------------------------- | ---------------------- | ------------------------ |
| `data/hitom/Hi-ToM_data.json` | `results_hitom.jsonl`  | `evaluation_hitom.json`  |
| `data/bigtom/something.json`  | `results_bigtom.jsonl` | `evaluation_bigtom.json` |

The dataset name is derived from the **parent directory name** of the dataset path, so no manual configuration is required.

### Files changed

**`main.py` — `run_batch()`**

Added at the top of the function (before the data load):

```python
dataset_name = Path(dataset_path).parent.name          # e.g. "hitom"
results_file = f"results_{dataset_name}.jsonl"
if "evaluation" not in config:
    config["evaluation"] = {}
config["evaluation"]["results_file"] = results_file    # picked up by AIUser
output_dir = Path(config["evaluation"].get("output_dir", "outputs/"))
results_path = output_dir / results_file
if results_path.exists():
    results_path.unlink()                              # reset before run
```

Updated the evaluator call at the end:

```python
evaluator.evaluate_from_jsonl(
    results_file=results_file,
    output_file=f"evaluation_{dataset_name}.json"
)
```

**`user/ai_user.py` — `_save_result()`**

Changed from a hardcoded filename to reading from config:

```python
# Before
log_path = self.output_dir / "results.jsonl"

# After
results_file = self.config.get("evaluation", {}).get("results_file", "results.jsonl")
log_path = self.output_dir / results_file
```

**`evaluation/evaluator.py` — `evaluate_from_jsonl()`**

Extended signature to accept optional `results_file` and `output_file` parameters while preserving backward compatibility (defaults keep existing behavior for `ablation.py`):

```python
# Before
def evaluate_from_jsonl(self, jsonl_path: str = None) -> dict:

# After
def evaluate_from_jsonl(self, jsonl_path: str = None, results_file: str = None, output_file: str = "evaluation_summary.json") -> dict:
```

Resolution order for the input path: explicit `jsonl_path` → `results_file` relative to `output_dir` → default `results.jsonl`.

### Why

Evaluation metrics were silently inflated or deflated depending on which previous runs had appended to the shared file. Separating by dataset name makes each run's metrics self-contained and reproducible without requiring manual file cleanup between experiments.

---

## [4] Adapter Pattern 도입 — 데이터 수집 레이어 OCP 준수

### What changed

기존에는 `main.py`와 `ai_user.py`가 Big-ToM/Hi-ToM 데이터 포맷을 직접 파싱했다. 새 데이터셋 추가 시 파이프라인 코드를 직접 수정해야 했다(OCP 위반).

Adapter 패턴을 도입하여 **새 데이터셋 추가 = 어댑터 파일 1개 + `__init__.py` 2줄**만 수정하면 되도록 분리했다.

### Files changed

**`core/tom_task.py` (신규 생성)**

파이프라인과 어댑터 사이의 단일 계약 인터페이스:

```python
@dataclass
class ToMTask:
    context: str
    question: str
    gold_answer: Optional[str] = None
    dataset_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
```

**`data/adapters/dataset_adapter.py` (신규 생성)**

추상 기반 클래스. `load() -> Iterator[ToMTask]` 구현을 강제:

```python
class DatasetAdapter(ABC):
    @abstractmethod
    def load(self) -> Iterator[ToMTask]: ...
```

**`data/adapters/bigtom_adapter.py` (신규 생성)**

Big-ToM 통합 JSON 파일을 읽어 `ToMTask`로 변환. gold answer가 이미 알파벳 레터.

**`data/adapters/hitom_adapter.py` (신규 생성)**

Hi-ToM 통합 JSON을 읽어 `ToMTask`로 변환. `_resolve_answer()`로 텍스트 답변을 선택지 레터로 역매핑.

**`data/adapters/__init__.py` (신규 생성)**

어댑터 레지스트리 및 팩토리. 새 데이터셋 추가 시 이 파일에만 2줄 추가:

```python
REGISTRY: dict[str, type[DatasetAdapter]] = {
    "bigtom": BigToMAdapter,
    "hitom": HiToMAdapter,
}
def get_adapter(dataset_name: str, path: str) -> DatasetAdapter: ...
```

**`user/ai_user.py`**

`submit()` 시그니처 변경: 개별 파라미터 → `ToMTask` 단일 객체 수용.
`questions` 리스트와 `ground_truth`를 `ToMTask`로부터 내부에서 조립.

**`main.py` — `run_batch()`**

기존 수동 파싱 코드 제거. `get_adapter(dataset_name, path)` 호출 후 `adapter.load()` 이터레이션:

```python
adapter = get_adapter(dataset_name, dataset_path)
for task in adapter.load():
    ai_user.submit(task)
```

**`evaluation/ablation.py`**

루프 내부에서 `ToMTask`를 직접 생성하여 `ai_user.submit(task)` 호출.

### Why

데이터셋별 파싱 로직이 파이프라인 코드에 섞여 있으면 실험마다 `main.py`를 건드려야 한다. 어댑터를 레지스트리로 관리하면 파이프라인은 `ToMTask`만 알고, 데이터 포맷 차이는 어댑터가 흡수한다.

---

## [5] Data-Independence 리팩토링 — 고정 필드 → 동적 리스트 스키마

### What changed

`ToMState.questions`가 `{"q1": "...", "q2": "...", "q3": "..."}` 고정 딕셔너리, `ToMAnswers`가 `q1_belief`/`q2_desire`/`q3_action` 고정 필드였다. Big-ToM의 3문항 구조를 하드코딩한 것으로, 문항 수가 다른 데이터셋에서 코드 전체를 수정해야 했다(OCP 위반).

두 자료구조를 **동적 리스트 기반**으로 교체했다.

| 구분 | Before | After |
|------|--------|-------|
| `ToMState.questions` | `{"q1": "text", "q2": "text", "q3": "text"}` | `[{"id": "q1", "text": "..."}, ...]` |
| `ToMAnswers` | `q1_belief`, `q2_desire`, `q3_action` 고정 필드 | `answers: list` — `[{"id": "q1", "value": "A"}, ...]` |

### Files changed

**`core/context_file.py`**

- `ToMAnswers` 필드를 `answers: list`로 교체. `.get_value(question_id)` 메서드 추가.
- 모듈 레벨 공유 헬퍼 `get_answer_value(tom_answers, question_id)` 추가 (list/legacy dict 양쪽 처리).
- `_load_tom_answers(raw)` — 구버전 dict 포맷을 list로 마이그레이션하는 역직렬화 헬퍼.
- `ToMState.questions`를 list로 교체. `from_json()`에서 구버전 dict를 자동 변환.
- `to_markdown()`이 고정 키 접근 대신 리스트 이터레이션으로 변경.

**`user/ai_user.py`**

`submit()`에서 `questions` 리스트와 `ToMAnswers(answers=[...])` 조립 방식 변경:

```python
questions = [{"id": "q1", "text": task.question}]
gt_answers = [{"id": "q1", "value": task.gold_answer}]
ground_truth = ToMAnswers(answers=gt_answers)
```

**`agents/base_agent.py`**

- `reason()` 추상 메서드에서 `debate_context` 파라미터 제거 (모든 호출부에서 항상 `None`이었음).
- `_build_tom_answers(parsed, state_dict)` 공통 메서드 추가 — `tom_answers` 리스트를 파싱 결과에서 추출.

**`agents/agent1_context.py`, `agent2_character.py`, `agent3_perspective.py`**

- `reason(self, state_dict)` — `debate_context` 파라미터 제거.
- `self._build_tom_answers(parsed, state_dict)` 사용.
- 반환값 `"tom_answers"` 필드가 list 형태.

**`supervisor/supervisor.py`**

- `run_in_executor` 호출에서 `None` 인자 제거.
- `_call_supervisor()`: `answer_table`을 `{agent_key: {q_id: letter}}` 구조로 구성 (tom_answers 리스트 이터레이션).
- `_extract_final_answer()`: `_parse_final_answer()` + `get_answer_value()` 사용, `ToMAnswers(answers=[...])` 반환.

**`supervisor/debate.py`**

- `run_in_executor` 호출에서 `None` 인자 제거.
- `_majority_vote()`: question ID를 에이전트 출력에서 동적으로 탐색 (하드코딩된 q1/q2/q3 제거).
- `_extract_answer()`: `_parse_final_answer()` + `get_answer_value()` 사용.

**`prompts/agent1_prompt.txt`, `agent2_prompt.txt`, `agent3_prompt.txt`**

- INPUT 섹션: `Questions:{questions_list}` — `{"id","text"}` 객체 리스트 형태로 설명.
- R8(agent1), R8(agent2), R5b(agent3): 입력 questions_list의 각 질문에 대해 `tom_answers`에 항목을 추가하도록 지시.
- OUTPUT SCHEMA에 root-level `tom_answers` 배열 추가.

**`prompts/supervisor_prompt.txt`**

- Step 1: 에이전트 답변이 `tom_answers` 리스트(`[{"id": "q1", "value": "K"}, ...]`) 형태로 기술.
- `final_answer` 출력 형식: dict → list (`[{"id": "q1", "value": "..."}, ...]`).

**`evaluation/evaluator.py`**

- `_to_map(answers_dict)` 헬퍼 추가: 신버전 list와 구버전 dict 양쪽을 `{q_id: value}` 맵으로 변환.
- `evaluate_single()`: `_to_map()`으로 정규화 후 비교. `_match()`에서 빈 문자열도 False 처리.

**`core/run_logger.py`**

- `from core.context_file import get_answer_value` 임포트 추가.
- `log_agent_outputs()`: `tom_answers.get("q1_belief")` → `get_answer_value(tom_ans_raw, "q1")`.
- `_print_agent_summary()`: `full.get("q2_desire")` → `get_answer_value(full, "q2")`.
- `_format_debate_context()`: `tom_answers.get("q1_belief")` → `get_answer_value(..., "q1")`.
- `_extract_tom_answers()`: 기본값 `{}` → `[]`.
- `_print_debate_round()`: `ans.get('q1_belief')` → `get_answer_value(ans, "q1")`.
- `log_final_summary()`: `fa.get('q1_belief')` → `fa.get("answers", [])` 후 `get_answer_value()`.

**`main.py` — `run_single()`**

```python
# Before
print(f"  Q1 (Belief)    : {result.final_answer.q1_belief}")

# After
print(f"  Q1 (Belief)    : {result.final_answer.get_value('q1')}")
```

### Why

고정 3문항 구조는 데이터셋마다 문항 수가 다를 수 있다는 전제를 무시했다. 리스트 기반 스키마로 전환하면 문항이 1개든 5개든 파이프라인 코드 변경 없이 동작한다. 구버전 JSONL 결과 파일 호환성은 `_to_map()`, `from_json()`, `_load_tom_answers()`의 자동 마이그레이션으로 유지된다.
