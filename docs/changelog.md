# Changelog

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
