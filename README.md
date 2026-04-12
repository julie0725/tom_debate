# ToM Multi-Agent Debate System

Theory of Mind 추론을 위한 멀티에이전트 토론 프레임워크

---

## 설치

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_api_key
```

---

## 실행

```bash
# 단일 샘플 테스트 (Sally-Anne)
python main.py --mode single

# 데이터셋 전체 실행
python main.py --mode batch --dataset data/bigtom/dataset.jsonl

# Ablation study
python main.py --mode ablation --dataset data/bigtom/dataset.jsonl

# 저장된 결과만 평가
python main.py --mode eval
```

---

## 프로젝트 구조

```
tom_debate/
├── config/config.yaml          # 실험 설정 (max_rounds, ablation 스위치 등)
├── agents/                     # Agent 1/2/3 구현
├── supervisor/                 # 감독관 + Debate loop
├── user/                       # AI User (진입점)
├── core/                       # Context File 스키마 + 메시지 풀
├── prompts/                    # 각 Agent 시스템 프롬프트
├── data/                       # Big-ToM / Hi-ToM 데이터셋
├── evaluation/                 # 정량 평가 + Ablation 러너
├── outputs/                    # 실험 결과 저장
└── main.py                     # 진입점
```

---

## 데이터셋 형식

`data/bigtom/dataset.jsonl` 각 줄:

```json
{
  "id": "bigtom_001",
  "scenario": "Sally puts her marble in a basket...",
  "q1": "Where does Sally think the marble is? A) basket B) box",
  "q2": "Where does Sally want to look? A) basket B) box",
  "q3": "Where will Sally look for the marble?",
  "ground_truth": {
    "q1_belief": "A",
    "q2_desire": "A",
    "q3_action": "Sally will look in the basket"
  }
}
```

---

## Ablation 조건

| 조건 | Agent1 | Agent2 | Agent3 | Debate |
|------|--------|--------|--------|--------|
| full_system | ✅ | ✅ | ✅ | ✅ |
| no_debate | ✅ | ✅ | ✅ | ❌ |
| agent1_only | ✅ | ❌ | ❌ | ❌ |
| agent2_only | ❌ | ✅ | ❌ | ❌ |
| agent3_only | ❌ | ❌ | ✅ | ❌ |
| no_agent3 | ✅ | ✅ | ❌ | ✅ |

---

## 설계 결정 사항 (모호한 부분 판단 기준)

| 항목 | 결정 | 이유 |
|------|------|------|
| max_rounds | 3 | 비용/시간 vs 성능 균형. 논문 실험에서 통제 가능한 범위 |
| 동점 tiebreak | Agent 3 우선 | 고차원 추론 전담이므로 가장 신뢰도 높음 |
| Agent 동시 실행 | asyncio + run_in_executor | 순수 Python으로 병렬성 확보, LangGraph 미사용 |
| Q3 Action 평가 | 키워드 매칭 (50% 이상) | 개방형이라 완전 일치 불가. 추후 LLM-judge로 교체 가능 |
| 결과 저장 | jsonl append 방식 | 실험 중단 후 재시작해도 결과 누적 가능 |
| 감독관 판단 | 프롬프트 기반 | 설계 요구사항. 추후 알고리즘으로 교체 가능하도록 모듈화 |
