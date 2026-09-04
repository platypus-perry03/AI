# Reasoning × Memorization Representation Decomposition

> **Mechanistic analysis of Reasoning and Memorization in Transformer-based language models**

LiReF에서 관찰된 Reasoning과 Memorization의 내부 표현 차이를 바탕으로,
여러 Base 모델의 어떤 Attention Head와 FFN Neuron이 이 차이에 기여하는지
분석한 연구이다.

## Research Overview · 연구 소개

대규모 언어 모델은 주어진 정보를 조합해 답을 **추론(Reasoning)**하기도
하고, 학습 중 습득한 사실을 **회상(Memorization)**하기도 한다. 이 연구는
두 과정이 Transformer 내부에서 어떻게 다르게 표현되는지 살펴보고, 그
차이에 기여하는 Layer, Attention Head와 FFN Neuron을 찾는다.

분석은 내부 반응을 관찰하는 데서 끝나지 않는다. 후보 부품을 직접
억제한 뒤 R/M 표현 차이가 줄어드는지, 같은 기능적 패턴이 다른 Base
모델에서도 반복되는지, 실제 정답 선택에도 영향을 주는지를 단계적으로
검증한다.

## Research Background · 연구 배경

모델이 같은 정답을 출력하더라도, 주어진 정보로 답을 계산했는지 학습된
지식을 회상했는지는 출력만으로 알기 어렵다. LiReF는 Reasoning과
Memorization 문항의 hidden representation이 하나의 선형 방향에서
구분되는 현상을 보여주었다.

그러나 표현 차이가 보인다는 사실만으로는 다음을 알 수 없다.

- 모델 내부의 어떤 Head와 Neuron이 그 차이에 기여하는가?
- 각 부품은 R 방향과 M 방향 중 어디에 기여하는가?
- 같은 기능의 부품이 서로 다른 Base 모델에서도 나타나는가?
- 내부 표현 차이가 실제 정답 선택에 필요한가?

이 연구는 LiReF의 관찰 결과를 component 수준으로 분해하고, 억제 실험을
통해 후보의 기능적 기여를 검증하기 위해 시작되었다.

## Research Question / Goal · 연구 질문 및 목표

### 핵심 연구 질문

> **LiReF에서 관찰된 R/M 표현 차이는 Transformer 내부의 어떤 Head와
> Neuron의 기여로 형성되며, 이 기능은 여러 Base 모델과 실제 정답 선택
> 행동에서도 확인되는가?**

이를 다음 네 질문으로 나누어 분석한다.

1. **R/M Gap:** 어떤 부품이 R과 M의 내부 표현 간격을 유지하는가?
2. **R/M Direction:** 어떤 부품이 R 문항을 R 방향으로, M 문항을 M
   방향으로 미는가?
3. **Cross-model recurrence:** 다른 Base 모델에도 기능적으로 비슷한
   Head와 Neuron이 존재하는가?
4. **Behavioral necessity:** 해당 부품을 억제하면 실제 정답 선택 성능도
   안정적으로 감소하는가?

최종 목표는 특정 부품을 곧바로 “Reasoning neuron” 또는 “Memorization
neuron”이라고 명명하는 것이 아니다. R/M 표현 차이가 만들어지는 내부
경로를 검증 가능한 component와 circuit 수준에서 설명하고, 그 차이가
모델 행동에 필요한지 밝히는 것이 목표다.

## Research Approach · 연구 방법

### 1. 데이터와 모델

- **주 분석 데이터:** `MMLU-Pro` 3,000문항
- **R/M 구분:** 기존 `memory_reason_score`에서 `> 0.5`를 R로 사용
- **분할:** 후보 탐색 2,400문항 / 후보 선정에 사용하지 않은 heldout
  600문항
- **Base 모델:** Meta-Llama-3-8B, Mistral-7B-v0.3,
  OLMo-2-1124-7B, Gemma-2-9B
- **측정 위치:** 각 문항의 마지막 prompt token

### 2. Component 탐색과 특성 분석

```text
R/M hidden representation
        ↓
Layer별 차이 측정
        ↓
Attention / FFN 분해
        ↓
개별 Head / Neuron 후보 탐색
        ↓
Relation response / Task relevance 분석
```

각 부품의 출력이 모델별로 동결된 LiReF R/M 방향에 얼마나 기여하는지
계산하고, Discovery 데이터에서 후보를 선정한 뒤 heldout 데이터에서 같은
패턴이 반복되는지 확인한다.

### 3. Gap과 Direction의 정의

LiReF 축은 다음처럼 해석한다.

```text
M 방향 (−)  ←──────── 0 ────────→  (+) R 방향
```

- **R/M Gap component:** 해당 부품을 억제했을 때 R과 M의 평균 표현
  간격이 줄어드는 부품
- **R-direction component:** R 문항에서 평균 기여가 양수이고, M
  문항보다 더 R 방향으로 기여하는 부품
- **M-direction component:** M 문항에서 평균 기여가 음수이고, R
  문항보다 더 M 방향으로 기여하는 부품

Direction 분석은 Gap 후보를 단순 분류한 것이 아니라, 전체 Layer의
Head와 Neuron을 방향별 기준으로 각각 다시 탐색한 별도 실험이다.

### 4. Causal suppression

후보가 단순히 R/M과 함께 반응하는지, 실제로 표현 차이에 기여하는지를
확인하기 위해 다음 개입을 수행했다.

- 후보 Head 출력은 50% 또는 100% 감소
- 후보 Neuron 값은 Discovery 평균 쪽으로 50% 또는 100% 이동
- 같은 Layer·같은 종류에서 반응 규모가 비슷한 matched control 1개와
  random control 3개를 동일하게 억제
- 억제 강도에 따라 효과가 커지는지, control보다 효과가 큰지,
  bootstrap·permutation·FDR 기준을 통과하는지 확인

### 5. Behavioral Validation

내부 표현 변화가 정답 선택 변화로 이어지는지 확인하기 위해
Meta-Llama의 R/M-direction 고유 후보 13개를 외부 문항에서 50%·100%
억제하고 정답률과 정답 확률을 측정했다.

- **M task-family:** C-Eval-H
- **R task-family:** GSM8K, GSM-Symbolic, MGSM-en
- 생성형 수학 문항은 동결된 A–D 객관식 형식으로 변환

이 R/M 구분은 문항별 `memory_reason_score`가 아니라 LiReF에서 사용한
task-family 기준이라는 제한이 있다.

## Current Progress · 현재 진행 상황

### 1. Component-level 결과

| Base 모델 | R/M Gap | R-direction | M-direction |
|---|---:|---:|---:|
| Meta-Llama-3-8B | 9 (Head 4 + Neuron 5) | 10 (Head 5 + Neuron 5) | 7 (Head 4 + Neuron 3) |
| Mistral-7B-v0.3 | 10 (Head 5 + Neuron 5) | 9 (Head 5 + Neuron 4) | 9 (Head 5 + Neuron 4) |
| OLMo-2-1124-7B | 10 (Head 5 + Neuron 5) | 10 (Head 5 + Neuron 5) | 7 (Head 3 + Neuron 4) |
| Gemma-2-9B | 0 | 0 | 2 (Head 2) |

Meta-Llama의 Gap 분석에서는 Stage A 후보 20개를 모두 같은 heldout,
dose, control, 통계 기준으로 검증했고 9개가 strict PASS했다. Cross-model
및 Direction 분석은 각 모델에서 전체 Layer를 검색하고 유형별 최대 5개를
후보로 지명했다.

따라서 위 숫자는 확인된 후보 수를 뜻하지만, 모델에 존재하는 전체
R/M component 수나 모델 간 component 양의 직접 비교를 의미하지 않는다.
모델마다 같은 번호의 부품을 찾은 것도 아니며, 선택성·방향·억제 효과가
비슷한 **기능적 후보**를 찾은 결과다.

### 2. Behavioral Validation 결과

- Meta-Llama R/M-direction 후보의 중복을 제거한 고유 후보: **13개**
- 안정적인 정답률 감소까지 보인 strict behavioral signal: **0개**
- 정답 확률만 소폭 변한 Head: `L29H31`, `L30H6`, `L31H3`
- confirmation에서 MGSM baseline gate가 실패하여 추가 개입과 나머지 세
  모델 실행은 중단

즉 특정 Head와 Neuron이 R/M 내부 표현 차이에 기여한다는 결과는
확인했지만, 이 부품들이 외부 객관식 정답 선택에 필수적이라는 증거는
확인하지 못했다.

### 3. 탐색적 입력 특징 분석

기존 MMLU-Pro 문항에서 `transformation_required`가 R label과 Layer 31
LiReF 반응에 연관되는 현상을 발견했다. 하지만 Transformation 문항이
대부분 R 문항이어서 두 효과를 분리하기 어렵고, 현재 결과는 인과적
특징이 아닌 **탐색적 association**으로만 해석한다. 독립 재현용 synthetic
task v4·v5는 behavioral calibration을 통과하지 못해 Layer 31 독립 재현과
intervention을 실행하지 않았다.

### 현재 결론

> **여러 Base 모델에서 특정 Head와 Neuron이 R/M 표현 차이와 각 방향에
> 기여하는 현상은 확인했지만, 보편적인 단일 R/M mechanism이나 실제
> 정답 선택에 필수적인 component는 확인하지 못했다.**

## Future Work · 향후 연구 방향

1. **정식 Behavioral Validation**

   모델이 안정적으로 풀 수 있는 독립 문항을 확보하고, component 억제가
   정답률과 정답 확률에 미치는 영향을 다시 검증한다.

2. **Circuit Analysis**

   개별 후보를 넘어 Head–Neuron 사이의 연결, 정보 전달 순서와 joint
   suppression 효과를 분석한다.

3. **Input Feature Discovery**

   R/M label과 계산·변환 요구가 겹치지 않는 데이터에서 어떤 입력 특징이
   component를 방향별로 작동시키는지 확인한다.

4. **Cross-dataset / Cross-model Generalization**

   동일한 후보 선정·검증 규칙을 적용해 데이터셋, random seed와 모델
   구조가 달라져도 기능적 패턴이 유지되는지 검증한다.

5. **Behavior Control**

   충분한 인과·행동 근거가 확보된 뒤에만 R/M 행동을 선택적으로 조절할
   수 있는 intervention 가능성을 탐색한다.

## Repository Structure · 파일 구조

```text
reenact/
├── README.md
├── liref/
│   ├── README.md
│   ├── REPRODUCTION_KR.md
│   ├── STUDY.md
│   └── reasoning_representation/
├── experiments/
│   ├── rm_decomp/                  # Stage A: Layer/Attention/FFN 탐색
│   ├── rm_decomp_b/                # Stage B: Head/Neuron 특성 분석
│   ├── rm_decomp_b_extension/      # 관계·task relevance 및 control 확장
│   ├── rm_decomp_causal/           # Stage C: 후보 suppression 검증
│   ├── rm_decomp_cross_model/      # Gap/R/M-direction cross-model 분석
│   ├── rm_decomp_behavioral_validation/
│   │                               # 외부 객관식 행동 검증
│   ├── rm_decomp_feature_causal/   # 입력 특징·patching 탐색
│   ├── rm_decomp_pre_stage_c/      # Stage C 이전 진단
│   └── rm_decomp_stage_e/          # Calibration 및 feature 후속 연구
├── scripts/                        # LiReF·robustness 분석 유틸리티
├── pdf/
│   ├── ReMem.pptx                  # 발표 자료
│   └── update_remem_pptx.py        # 발표 자료 생성/수정 코드
├── run_liref_hidden_states.sh
└── run_mgsm_language_robustness.sh
```

주요 발표 자료는 [`pdf/ReMem.pptx`](pdf/ReMem.pptx), Stage E의 현재 해석
기준은
[`experiments/rm_decomp_stage_e/STAGE_E_CURRENT_STATUS_KO.md`](experiments/rm_decomp_stage_e/STAGE_E_CURRENT_STATUS_KO.md)에서
확인할 수 있다.
