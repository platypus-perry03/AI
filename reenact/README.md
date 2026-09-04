# R/M Representation Decomposition

LiReF에서 관찰된 Reasoning/Memory 표현 차이를 Head와 FFN Neuron 수준으로
분해하고, 방향성·cross-model 재현·suppression 효과를 검증한 실험 코드와
연구 기록이다.

## 구성

- `liref/`: LiReF 재현 코드와 연구 문서
- `experiments/`: R/M component 탐색, causal validation, cross-model 및
  behavioral validation 코드와 동결된 설계 기록
- `scripts/`: hidden-state 및 robustness 분석 유틸리티
- `pdf/ReMem.pptx`: 발표 자료

## Git에서 제외되는 로컬 파일

다음 항목은 크기, 재생성 가능성 또는 민감정보 위험 때문에 저장소에
포함하지 않는다.

- `models/`: 로컬 모델 가중치
- `liref_outputs/`: hidden states, checkpoints, raw experiment outputs
- `liref/dataset/`: 로컬 데이터셋
- Python cache, 실행 로그, LaTeX 중간 산출물
- 참고 논문 및 중간 생성 PDF (`pdf/result.pdf`는 존재할 경우 예외)

모델과 데이터의 실제 로컬 경로는 각 frozen config와 실행 스크립트에서
확인할 수 있다.
