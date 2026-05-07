# 🧠 BrainScan XAI

뇌종양 MRI 이미지를 분류하고, XAI 기법으로 모델의 판단 근거를 시각화하는 프로젝트

---

## 📌 프로젝트 소개

EfficientNet-B3 모델로 뇌종양 MRI를 4가지 클래스로 분류하고,
Grad-CAM을 통해 모델이 어느 영역을 보고 판단했는지 히트맵으로 시각화합니다.
Streamlit 대시보드로 누구나 MRI 이미지를 업로드해 결과를 확인할 수 있습니다.

---

## 🏷️ 분류 클래스

| 클래스 | 설명 |
|---|---|
| glioma | 신경교종 |
| meningioma | 수막종 |
| notumor | 정상 |
| pituitary | 뇌하수체 종양 |

---

## 🛠️ 기술 스택

- **모델:** EfficientNet-B3 (ImageNet pretrained, Fine-tuning)
- **XAI:** Grad-CAM
- **대시보드:** Streamlit
- **학습 환경:** Google Colab (T4 GPU)
- **언어:** Python 3.12

---

## 📁 프로젝트 구조

```
BrainScan_XAI/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_train.ipynb
│   └── 04_xai.ipynb
├── app/
│   └── dashboard.py
├── models/          # best_model.pth (별도 다운로드)
├── docs/
│   └── pipeline.md
└── README.md
```

---

## 📊 데이터셋

- **출처:** [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **구성:** 총 7,200장 / 4 클래스
- **분할:** Train 4,493 / Val 1,123 / Test 1,600

---

## 📈 학습 결과

| 항목 | 값 |
|---|---|
| 모델 | EfficientNet-B3 |
| Optimizer | Adam (lr=0.0005) |
| Epochs | 20 |
| Val Accuracy | 99.29% |
| Test Accuracy | 94.81% |

---

## 🔍 XAI 시각화 결과

Grad-CAM을 통해 모델이 종양 위치에 집중하여 판단함을 확인

| 클래스 | 신뢰도 | 설명 |
|---|---|---|
| meningioma | 100.00% | 종양 위치 정확히 집중 |
| notumor | 99.99% | 뇌 전체 고르게 분산 |
| pituitary | 99.79% | 뇌 하단 중앙 집중 |
| glioma | - | meningioma와 혼동 (주요 오류 원인) |

---

## 🚀 실행 방법

```bash
# 1. 가상환경 활성화
venv\Scripts\activate

# 2. 패키지 설치
pip install torch torchvision grad-cam streamlit pillow matplotlib numpy

# 3. 모델 파일 다운로드 후 models/ 폴더에 저장

# 4. 대시보드 실행
streamlit run app/dashboard.py
```

---

## 🗺️ 파이프라인

자세한 파이프라인은 [docs/pipeline.md](docs/pipeline.md) 참고