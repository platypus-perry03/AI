---
config:
  layout: elk
---
graph TD
    A["🧠 Brain Tumor MRI Dataset<br/>7,200장 · 4 클래스"] --> B["1단계 · 데이터 준비"]
    B --> C["EDA<br/>클래스 분포 · 이미지 크기<br/>픽셀 밝기"]
    C --> D["전처리<br/>Resize 224x224 · Normalize<br/>Augmentation"]
    D --> E["DataLoader<br/>Train 4480 · Val 1120<br/>Test 1600"]
    E --> F["2단계 · 모델 학습"]
    F --> G["EfficientNet-B0<br/>ImageNet pretrained"]
    G --> H["Fine-tuning<br/>FC 레이어 교체"]
    H --> I["모델 평가<br/>Accuracy · Confusion Matrix"]
    I --> J["3단계 · XAI 적용"]
    J --> K["Grad-CAM<br/>종양 위치 히트맵"]
    J --> L["SHAP DeepExplainer<br/>픽셀별 기여도"]
    K --> M["결과 저장<br/>원본 + 오버레이 이미지"]
    L --> M
    M --> N["4단계 · 대시보드"]
    N --> O["Streamlit<br/>MRI 업로드 → 예측 + 시각화"]
    O --> P["배포<br/>Hugging Face Spaces"]
    
    classDef stage fill:#eef2ff,stroke:#818cf8,color:#1e1b4b,font-weight:bold
    classDef process fill:#f0fdfa,stroke:#2dd4bf,color:#1e1b4b
    classDef model fill:#f5f3ff,stroke:#a78bfa,color:#1e1b4b
    classDef xai fill:#fff7ed,stroke:#fb923c,color:#1e1b4b
    classDef deploy fill:#f0fdf4,stroke:#4ade80,color:#1e1b4b
    
    class A,B,F,J,N stage
    class C,D,E process
    class G,H,I model
    class K,L,M xai
    class O,P deploy