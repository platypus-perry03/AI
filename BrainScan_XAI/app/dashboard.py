# MRI 이미지를 업로드하면 예측 결과와 XAI 시각화를 보여주는 대시보드

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from torchvision.models.efficientnet import MBConv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

# ── 설정 ────────────────────────────────────────────────────
CLASSES    = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_PATH = "models/best_model.pth"
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 모델 로드 ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    def patched_forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result = result + input
        return result
    MBConv.forward = patched_forward

    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 4)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    for module in model.modules():
        if isinstance(module, nn.SiLU):
            module.inplace = False

    model.to(device).eval()
    return model

# ── 전처리 ───────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess(image: Image.Image):
    return transform(image).unsqueeze(0).to(device)

# ── Grad-CAM ─────────────────────────────────────────────────
def get_gradcam(model, input_tensor, pred_idx, raw_img):
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(pred_idx)])[0]
    raw_resized = np.array(raw_img.resize((224, 224))).astype(np.float32) / 255.0
    return show_cam_on_image(raw_resized, grayscale_cam, use_rgb=True)

# ── UI ───────────────────────────────────────────────────────
st.title("🧠 BrainScan XAI")
st.markdown("MRI 이미지를 업로드하면 종양 유형을 분류하고 판단 근거를 시각화합니다.")

uploaded = st.file_uploader("MRI 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded:
    model = load_model()
    image = Image.open(uploaded).convert("RGB")
    input_tensor = preprocess(image)

    # ── 예측 ──
    with torch.no_grad():
        output = model(input_tensor)
        probs    = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_cls = CLASSES[pred_idx]
        confidence = probs[pred_idx].item()

    # ── 결과 표시 ──
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="업로드된 MRI", use_container_width=True)
    with col2:
        st.metric("예측 클래스", pred_cls)
        st.metric("신뢰도", f"{confidence:.2%}")
        st.bar_chart({cls: probs[i].item() for i, cls in enumerate(CLASSES)})

    st.divider()

    # ── Grad-CAM ──
    st.subheader("🔥 Grad-CAM — 모델이 집중한 영역")
    cam_img = get_gradcam(model, input_tensor, pred_idx, image)
    st.image(cam_img, caption="Grad-CAM 히트맵 오버레이", use_container_width=True)