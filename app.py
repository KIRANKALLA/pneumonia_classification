import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch

st.set_page_config(page_title="Pneumonia Classifier", layout="centered")

@st.cache_resource
def load_model():
    # Path to your copied Kaggle weight
    model = YOLO("pneumonia_yolo11.pt")  # best.pt renamed
    # Force CPU if you’re on a CPU-only host
    model.to("cpu")
    return model

model = load_model()

st.title("Pneumonia Classification (YOLOv11-cls)")
st.write("Upload a chest X-ray (JPEG/PNG). The model predicts **NORMAL** vs **PNEUMONIA**.")

img_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
imgsz = st.slider("Image size", 224, 384, 320, step=32)
use_tta = st.checkbox("Test-Time Augmentation (TTA)", value=False)

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Input", use_container_width=True)

    with st.spinner("Running inference..."):
        results = model.predict(
            source=image,
            imgsz=imgsz,
            device="cpu",
            conf=0.0    # classification uses probs; conf not used for filtering
        
        )

    r = results[0]
    probs = r.probs  # ultralytics.engine.results.Probs
    names = r.names  # dict: {0: 'NORMAL', 1: 'PNEUMONIA'} (or vice versa)

    # Top-1
    top1_idx = int(probs.top1)
    top1_name = names[top1_idx]
    top1_score = float(probs.top1conf)

    # Full distribution
    dist = probs.data.cpu().numpy().flatten()
    st.subheader("Prediction")
    st.write(f"**{top1_name}**  (confidence: {top1_score:.4f})")

    # show both class probs
    st.write({names[i]: float(dist[i]) for i in range(len(dist))})
