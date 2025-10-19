import os, json, time
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
import cv2
import onnxruntime as ort

# ---------- Paths ----------
PROJECT_ROOT = Path(r"C:\Users\Saeid_Eshghi\Project_7")
OUT_ROOT = PROJECT_ROOT / "out"
EXPORT = OUT_ROOT / "exports"
CLASSES = json.load(open(OUT_ROOT / "classes.json","r",encoding="utf-8"))
IMG_SIZE = 224
ONNX_PATH = EXPORT / "model.onnx"

# ---------- ONNX runtime session ----------
@st.cache_resource
def load_onnx_session():
    so = ort.SessionOptions()
    # می‌تونی لاگ را کم/زیاد کنی: so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(ONNX_PATH), sess_options=so, providers=providers)
    # نام‌های ورودی/خروجی (برای مدل‌های Keras معمولاً تک ورودی/خروجی)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name

def preprocess_pil(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype("float32")/255.0  # [0,1]
    x = np.expand_dims(x, 0)                   # (1,224,224,3)
    return img, x

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def occlusion_heatmap_onnx(sess, in_name, out_name, img_float01, patch=15, stride=8, fill=0.0):
    H,W,_ = img_float01.shape
    base = sess.run([out_name], {in_name: img_float01[None,...].astype("float32")})[0][0]
    # اگر خروجی logits بود، softmax:
    if base.ndim == 1:
        base = softmax(base[None,...])[0]
    target_idx = int(np.argmax(base)); base_conf = float(base[target_idx])

    heat = np.zeros((H,W), np.float32); counts = np.zeros((H,W), np.float32)
    for y in range(0, H-patch+1, stride):
        for x in range(0, W-patch+1, stride):
            im2 = img_float01.copy()
            im2[y:y+patch, x:x+patch] = fill
            prob = sess.run([out_name], {in_name: im2[None,...].astype("float32")})[0][0]
            if prob.ndim == 1:
                prob = softmax(prob[None,...])[0]
            drop = base_conf - float(prob[target_idx])
            heat[y:y+patch, x:x+patch] += drop
            counts[y:y+patch, x:x+patch] += 1
    heat /= (counts+1e-8)
    heat = np.maximum(heat, 0); heat /= (heat.max()+1e-8)
    return heat, target_idx, base_conf

def save_frame(img_pil, suffix):
    demo_dir = OUT_ROOT / "demo_frames"; demo_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    p = demo_dir / f"frame_{ts}_{suffix}.png"
    img_pil.save(p); return p

# ---------- UI ----------
st.set_page_config(page_title="♻️ Trash Classifier (ONNX)", layout="wide")
st.title("♻️ Trash Classification — ONNXRuntime")
st.caption("Classes: " + ", ".join(CLASSES))
sess, in_name, out_name = load_onnx_session()

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Input source", ["Upload image", "Camera"], index=0)
    topk = st.slider("Top-K bars", 3, min(5, len(CLASSES)), 5)
    explain = st.checkbox("Enable Explain (Occlusion)", value=True)
    st.caption(f"ONNX file: {ONNX_PATH}")

col1, col2 = st.columns([1,1])

# Input
input_image = None
if mode == "Upload image":
    f = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])
    if f: input_image = Image.open(f)
else:
    cam = st.camera_input("Take a photo")
    if cam: input_image = Image.open(cam)

if input_image is None:
    st.info("یک تصویر آپلود کن یا با Camera عکس بگیر 🙂")
    st.stop()

# Predict
pil_resized, xb = preprocess_pil(input_image)
out = sess.run([out_name], {in_name: xb.astype("float32")})[0]  # (1,6) یا (1,logits)
out = out[0]
if out.ndim == 1:
    prob = softmax(out[None,...])[0]
else:
    prob = out

pred_idx = int(np.argmax(prob)); pred = CLASSES[pred_idx]; conf = float(prob[pred_idx])

with col1:
    st.image(pil_resized, caption=f"Pred: {pred}  ({conf*100:.1f}%)", use_column_width=True)
    if st.button("Save frame"):
        p = save_frame(pil_resized, f"{pred}_{int(conf*100)}")
        st.success(f"Saved: {p}")

with col2:
    srt = np.argsort(prob)[::-1][:topk]
    df = pd.DataFrame({"class":[CLASSES[i] for i in srt], "prob":[float(prob[i]) for i in srt]})
    st.subheader("Top-K probabilities")
    st.bar_chart(df.set_index("class"))

if explain:
    st.subheader("Occlusion Heatmap (Explain)")
    x = np.array(pil_resized).astype("float32")/255.0
    with st.spinner("Computing heatmap..."):
        heat, idx, base_conf = occlusion_heatmap_onnx(sess, in_name, out_name, x, patch=15, stride=8)
    hm_u8 = (heat*255).astype("uint8")
    hm_color = cv2.applyColorMap(cv2.resize(hm_u8, (IMG_SIZE, IMG_SIZE)), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(pil_resized).astype("uint8"), 0.7, hm_color, 0.3, 0)
    c1, c2 = st.columns(2)
    with c1: st.image(heat, caption=f"Heatmap — Pred: {CLASSES[idx]} ({base_conf*100:.1f}%)", use_column_width=True, clamp=True)
    with c2: st.image(overlay, caption="Overlay", use_column_width=True)
