import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="wide")
st.title("FluoLens Compare v3 - Analisi Clinica")

# ---------- FUNZIONI ----------

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def detect_lens(img):
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=50,
        param2=30,
        minRadius=50,
        maxRadius=400
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        return int(x), int(y), int(r)
    else:
        h, w = img.shape
        return w // 2, h // 2, min(w, h) // 3

def get_fluo_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([35, 50, 50])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def centroid(mask):
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy

def clearance_level(value, ref=5.0):
    if value < ref * 0.7:
        return "basso"
    elif value > ref * 1.3:
        return "alto"
    else:
        return "in target"

def decentration_direction(cx, cy, gx, gy):
    dx = gx - cx
    dy = gy - cy

    if abs(dx) < 10 and abs(dy) < 10:
        return "centrata"

    if abs(dx) > abs(dy):
        return "nasale" if dx < 0 else "temporale"
    else:
        return "superiore" if dy < 0 else "inferiore"

def clinical_interpretation(clearance, decentration):
    if clearance == "basso" and decentration == "superiore":
        return "Lente piatta (sagittale insufficiente)"
    if clearance == "alto" and decentration == "inferiore":
        return "Lente profonda (sagittale eccessiva)"
    if clearance == "alto":
        return "Clearance elevato – lente probabilmente profonda"
    if clearance == "basso":
        return "Clearance ridotto – lente probabilmente piatta"
    return "Fit nella norma"

def heatmap_diff(ref, sample):
    # Uniforma le dimensioni prima del confronto
    if ref.shape[:2] != sample.shape[:2]:
        sample = cv2.resize(sample, (ref.shape[1], ref.shape[0]))

    diff = cv2.absdiff(ref, sample)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return diff_norm

# ---------- UI ----------

col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader("Carica immagine IDEALE", type=["jpg", "png", "jpeg"])

with col2:
    sample_file = st.file_uploader("Carica immagine CAMPIONE", type=["jpg", "png", "jpeg"])

if ref_file and sample_file:
    ref_img = load_image(ref_file)
    sample_img = load_image(sample_file)

    ref_gray = preprocess(ref_img)
    sample_gray = preprocess(sample_img)

    rx, ry, rr = detect_lens(ref_gray)
    sx, sy, sr = detect_lens(sample_gray)

    ref_mask = get_fluo_mask(ref_img)
    sample_mask = get_fluo_mask(sample_img)

    ref_centroid = centroid(ref_mask)
    sample_centroid = centroid(sample_mask)

    tabs = st.tabs(["Analisi", "Heatmap", "Maschere"])

    with tabs[0]:
        st.subheader("Analisi Clinica")

        if sample_centroid:
            cx, cy = sample_centroid
        else:
            cx, cy = sx, sy

        direction = decentration_direction(sx, sy, cx, cy)

        # stima semplice del clearance basata sulla quota di fluoresceina rilevata
        clearance_value = float(np.mean(sample_mask) / 25.5)  # scala più ragionevole 0-10 circa
        clearance = clearance_level(clearance_value)

        st.write(f"Clearance: **{clearance}**")
        st.write(f"Decentramento: **{direction}**")
        st.write(f"Valore stimato clearance: **{clearance_value:.2f}**")

        interpretation = clinical_interpretation(clearance, direction)
        st.success(f"Interpretazione: {interpretation}")

    with tabs[1]:
        st.subheader("Heatmap differenze")
        diff = heatmap_diff(ref_gray, sample_gray)
        st.image(diff, caption="Differenze tra reference e campione", use_container_width=True)

    with tabs[2]:
        st.subheader("Maschere fluoresceina")
        c1, c2 = st.columns(2)
        with c1:
            st.image(ref_mask, caption="Reference mask", use_container_width=True)
        with c2:
            st.image(sample_mask, caption="Sample mask", use_container_width=True)
