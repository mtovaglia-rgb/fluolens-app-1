import math
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(page_title="FluoLens Compare", layout="wide")


@dataclass
class ZoneMetrics:
    mean_intensity: float
    median_intensity: float
    std_intensity: float
    active_area_ratio: float


# -----------------------------
# Utility functions
# -----------------------------
def load_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def resize_for_display(img: np.ndarray, max_side: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def preprocess_fluorescein(img_rgb: np.ndarray) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    # Typical green/yellow-green fluorescein range; adjustable later.
    lower = np.array([25, 40, 20])
    upper = np.array([110, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    signal = cv2.addWeighted(value, 0.7, saturation, 0.3, 0)
    signal = cv2.normalize(signal, None, 0, 255, cv2.NORM_MINMAX)

    masked_signal = cv2.bitwise_and(signal, signal, mask=mask)
    return masked_signal


def create_zone_masks(shape: Tuple[int, int], center: Tuple[int, int], radius: int) -> Dict[str, np.ndarray]:
    h, w = shape
    y_indices, x_indices = np.indices((h, w))
    dist = np.sqrt((x_indices - center[0]) ** 2 + (y_indices - center[1]) ** 2)

    masks = {
        "Centrale": dist <= radius * 0.25,
        "Paracentrale": (dist > radius * 0.25) & (dist <= radius * 0.50),
        "Medio-periferica": (dist > radius * 0.50) & (dist <= radius * 0.75),
        "Periferica": (dist > radius * 0.75) & (dist <= radius),
    }
    return masks


def compute_zone_metrics(signal: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, ZoneMetrics]:
    metrics = {}
    for zone_name, mask in masks.items():
        values = signal[mask]
        if values.size == 0:
            metrics[zone_name] = ZoneMetrics(0.0, 0.0, 0.0, 0.0)
            continue
        active = values[values > 0]
        metrics[zone_name] = ZoneMetrics(
            mean_intensity=float(values.mean()),
            median_intensity=float(np.median(values)),
            std_intensity=float(values.std()),
            active_area_ratio=float(active.size / values.size),
        )
    return metrics


def draw_overlay(img_rgb: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
    out = img_rgb.copy()
    colors = [(0, 255, 255), (0, 200, 255), (255, 180, 0), (255, 100, 0)]
    for i, frac in enumerate([0.25, 0.50, 0.75, 1.0]):
        cv2.circle(out, center, int(radius * frac), colors[i], 2)
    cv2.circle(out, center, 5, (255, 0, 0), -1)
    return out


def generate_interpretation(ref: Dict[str, ZoneMetrics], sam: Dict[str, ZoneMetrics], ref_clearance_um: float):
    ref_c = ref["Centrale"].mean_intensity
    sam_c = sam["Centrale"].mean_intensity

    if ref_c < 1:
        ratio = 1.0
    else:
        ratio = sam_c / ref_c

    delta_pct = (ratio - 1.0) * 100

    if delta_pct > 15:
        central_judgement = "probabilmente maggiore"
    elif delta_pct < -15:
        central_judgement = "probabilmente minore"
    else:
        central_judgement = "grossolanamente simile"

    estimated = max(0.0, ref_clearance_um * ratio)

    zone_notes = []
    ordered_zones = ["Paracentrale", "Medio-periferica", "Periferica"]
    for z in ordered_zones:
        ref_m = ref[z].mean_intensity
        sam_m = sam[z].mean_intensity
        if ref_m < 1:
            z_ratio = 1.0
        else:
            z_ratio = sam_m / ref_m
        z_pct = (z_ratio - 1.0) * 100
        if z_pct > 20:
            zone_notes.append(f"{z}: fluorescenza più marcata rispetto alla reference")
        elif z_pct < -20:
            zone_notes.append(f"{z}: fluorescenza ridotta rispetto alla reference")
        else:
            zone_notes.append(f"{z}: andamento simile alla reference")

    confidence = "media"
    if ref["Centrale"].active_area_ratio < 0.15 or sam["Centrale"].active_area_ratio < 0.15:
        confidence = "bassa"
    elif abs(delta_pct) < 10:
        confidence = "medio-alta"

    return {
        "central_judgement": central_judgement,
        "estimated_clearance": estimated,
        "delta_pct": delta_pct,
        "zone_notes": zone_notes,
        "confidence": confidence,
    }


def radial_profile(signal: np.ndarray, center: Tuple[int, int], radius: int, bins: int = 100):
    y_indices, x_indices = np.indices(signal.shape)
    dist = np.sqrt((x_indices - center[0]) ** 2 + (y_indices - center[1]) ** 2)
    mask = dist <= radius
    dist = dist[mask]
    values = signal[mask]

    if dist.size == 0:
        return np.array([]), np.array([])

    bin_edges = np.linspace(0, radius, bins + 1)
    bin_ids = np.digitize(dist, bin_edges) - 1

    profile = np.zeros(bins, dtype=float)
    counts = np.zeros(bins, dtype=float)

    for idx, val in zip(bin_ids, values):
        if 0 <= idx < bins:
            profile[idx] += val
            counts[idx] += 1

    counts[counts == 0] = 1
    profile /= counts
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, profile


# -----------------------------
# UI
# -----------------------------
st.title("FluoLens Compare")
st.caption("Confronto assistito tra immagine reference e immagine campione di fluoresceina")

with st.sidebar:
    st.header("Parametri")
    ref_clearance = st.number_input("Clearance reference (µm)", min_value=0.1, value=5.0, step=0.5)
    auto_resize = st.checkbox("Ridimensiona immagini per velocizzare", value=True)
    st.markdown(
        """
        **Nota**
        La stima in micron è solo approssimativa e dipende fortemente dalla standardizzazione dell'acquisizione.
        """
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Lente ideale / reference")
    ref_file = st.file_uploader("Carica immagine reference", type=["jpg", "jpeg", "png", "tif", "tiff"], key="ref")

with col2:
    st.subheader("Lente da valutare")
    sample_file = st.file_uploader("Carica immagine campione", type=["jpg", "jpeg", "png", "tif", "tiff"], key="sample")

if ref_file and sample_file:
    ref_img = load_image(ref_file)
    sample_img = load_image(sample_file)

    if auto_resize:
        ref_img = resize_for_display(ref_img)
        sample_img = resize_for_display(sample_img)

    st.markdown("### Definizione centro e raggio")
    control_ref, control_sam = st.columns(2)

    with control_ref:
        st.markdown("**Reference**")
        ref_h, ref_w = ref_img.shape[:2]
        ref_cx = st.slider("Centro X reference", 0, ref_w - 1, ref_w // 2)
        ref_cy = st.slider("Centro Y reference", 0, ref_h - 1, ref_h // 2)
        ref_radius = st.slider("Raggio reference", 10, min(ref_w, ref_h) // 2, min(ref_w, ref_h) // 3)

    with control_sam:
        st.markdown("**Campione**")
        sam_h, sam_w = sample_img.shape[:2]
        sam_cx = st.slider("Centro X campione", 0, sam_w - 1, sam_w // 2)
        sam_cy = st.slider("Centro Y campione", 0, sam_h - 1, sam_h // 2)
        sam_radius = st.slider("Raggio campione", 10, min(sam_w, sam_h) // 2, min(sam_w, sam_h) // 3)

    ref_center = (ref_cx, ref_cy)
    sam_center = (sam_cx, sam_cy)

    ref_signal = preprocess_fluorescein(ref_img)
    sam_signal = preprocess_fluorescein(sample_img)

    ref_masks = create_zone_masks(ref_signal.shape, ref_center, ref_radius)
    sam_masks = create_zone_masks(sam_signal.shape, sam_center, sam_radius)

    ref_metrics = compute_zone_metrics(ref_signal, ref_masks)
    sam_metrics = compute_zone_metrics(sam_signal, sam_masks)

    interpretation = generate_interpretation(ref_metrics, sam_metrics, ref_clearance)

    disp1, disp2 = st.columns(2)
    with disp1:
        st.image(draw_overlay(ref_img, ref_center, ref_radius), caption="Reference con overlay", use_container_width=True)
        st.image(ref_signal, caption="Segnale fluoresceina - reference", use_container_width=True, clamp=True)
    with disp2:
        st.image(draw_overlay(sample_img, sam_center, sam_radius), caption="Campione con overlay", use_container_width=True)
        st.image(sam_signal, caption="Segnale fluoresceina - campione", use_container_width=True, clamp=True)

    st.markdown("### Risultato sintetico")
    m1, m2, m3 = st.columns(3)
    m1.metric("Giudizio clearance centrale", interpretation["central_judgement"])
    m2.metric("Stima clearance campione", f"{interpretation['estimated_clearance']:.1f} µm")
    m3.metric("Delta vs reference", f"{interpretation['delta_pct']:+.1f}%")

    st.markdown(f"**Confidenza stimata:** {interpretation['confidence']}")

    st.markdown("### Considerazioni")
    st.write(f"- Clearance centrale: **{interpretation['central_judgement']}** rispetto alla reference impostata a **{ref_clearance:.1f} µm**.")
    for note in interpretation["zone_notes"]:
        st.write(f"- {note}")

    st.markdown("### Tabella metriche")
    table_rows = []
    for zone in ["Centrale", "Paracentrale", "Medio-periferica", "Periferica"]:
        table_rows.append(
            {
                "Zona": zone,
                "Mean Ref": round(ref_metrics[zone].mean_intensity, 2),
                "Mean Campione": round(sam_metrics[zone].mean_intensity, 2),
                "Median Ref": round(ref_metrics[zone].median_intensity, 2),
                "Median Campione": round(sam_metrics[zone].median_intensity, 2),
                "Area attiva Ref": round(ref_metrics[zone].active_area_ratio, 3),
                "Area attiva Campione": round(sam_metrics[zone].active_area_ratio, 3),
            }
        )
    st.dataframe(table_rows, use_container_width=True)

    x_ref, y_ref = radial_profile(ref_signal, ref_center, ref_radius)
    x_sam, y_sam = radial_profile(sam_signal, sam_center, sam_radius)

    if x_ref.size and x_sam.size:
        st.markdown("### Profilo radiale")
        chart_data = {
            "distanza": x_ref,
            "reference": y_ref,
            "campione": y_sam if len(y_sam) == len(y_ref) else np.interp(x_ref, x_sam, y_sam),
        }
        st.line_chart(chart_data, x="distanza")

    with st.expander("Avvertenze clinico-tecniche"):
        st.markdown(
            """
            - Il software fornisce un **confronto assistito** e non una misurazione strumentale assoluta.
            - Le stime in micron sono solo orientative.
            - Illuminazione, esposizione, filtro, angolo di ripresa, film lacrimale e tempo dall'instillazione influenzano il risultato.
            - Per risultati più robusti è consigliata una procedura di acquisizione standardizzata.
            """
        )
else:
    st.info("Carica entrambe le immagini per iniziare il confronto.")
