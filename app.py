import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("FluoLens Compare v2 - Analisi avanzata e fitting clinico")

# -------------------------
# Utility base
# -------------------------

def load_image(file):
    return np.array(Image.open(file).convert("RGB"))

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def detect_lens(gray):
    """
    Rilevamento semplice della lente.
    Prova HoughCircles, altrimenti fallback al centro immagine.
    """
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(50, gray.shape[0] // 4),
        param1=80,
        param2=30,
        minRadius=max(30, min(gray.shape) // 8),
        maxRadius=max(60, min(gray.shape) // 2),
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        x, y, r = circles[0]
        return int(x), int(y), int(r)

    h, w = gray.shape
    return w // 2, h // 2, min(w, h) // 3

def crop_with_padding(img, center, radius, pad_factor=1.2):
    x, y = center
    r = int(radius * pad_factor)

    h, w = img.shape[:2]
    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(w, x + r)
    y2 = min(h, y + r)

    return img[y1:y2, x1:x2]

def resize_to_match_radius(ref_img, sample_img, ref_radius, sample_radius):
    if sample_radius <= 0:
        return sample_img

    scale = ref_radius / sample_radius
    new_w = max(1, int(sample_img.shape[1] * scale))
    new_h = max(1, int(sample_img.shape[0] * scale))
    return cv2.resize(sample_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def center_crop_or_pad(img, target_h, target_w):
    """
    Porta l'immagine a dimensione target centrando il contenuto.
    """
    h, w = img.shape[:2]
    out = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)

    y_offset = max(0, (target_h - h) // 2)
    x_offset = max(0, (target_w - w) // 2)

    y1_src = max(0, (h - target_h) // 2)
    x1_src = max(0, (w - target_w) // 2)

    y2_src = min(h, y1_src + target_h)
    x2_src = min(w, x1_src + target_w)

    crop = img[y1_src:y2_src, x1_src:x2_src]

    ch, cw = crop.shape[:2]
    out[y_offset:y_offset + ch, x_offset:x_offset + cw] = crop
    return out

def normalize_green_channel(ref_img, sample_img):
    """
    Normalizza il canale verde del campione rispetto alla reference.
    """
    ref = ref_img.copy().astype(np.float32)
    sam = sample_img.copy().astype(np.float32)

    ref_g = ref[:, :, 1]
    sam_g = sam[:, :, 1]

    ref_mean = np.mean(ref_g)
    sam_mean = np.mean(sam_g)

    if sam_mean < 1e-6:
        return sample_img

    factor = ref_mean / sam_mean
    sam[:, :, 1] = np.clip(sam_g * factor, 0, 255)

    return sam.astype(np.uint8)

def get_green_mask(img, threshold=25):
    green = img[:, :, 1].astype(np.int16)
    red = img[:, :, 0].astype(np.int16)
    blue = img[:, :, 2].astype(np.int16)

    score = green - ((red + blue) / 2.0)
    mask = (score > threshold).astype(np.uint8) * 255
    return mask

def centroid_from_mask(mask):
    m = cv2.moments(mask)
    if m["m00"] == 0:
        return None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy

# -------------------------
# Analisi profili
# -------------------------

def radial_profile(img_channel, center):
    """
    Media radiale dell'intensità attorno al centro.
    """
    h, w = img_channel.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(np.int32)

    tbin = np.bincount(r.ravel(), img_channel.ravel().astype(np.float64))
    nr = np.bincount(r.ravel())
    profile = tbin / np.maximum(nr, 1)
    return profile

def smooth_profile(profile, ksize=9):
    if len(profile) < 3:
        return profile
    ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
    kernel = np.ones(ksize, dtype=np.float64) / ksize
    return np.convolve(profile, kernel, mode="same")

def compute_clearance_profile(ref_profile, sam_profile, ref_clearance_um=5.0):
    n = min(len(ref_profile), len(sam_profile))
    ref_p = ref_profile[:n]
    sam_p = sam_profile[:n]

    ratio = sam_p / np.maximum(ref_p, 1e-6)
    clearance_est = ratio * ref_clearance_um
    return clearance_est

def zone_means(profile, radius_px):
    """
    Divide il profilo in 4 zone concentriche.
    """
    r1 = max(1, int(radius_px * 0.25))
    r2 = max(r1 + 1, int(radius_px * 0.50))
    r3 = max(r2 + 1, int(radius_px * 0.75))
    r4 = max(r3 + 1, int(radius_px * 1.00))

    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    return {
        "centrale": safe_mean(profile[:r1]),
        "paracentrale": safe_mean(profile[r1:r2]),
        "medio_periferica": safe_mean(profile[r2:r3]),
        "periferica": safe_mean(profile[r3:r4]),
    }

# -------------------------
# Heatmap e overlay
# -------------------------

def heatmap_diff(ref_gray, sample_gray):
    if ref_gray.shape != sample_gray.shape:
        sample_gray = cv2.resize(sample_gray, (ref_gray.shape[1], ref_gray.shape[0]))
    diff = cv2.absdiff(ref_gray, sample_gray)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

def draw_overlay(img, lens_center, lens_radius, fluo_centroid=None):
    out = img.copy()

    cv2.circle(out, lens_center, lens_radius, (255, 255, 255), 2)
    cv2.circle(out, lens_center, 4, (255, 0, 0), -1)

    if fluo_centroid is not None:
        cv2.circle(out, fluo_centroid, 4, (255, 255, 0), -1)
        cv2.line(out, lens_center, fluo_centroid, (255, 255, 0), 2)

    return out

# -------------------------
# Fitting clinico
# -------------------------

def classify_clearance(center_clearance, target=5.0):
    if center_clearance < target * 0.5:
        return "molto basso"
    if center_clearance < target * 0.8:
        return "basso"
    if center_clearance <= target * 1.2:
        return "in target"
    if center_clearance <= target * 1.5:
        return "alto"
    return "molto alto"

def decentration_direction(lens_center, fluo_center, tolerance_px=10):
    if fluo_center is None:
        return "non determinabile", 0.0, 0.0, 0.0

    dx = fluo_center[0] - lens_center[0]
    dy = fluo_center[1] - lens_center[1]
    dist = float(np.sqrt(dx**2 + dy**2))

    if abs(dx) < tolerance_px and abs(dy) < tolerance_px:
        return "centrata", dx, dy, dist

    if abs(dx) > abs(dy):
        return ("temporale" if dx > 0 else "nasale"), dx, dy, dist
    return ("inferiore" if dy > 0 else "superiore"), dx, dy, dist

def decentration_amount(dist, radius):
    if radius <= 0:
        return "non determinabile"
    frac = dist / radius
    if frac < 0.05:
        return "minimo"
    if frac < 0.12:
        return "lieve"
    if frac < 0.22:
        return "moderato"
    return "marcato"

def clinical_interpretation(clearance_class, dec_dir, zone_sample, zone_ref):
    notes = []
    suggestions = []

    if clearance_class in ["molto basso", "basso"] and dec_dir == "superiore":
        notes.append("Quadro compatibile con lente piatta o sagittale insufficiente.")
        suggestions.append("Valutare aumento della profondità sagittale.")
    elif clearance_class in ["alto", "molto alto"] and dec_dir == "inferiore":
        notes.append("Quadro compatibile con lente profonda o chiusa.")
        suggestions.append("Valutare riduzione della profondità sagittale o apertura del landing.")
    elif clearance_class in ["alto", "molto alto"] and dec_dir == "centrata":
        notes.append("Clearance elevato con centraggio discreto: possibile lente troppo profonda.")
        suggestions.append("Valutare riduzione del vault centrale.")
    elif clearance_class in ["molto basso", "basso"] and dec_dir == "centrata":
        notes.append("Clearance ridotto con centraggio discreto: possibile vault borderline o insufficiente.")
        suggestions.append("Valutare incremento del vault.")
    elif dec_dir in ["nasale", "temporale"]:
        notes.append("Decentramento orizzontale: possibile asimmetria sclero-congiuntivale.")
        suggestions.append("Valutare geometria torica o landing asimmetrico.")
    else:
        notes.append("Pattern complessivamente compatibile con fitting relativamente regolare.")
        suggestions.append("Confermare clinicamente con osservazione dinamica.")

    if zone_sample["periferica"] < zone_ref["periferica"] * 0.8:
        notes.append("Segnale periferico ridotto rispetto alla reference.")
        suggestions.append("Valutare periferia più aperta se clinicamente coerente.")

    if zone_sample["centrale"] > zone_ref["centrale"] * 1.25:
        notes.append("Pooling centrale aumentato rispetto alla reference.")
    elif zone_sample["centrale"] < zone_ref["centrale"] * 0.75:
        notes.append("Segnale centrale ridotto rispetto alla reference.")

    return notes, suggestions

# -------------------------
# Interfaccia
# -------------------------

with st.sidebar:
    st.header("Parametri")
    target_clearance = st.number_input(
        "Clearance reference centrale (µm stimati)",
        min_value=0.1,
        value=5.0,
        step=0.5
    )
    green_threshold = st.slider("Soglia verde", 0, 100, 25, 1)
    show_debug = st.checkbox("Mostra dettagli tecnici", value=False)

col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader("Carica immagine IDEALE", type=["jpg", "jpeg", "png"], key="ref")

with col2:
    sample_file = st.file_uploader("Carica immagine CAMPIONE", type=["jpg", "jpeg", "png"], key="sample")

if ref_file and sample_file:
    ref_img = load_image(ref_file)
    sample_img = load_image(sample_file)

    # Rilevamento lente su immagini originali
    ref_gray_full = to_gray(ref_img)
    sample_gray_full = to_gray(sample_img)

    rx, ry, rr = detect_lens(ref_gray_full)
    sx, sy, sr = detect_lens(sample_gray_full)

    # Crop sulle zone lente
    ref_crop = crop_with_padding(ref_img, (rx, ry), rr, pad_factor=1.25)
    sample_crop = crop_with_padding(sample_img, (sx, sy), sr, pad_factor=1.25)

    # Rileva nuovamente sulla crop reference
    ref_gray_crop = to_gray(ref_crop)
    sample_gray_crop = to_gray(sample_crop)

    crx, cry, crr = detect_lens(ref_gray_crop)
    csx, csy, csr = detect_lens(sample_gray_crop)

    # Normalizzazione scala campione -> reference
    sample_scaled = resize_to_match_radius(ref_crop, sample_crop, crr, max(csr, 1))
    sample_scaled = center_crop_or_pad(sample_scaled, ref_crop.shape[0], ref_crop.shape[1])

    # Normalizzazione cromatica
    sample_norm = normalize_green_channel(ref_crop, sample_scaled)

    # Gray uniformati
    ref_gray = to_gray(ref_crop)
    sample_gray = to_gray(sample_norm)

    # Re-detect dopo uniformazione
    urx, ury, urr = detect_lens(ref_gray)
    usx, usy, usr = detect_lens(sample_gray)

    # Maschere verde
    ref_mask = get_green_mask(ref_crop, threshold=green_threshold)
    sample_mask = get_green_mask(sample_norm, threshold=green_threshold)

    ref_fluo_centroid = centroid_from_mask(ref_mask)
    sample_fluo_centroid = centroid_from_mask(sample_mask)

    # Overlay
    ref_overlay = draw_overlay(ref_crop, (urx, ury), urr, ref_fluo_centroid)
    sample_overlay = draw_overlay(sample_norm, (usx, usy), usr, sample_fluo_centroid)

    # Profili
    ref_green = ref_crop[:, :, 1]
    sample_green = sample_norm[:, :, 1]

    ref_profile = smooth_profile(radial_profile(ref_green, (urx, ury)))
    sample_profile = smooth_profile(radial_profile(sample_green, (usx, usy)))

    clearance_profile = compute_clearance_profile(ref_profile, sample_profile, ref_clearance_um=target_clearance)

    usable_radius = min(urr, usr, len(ref_profile), len(sample_profile), len(clearance_profile)) - 1
    usable_radius = max(usable_radius, 10)

    ref_zone = zone_means(ref_profile[:usable_radius], usable_radius)
    sample_zone = zone_means(sample_profile[:usable_radius], usable_radius)

    center_clearance = float(np.mean(clearance_profile[:max(3, int(usable_radius * 0.1))]))
    clearance_class = classify_clearance(center_clearance, target_clearance)

    dec_dir, dx, dy, dist = decentration_direction((usx, usy), sample_fluo_centroid)
    dec_amt = decentration_amount(dist, usr)

    notes, suggestions = clinical_interpretation(clearance_class, dec_dir, sample_zone, ref_zone)

    diff_img = heatmap_diff(ref_gray, sample_gray)

    tabs = st.tabs([
        "Analisi clinica",
        "Immagini",
        "Istogramma verde",
        "Profilo radiale",
        "Clearance stimato",
        "Heatmap",
        "Maschere"
    ])

    with tabs[0]:
        st.subheader("Referto sintetico")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Clearance centrale stimato", f"{center_clearance:.2f} µm")
        with c2:
            st.metric("Classe clearance", clearance_class)
        with c3:
            st.metric("Decentramento", f"{dec_dir} ({dec_amt})")

        st.markdown("**Interpretazione clinica**")
        for n in notes:
            st.write(f"- {n}")

        st.markdown("**Suggerimenti**")
        for s in suggestions:
            st.write(f"- {s}")

        st.markdown("**Analisi zonale del segnale**")
        st.write(
            {
                "reference": ref_zone,
                "campione": sample_zone,
            }
        )

    with tabs[1]:
        st.subheader("Immagini confrontate")
        c1, c2 = st.columns(2)
        with c1:
            st.image(ref_overlay, caption="Reference con overlay", use_container_width=True)
        with c2:
            st.image(sample_overlay, caption="Campione normalizzato con overlay", use_container_width=True)

    with tabs[2]:
        st.subheader("Istogramma del canale verde")

        ref_g = ref_crop[:, :, 1].ravel()
        sam_g = sample_norm[:, :, 1].ravel()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(ref_g, bins=60, alpha=0.5, label="Reference")
        ax.hist(sam_g, bins=60, alpha=0.5, label="Campione normalizzato")
        ax.set_xlabel("Intensità verde")
        ax.set_ylabel("Numero pixel")
        ax.set_title("Distribuzione del verde")
        ax.legend()
        st.pyplot(fig)

    with tabs[3]:
        st.subheader("Profilo radiale fluoresceinico")

        n = usable_radius
        x = np.arange(n)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, ref_profile[:n], label="Reference")
        ax.plot(x, sample_profile[:n], label="Campione")
        ax.set_xlabel("Distanza dal centro (pixel)")
        ax.set_ylabel("Intensità media verde")
        ax.set_title("Profilo radiale")
        ax.legend()
        st.pyplot(fig)

    with tabs[4]:
        st.subheader("Profilo di clearance stimato")

        n = usable_radius
        x = np.arange(n)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, clearance_profile[:n], label="Clearance stimato")
        ax.axhline(target_clearance, linestyle="--", label="Reference target")
        ax.set_xlabel("Distanza dal centro (pixel)")
        ax.set_ylabel("µm stimati")
        ax.set_title("Grafico ipotetico del clearance")
        ax.legend()
        st.pyplot(fig)

    with tabs[5]:
        st.subheader("Heatmap differenze")
        st.image(diff_img, caption="Differenze tra reference e campione normalizzato", use_container_width=True)

    with tabs[6]:
        st.subheader("Maschere del segnale verde")
        c1, c2 = st.columns(2)
        with c1:
            st.image(ref_mask, caption="Maschera verde reference", use_container_width=True)
        with c2:
            st.image(sample_mask, caption="Maschera verde campione", use_container_width=True)

    if show_debug:
        st.markdown("---")
        st.subheader("Dettagli tecnici")
        st.write({
            "reference_lens_center": (rx, ry),
            "reference_lens_radius": rr,
            "sample_lens_center": (sx, sy),
            "sample_lens_radius": sr,
            "uniform_ref_center": (urx, ury),
            "uniform_ref_radius": urr,
            "uniform_sample_center": (usx, usy),
            "uniform_sample_radius": usr,
            "sample_fluo_centroid": sample_fluo_centroid,
            "decentration_dx_px": dx,
            "decentration_dy_px": dy,
            "decentration_distance_px": dist,
        })

else:
    st.info("Carica una immagine reference e una immagine campione per iniziare.")
