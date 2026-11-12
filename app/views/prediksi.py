import io
import sys
import time
from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from app_config import (
    MODELS,
    ASSETS,
    CLASSES,
    class_meta,
    class_order,
    id_num,
    inject_css,
    page_head,
)

inject_css()

page_head(
    "📸 Prediksi Makanan",
    "Unggah foto, ambil dari kamera, atau pakai contoh — model langsung menebak makanannya.",
)


@st.cache_resource(show_spinner="Memuat model…")
def load_model(key):
    import tensorflow as tf

    model = tf.keras.models.load_model(ROOT / "model" / MODELS[key]["file"])
    model.predict(tf.zeros((1, 224, 224, 3)), verbose=0)
    return model


def preprocess(img_bytes):
    import numpy as np
    import tensorflow as tf
    from PIL import Image, ImageOps

    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    tensor = tf.convert_to_tensor(np.asarray(pil), dtype=tf.float32)
    return pil, tf.image.resize(tensor, (224, 224))[None, ...]


def render_result(pil, probs, model_key, ms):
    order = class_order()
    pairs = sorted(zip(order, probs), key=lambda x: -x[1])
    top_key, top_p = pairs[0]
    meta = class_meta(top_key)
    m = MODELS[model_key]

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.image(pil, use_container_width=True)
        st.caption("Pratinjau gambar setelah koreksi orientasi EXIF.")
    with right:
        st.markdown(
            f"<div class='result-card'>"
            f"<div class='result-emoji'>{meta['emoji']}</div>"
            f"<div><div class='result-kicker'>Prediksi</div>"
            f"<div class='result-name'>{meta['name']}</div>"
            f"<div class='result-conf'>keyakinan {id_num(top_p * 100)}%</div></div></div>",
            unsafe_allow_html=True,
        )
        rows = []
        for key, p in pairs:
            cm = class_meta(key)
            rows.append(
                f"<div class='bar-row'>"
                f"<div class='bar-label'>{cm['emoji']} {cm['name']}</div>"
                f"<div class='bar-track'><div class='bar-fill' style='width:{p * 100:.2f}%;background:{cm['color']}'></div></div>"
                f"<div class='bar-val'>{id_num(p * 100)}%</div></div>"
            )
        st.markdown(f"<div class='bars'>{''.join(rows)}</div>", unsafe_allow_html=True)
        if top_p < 0.6:
            st.warning(
                "Keyakinan model rendah. Foto mungkin bukan Bakpia, Gudeg, atau Sate Klathak — atau pengambilannya tidak biasa.",
                icon="⚠️",
            )
        st.caption(
            f"Model: {m['label']} · inferensi {id_num(ms, 0)} ms (CPU) · input 224×224, normalisasi di dalam model"
        )


model_key = st.pills(
    "Pilih model",
    options=["cnn", "mnv2"],
    format_func=lambda k: f"{MODELS[k]['emoji']} {MODELS[k]['label']}",
    default="mnv2",
    key="model_pills",
)

m = MODELS[model_key]
st.markdown(
    f"<div class='model-strip'><div class='ms-icon'>{m['emoji']}</div>"
    f"<div class='ms-body'><b>{m['label']}</b> — {m['tagline']}</div>"
    f"<div class='ms-chips'><span>akurasi test {id_num(m['test_acc'] * 100)}%</span>"
    f"<span>{m['params']} parameter</span><span>{m['size']}</span>"
    f"<span>{id_num(m['ms'])} ms/citra</span></div></div>",
    unsafe_allow_html=True,
)

tab_upload, tab_camera, tab_contoh = st.tabs(
    ["📁 Unggah gambar", "📷 Kamera", "🖼️ Contoh gambar"]
)

img_bytes = None

with tab_upload:
    up = st.file_uploader(
        "Pilih foto makanan",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )
    if up is not None:
        img_bytes = up.getvalue()
        st.session_state.pop("contoh", None)

with tab_camera:
    cam = st.camera_input("Ambil foto langsung dari kamera")
    if cam is not None and up is None:
        img_bytes = cam.getvalue()
        st.session_state.pop("contoh", None)

with tab_contoh:
    for c in CLASSES:
        st.markdown(
            f"**{c['emoji']} {c['name']}** — {c['desc']}"
        )
        cols = st.columns(3, gap="small")
        for col, f in zip(cols, sorted((ASSETS / "contoh").glob(f"{c['key']}*.png"))):
            with col:
                st.image(str(f), use_container_width=True)
                if st.button("Prediksi", key=f"ex_{f.name}", use_container_width=True):
                    st.session_state["contoh"] = str(f)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    if st.session_state.get("contoh"):
        st.caption(f"Contoh aktif: `{Path(st.session_state['contoh']).name}`")
        if img_bytes is None:
            img_bytes = Path(st.session_state["contoh"]).read_bytes()

if img_bytes is None:
    st.info(
        "Unggah foto, aktifkan kamera, atau pilih salah satu contoh untuk memulai prediksi.",
        icon="💡",
    )
else:
    try:
        pil, arr = preprocess(img_bytes)
    except Exception:
        st.error("File tidak dapat dibaca sebagai gambar. Coba file JPG/PNG lain.")
        st.stop()
    model = load_model(model_key)
    t0 = time.perf_counter()
    probs = model.predict(arr, verbose=0)[0]
    ms = (time.perf_counter() - t0) * 1000
    render_result(pil, probs, model_key, ms)
