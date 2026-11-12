import sys
from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from app_config import (
    CLASSES,
    MODELS,
    b64_img,
    example_image,
    footer,
    inject_css,
    section_head,
)

inject_css()

st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">PROYEK UTS · DEEP LEARNING · 2026</div>
      <h1>Kenali makanan khas<br><span class="accent">Yogyakarta</span> dari foto</h1>
      <p>Unggah satu foto, lalu dua model deep learning — Custom CNN dan MobileNetV2 —
      langsung menebak apakah itu Bakpia, Gudeg, atau Sate Klathak
      beserta tingkat keyakinannya.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cta1, cta2 = st.columns([1, 1], gap="small")
with cta1:
    if st.button(
        "📸 Coba Prediksi Sekarang", type="primary", use_container_width=True
    ):
        st.switch_page("views/prediksi.py")
with cta2:
    if st.button("📊 Lihat Dashboard Hasil", use_container_width=True):
        st.switch_page("views/dashboard.py")

st.markdown(
    """
    <div class="stats-strip">
      <div class="stat"><div class="stat-v">388</div><div class="stat-l">citra dataset (3 kelas)</div></div>
      <div class="stat"><div class="stat-v"><em>98,4%</em></div><div class="stat-l">akurasi test terbaik</div></div>
      <div class="stat"><div class="stat-v">2</div><div class="stat-l">model dibandingkan</div></div>
      <div class="stat"><div class="stat-v">4,7 <span style="font-size:.9rem">ms</span></div><div class="stat-l">inferensi tercepat</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

section_head(
    "Tiga kelas yang dikenali",
    "Dataset 388 citra: 204 foto sendiri dari 4 perangkat kamera berbeda + 199 foto internet, dibersihkan dari duplikat.",
)

cards = []
for c in CLASSES:
    img = example_image(c["key"])
    src = b64_img(img) if img else ""
    cards.append(
        f"<div class='class-card'><img src='{src}' alt='{c['name']}'>"
        f"<div class='pad'><h3>{c['emoji']} {c['name']}</h3><p>{c['desc']}</p></div></div>"
    )
st.markdown(
    f"<div class='class-grid'>{''.join(cards)}</div>", unsafe_allow_html=True
)

section_head(
    "Dua pendekatan, satu tujuan",
    "Kedua model dilatih pada dataset yang sama dan dievaluasi sekali di test set 61 citra yang tidak pernah dilihat saat training.",
)

model_cards = []
for m in MODELS.values():
    tag_cls = "tag-best" if m["key"] == "mnv2" else "tag-light"
    model_cards.append(
        f"<div class='model-card'>"
        f"<div class='model-top'><div class='m-emoji'>{m['emoji']}</div>"
        f"<div><h3>{m['label']}</h3><p class='m-tagline'>{m['tagline']}</p></div>"
        f"<span class='model-tag {tag_cls}'>{m['tag']}</span></div>"
        f"<p class='m-desc'>{m['desc']}</p>"
        f"<div class='chip-row'>"
        f"<div class='chip'><b>{str(m['test_acc']*100).replace('.', ',')[:5]}%</b><span>Akurasi test</span></div>"
        f"<div class='chip'><b>{m['params']}</b><span>Parameter</span></div>"
        f"<div class='chip'><b>{m['size']}</b><span>Ukuran</span></div>"
        f"<div class='chip'><b>{str(m['ms']).replace('.', ',')} ms</b><span>Per citra</span></div>"
        f"</div></div>"
    )
st.markdown(
    f"<div class='model-grid'>{''.join(model_cards)}</div>",
    unsafe_allow_html=True,
)

footer()
