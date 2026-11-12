import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from app_config import (
    MODELS,
    OUT,
    id_num,
    inject_css,
    page_head,
    section_head,
)

INK = "#201B15"
INK2 = "#4A4239"
MUTED = "#B9AFA3"
LINE = "#E7DFD3"
GRID = "#F1EBE1"
PRIMARY = "#E8590C"

inject_css()

page_head(
    "📊 Dashboard Hasil Training",
    "Ringkasan pelatihan, evaluasi test set, dan eksperimen — seluruh angka dibaca langsung dari outputs/modelling.",
)


@st.cache_data(show_spinner=False)
def read_csv(name, **kwargs):
    return pd.read_csv(OUT / name, **kwargs)


def style_fig(fig, height=380, xtitle="Epoch"):
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color=INK2),
        margin=dict(l=10, r=20, t=52, b=10),
        legend=dict(orientation="h", y=1.14, x=0),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(title_text=xtitle, showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)


def model_card_html(m):
    tag_cls = "tag-best" if m["key"] == "mnv2" else "tag-light"
    return (
        f"<div class='model-card'>"
        f"<div class='model-top'><div class='m-emoji'>{m['emoji']}</div>"
        f"<div><h3>{m['label']}</h3><p class='m-tagline'>{m['tagline']}</p></div>"
        f"<span class='model-tag {tag_cls}'>{m['tag']}</span></div>"
        f"<div class='chip-row'>"
        f"<div class='chip'><b>{id_num(m['test_acc'] * 100)}%</b><span>Akurasi test</span></div>"
        f"<div class='chip'><b>{id_num(m['f1'] * 100)}%</b><span>F1 macro</span></div>"
        f"<div class='chip'><b>{m['params']}</b><span>Parameter</span></div>"
        f"<div class='chip'><b>{m['size']}</b><span>Ukuran</span></div>"
        f"<div class='chip'><b>{id_num(m['ms'])} ms</b><span>Per citra</span></div>"
        f"</div></div>"
    )


tab_ringkasan, tab_kurva, tab_eval, tab_eksperimen, tab_analisis = st.tabs(
    [
        "🧾 Ringkasan",
        "📈 Kurva Training",
        "🎯 Evaluasi Test",
        "🧪 Eksperimen",
        "🔍 Analisis Kesalahan",
    ]
)

with tab_ringkasan:
    st.markdown(
        "<div class='model-grid'>"
        + model_card_html(MODELS["mnv2"])
        + model_card_html(MODELS["cnn"])
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    section_head("Perbandingan metrik lengkap")
    st.dataframe(read_csv("tabel_perbandingan.csv", index_col=0), use_container_width=True)

    section_head("Efisiensi training")
    st.dataframe(read_csv("tabel_efisiensi.csv", index_col=0), use_container_width=True)

    st.markdown(
        "<div class='note'><b>Kesimpulan utama.</b> MobileNetV2 jauh lebih akurat (98,4% vs 83,6%) "
        "dan stabil di semua kelas — hanya 1 kesalahan dari 61 citra test. Custom CNN unggul dalam "
        "ukuran file (3,1 MB vs 24,6 MB) dan kecepatan inferensi (4,7 ms vs 25,9 ms per citra), "
        "tetapi lemah mengenali gudeg (recall hanya 55%).</div>",
        unsafe_allow_html=True,
    )

with tab_kurva:
    model_key = st.pills(
        "Model",
        options=["cnn", "mnv2"],
        format_func=lambda k: f"{MODELS[k]['emoji']} {MODELS[k]['label']}",
        default="mnv2",
        key="kurva_model",
    )
    m = MODELS[model_key]
    h = read_csv(m["history"])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=h["epoch"], y=h["loss"], name="train loss",
                line=dict(color=MUTED, width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=h["epoch"], y=h["val_loss"], name="val loss",
                line=dict(color=PRIMARY, width=2.5),
            )
        )
        best_i = h["val_loss"].idxmin()
        fig.add_trace(
            go.Scatter(
                x=[h.loc[best_i, "epoch"]], y=[h.loc[best_i, "val_loss"]],
                name="epoch terbaik", mode="markers",
                marker=dict(color=INK, size=11, symbol="star"),
            )
        )
        fig.update_layout(title_text="Loss")
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=h["epoch"], y=h["accuracy"], name="train acc",
                line=dict(color=MUTED, width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=h["epoch"], y=h["val_accuracy"], name="val acc",
                line=dict(color=PRIMARY, width=2.5),
            )
        )
        fig.update_layout(title_text="Accuracy")
        style_fig(fig)
        fig.update_yaxes(tickformat=".0%", gridcolor=GRID, zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    if "fase" in h.columns:
        boundary = h.loc[h["fase"] == 2, "epoch"].min()
        st.info(
            f"Titik di epoch {int(boundary) - 1} menandai akhir fase 1 (feature extraction). "
            "Fase 2 (fine-tuning) membuka 54 layer backbone dengan learning rate 100× lebih kecil.",
            icon="ℹ️",
        )

with tab_eval:
    st.markdown(
        "<div class='section-head'><h2>Confusion Matrix — kedua model</h2></div>",
        unsafe_allow_html=True,
    )
    st.image(str(OUT / "confusion_matrix.png"), use_container_width=False)
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    model_key = st.pills(
        "Laporan klasifikasi",
        options=["cnn", "mnv2"],
        format_func=lambda k: f"{MODELS[k]['emoji']} {MODELS[k]['label']}",
        default="mnv2",
        key="eval_model",
    )
    rep = read_csv(MODELS[model_key]["report"], index_col=0)
    st.dataframe(rep.style.format(precision=3), use_container_width=True)
    st.caption("Dievaluasi satu kali pada 61 citra test yang tidak pernah dilihat saat training (perangkat ditahan + foto internet untuk gudeg).")

with tab_eksperimen:
    section_head("Pengaruh dropout (Custom CNN, divalidasi di validation set)")
    d = read_csv("eksperimen_dropout.csv")
    best_drop = d.loc[d["val_loss"].idxmin(), "dropout"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        colors = [PRIMARY if r == best_drop else MUTED for r in d["dropout"]]
        fig.add_trace(
            go.Bar(
                x=[id_num(x, 1) for x in d["dropout"]], y=d["val_loss"],
                marker_color=colors, text=[id_num(v, 3) for v in d["val_loss"]],
                textposition="outside", name="val loss",
            )
        )
        fig.update_layout(title_text="Validation loss per dropout")
        style_fig(fig, height=340, xtitle="Dropout")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[id_num(x, 1) for x in d["dropout"]], y=d["val_acc"],
                marker_color=colors, text=[id_num(v, 3) for v in d["val_acc"]],
                textposition="outside", name="val acc",
            )
        )
        fig.update_layout(title_text="Validation accuracy per dropout")
        style_fig(fig, height=340, xtitle="Dropout")
        fig.update_yaxes(tickformat=".0%", gridcolor=GRID, zeroline=False)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"<div class='note'>Dropout <b>{id_num(best_drop, 1)}</b> memberi val_loss terendah sehingga dipakai di model utama.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    section_head("Strategi training (Custom CNN)")
    s = read_csv("eksperimen_strategi_training.csv")
    label_map = {
        "A. LR tetap, tanpa callback": "A · LR tetap",
        "B. LR tetap + EarlyStopping": "B · LR tetap + EarlyStopping",
        "Model utama (LR turun + EarlyStopping)": "Model utama · LR turun + EarlyStopping",
    }
    s["label"] = s["konfigurasi"].map(label_map)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=s["label"], x=s["val_loss_model"], orientation="h",
            marker_color=[PRIMARY if "utama" in l else MUTED for l in s["label"]],
            text=[id_num(v, 3) for v in s["val_loss_model"]],
            textposition="outside",
        )
    )
    fig.update_layout(title_text="Val loss model yang dipakai per strategi", height=300)
    style_fig(fig, height=300, xtitle="val loss")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(s.drop(columns=["label"]), use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='note'>Pemilihan model dan hyperparameter hanya memakai <b>validation set</b>. "
        "Test set dibuka satu kali di akhir setelah model dikunci.</div>",
        unsafe_allow_html=True,
    )

with tab_analisis:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            "<div class='section-head'><h2>Contoh prediksi — Custom CNN</h2></div>",
            unsafe_allow_html=True,
        )
        st.image(str(OUT / "contoh_prediksi_custom_cnn.png"), use_container_width=True)
    with c2:
        st.markdown(
            "<div class='section-head'><h2>Contoh prediksi — MobileNetV2</h2></div>",
            unsafe_allow_html=True,
        )
        st.image(str(OUT / "contoh_prediksi_mobilenetv2.png"), use_container_width=True)
    st.markdown(
        "<div class='note' style='margin-top:1.2rem'>"
        "<b>Pol kesalahan.</b> Custom CNN salah pada 10 citra, hampir semuanya gudeg yang terprediksi sate klathak. "
        "MobileNetV2 hanya salah pada 1 citra (bakpia_012, keyakinan 0,99). "
        "Daftar lengkap 61 prediksi ada di outputs/modelling/prediksi_test.csv.</div>",
        unsafe_allow_html=True,
    )
