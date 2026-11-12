import base64
import json
import sys
from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OUT = ROOT / "outputs" / "modelling"
ASSETS = BASE / "assets"

if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

CLASSES = [
    {
        "key": "bakpia",
        "name": "Bakpia",
        "emoji": "🥮",
        "color": "#F59F00",
        "desc": "Pastri tipis berisi kacang hijau, oleh-oleh paling ikonik Yogyakarta.",
    },
    {
        "key": "gudeg",
        "name": "Gudeg",
        "emoji": "🍛",
        "color": "#D9480F",
        "desc": "Nangka muda dimasak santan dan gula aren berjam-jam, sarapan khas Jogja.",
    },
    {
        "key": "sate_klathak",
        "name": "Sate Klathak",
        "emoji": "🍢",
        "color": "#C92A2A",
        "desc": "Sate kambing tusukan besi tanpa bumbu manis, legendaris dari Bantul.",
    },
]

MODELS = {
    "cnn": {
        "key": "cnn",
        "label": "Custom CNN",
        "emoji": "🧱",
        "file": "custom_cnn_best.keras",
        "history": "history_custom_cnn.csv",
        "report": "classification_report_custom_cnn.csv",
        "tag": "RINGAN & CEPAT",
        "tagline": "Dibangun dari nol, hanya 249 ribu parameter",
        "params": "249.283",
        "size": "3,1 MB",
        "test_acc": 0.8361,
        "f1": 0.8261,
        "ms": 4.7,
        "desc": "Empat blok konvolusi sederhana dengan Global Average Pooling. Kalah akurat, tetapi menang di ukuran file dan kecepatan inferensi.",
    },
    "mnv2": {
        "key": "mnv2",
        "label": "MobileNetV2",
        "emoji": "🚀",
        "file": "mobilenetv2_best.keras",
        "history": "history_mobilenetv2.csv",
        "report": "classification_report_mobilenetv2.csv",
        "tag": "PALING AKURAT",
        "tagline": "Transfer learning ImageNet + fine-tuning",
        "params": "2.261.827",
        "size": "24,6 MB",
        "test_acc": 0.9836,
        "f1": 0.9837,
        "ms": 25.9,
        "desc": "Backbone pre-trained ImageNet dengan dua fase training (feature extraction lalu fine-tuning). Akurasi test 98,4% dengan hanya 1 kesalahan dari 61 citra.",
    },
}


def class_meta(key):
    return next(c for c in CLASSES if c["key"] == key)


def class_order():
    return json.loads((OUT / "model_info.json").read_text(encoding="utf-8"))["class_names"]


def id_num(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",")


def b64_img(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def example_image(class_key):
    files = sorted((ASSETS / "contoh").glob(f"{class_key}*.png"))
    return files[0] if files else None


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root{
  --primary:#E8590C;
  --primary-dark:#C74A08;
  --primary-soft:#FDEEE2;
  --ink:#201B15;
  --ink-2:#4A4239;
  --muted:#8A8075;
  --bg:#FAF8F4;
  --surface:#FFFFFF;
  --line:#EBE4D9;
  --shadow:0 10px 30px rgba(32,27,21,.06);
}

html,body,.stApp,[data-testid="stMarkdown"]{
  font-family:'Plus Jakarta Sans',ui-sans-serif,system-ui,-apple-system,sans-serif;
}
.stApp{background:var(--bg);color:var(--ink);}
p,li{color:var(--ink-2);}

#MainMenu,footer{visibility:hidden;}
[data-testid="stToolbar"]{display:none;}
header[data-testid="stHeader"]{background:rgba(250,248,244,.88);backdrop-filter:blur(10px);}

section[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--line);}
[data-testid="stSidebarNav"]{padding-top:.6rem;gap:.15rem;}
[data-testid="stSidebarNavLink"]{border-radius:12px;min-height:2.4rem;}
[data-testid="stSidebarNavLink"] span{font-size:.93rem;font-weight:600;color:var(--ink-2);}
[data-testid="stSidebarNavLink"]:hover{background:var(--primary-soft);}
[data-testid="stSidebarNavLink"]:hover span{color:var(--primary);}
.sidebar-note{margin-top:.9rem;padding:1.1rem .4rem .3rem;border-top:1px solid var(--line);color:var(--muted);font-size:.78rem;line-height:1.6;}

.block-container,[data-testid="stMainBlockContainer"]{max-width:1180px;padding-top:1.5rem;padding-bottom:4rem;}

.page-head h1{font-size:2rem;font-weight:800;letter-spacing:-.025em;margin:0 0 .3rem;}
.page-head p{color:var(--muted);font-size:.98rem;margin:0 0 1.6rem;}
.section-head h2{font-size:1.3rem;font-weight:800;letter-spacing:-.015em;margin:0 0 .25rem;}
.section-head p{color:var(--muted);font-size:.9rem;margin:0 0 1.1rem;}

.hero{
  position:relative;overflow:hidden;border-radius:26px;color:#fff;
  background:linear-gradient(135deg,#241C15 0%,#3A281A 55%,#5A341C 100%);
  padding:3.2rem 3rem 2.9rem;box-shadow:0 18px 44px rgba(32,27,21,.18);
}
.hero::after{
  content:"";position:absolute;right:-110px;top:-110px;width:340px;height:340px;border-radius:50%;
  background:radial-gradient(closest-side,rgba(232,89,12,.55),transparent);
}
.hero-badge{
  display:inline-flex;align-items:center;gap:.5rem;font-size:.7rem;font-weight:700;letter-spacing:.16em;
  color:#FFC9A3;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.16);
  border-radius:999px;padding:.38rem .95rem;margin-bottom:1.2rem;
}
.hero h1{font-size:2.9rem;font-weight:800;letter-spacing:-.03em;line-height:1.08;margin:0 0 1rem;}
.hero h1 .accent{color:#FF9B6A;}
.hero p{color:rgba(255,255,255,.78);font-size:1.02rem;line-height:1.65;max-width:35rem;margin:0;}

.stats-strip{display:flex;flex-wrap:wrap;gap:1rem;margin:1.8rem 0 2.6rem;}
.stat{flex:1;min-width:150px;background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:1.15rem 1.3rem;box-shadow:var(--shadow);}
.stat-v{font-size:1.75rem;font-weight:800;letter-spacing:-.02em;}
.stat-v em{color:var(--primary);font-style:normal;}
.stat-l{color:var(--muted);font-size:.8rem;margin-top:.2rem;}

.card{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:1.4rem;box-shadow:var(--shadow);}

.class-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.1rem;}
@media(max-width:820px){.class-grid{grid-template-columns:1fr;}}
.class-card{background:var(--surface);border:1px solid var(--line);border-radius:20px;overflow:hidden;box-shadow:var(--shadow);}
.class-card img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;}
.class-card .pad{padding:1.05rem 1.25rem 1.3rem;}
.class-card h3{margin:0 0 .3rem;font-size:1.12rem;font-weight:700;}
.class-card p{margin:0;color:var(--muted);font-size:.87rem;line-height:1.55;}

.model-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;}
@media(max-width:820px){.model-grid{grid-template-columns:1fr;}}
.model-card{background:var(--surface);border:1px solid var(--line);border-radius:20px;padding:1.5rem 1.6rem;box-shadow:var(--shadow);}
.model-top{display:flex;align-items:center;gap:.9rem;margin-bottom:.7rem;}
.model-top .m-emoji{font-size:1.7rem;}
.model-top h3{margin:0;font-size:1.15rem;font-weight:800;}
.model-top .m-tagline{margin:.1rem 0 0;font-size:.82rem;color:var(--muted);}
.model-tag{margin-left:auto;font-size:.66rem;font-weight:800;letter-spacing:.1em;padding:.32rem .75rem;border-radius:999px;}
.tag-best{background:#EBFBEE;color:#2F9E44;border:1px solid #C3ECCB;}
.tag-light{background:var(--primary-soft);color:var(--primary);border:1px solid #F5CBA8;}
.model-card .m-desc{color:var(--ink-2);font-size:.9rem;line-height:1.6;margin:0 0 1.15rem;}
.chip-row{display:flex;flex-wrap:wrap;gap:.6rem;}
.chip{background:var(--bg);border:1px solid var(--line);border-radius:14px;padding:.55rem .95rem;}
.chip b{display:block;font-size:1rem;font-weight:800;letter-spacing:-.01em;}
.chip span{font-size:.66rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);}

.model-strip{
  display:flex;flex-wrap:wrap;align-items:center;gap:1rem;background:var(--surface);
  border:1px solid var(--line);border-radius:16px;padding:.95rem 1.25rem;margin:.7rem 0 1.7rem;box-shadow:var(--shadow);
}
.ms-icon{font-size:1.55rem;}
.ms-body{font-size:.95rem;}
.ms-body b{font-weight:800;}
.ms-chips{margin-left:auto;display:flex;flex-wrap:wrap;gap:.45rem;}
.ms-chips span{font-size:.76rem;font-weight:600;background:var(--bg);border:1px solid var(--line);border-radius:999px;padding:.3rem .75rem;color:var(--ink-2);}

.result-card{
  display:flex;align-items:center;gap:1.3rem;background:var(--surface);
  border:1px solid var(--line);border-radius:22px;padding:1.5rem 1.8rem;box-shadow:var(--shadow);
}
.result-emoji{font-size:2.9rem;line-height:1;background:var(--primary-soft);border-radius:18px;padding:.9rem 1rem;}
.result-kicker{font-size:.68rem;font-weight:800;letter-spacing:.18em;color:var(--primary);text-transform:uppercase;}
.result-name{font-size:1.95rem;font-weight:800;letter-spacing:-.02em;margin:.1rem 0 .2rem;}
.result-conf{color:var(--muted);font-size:.95rem;}

.bars{margin-top:1.2rem;}
.bar-row{display:flex;align-items:center;gap:.9rem;margin:.6rem 0;}
.bar-label{width:132px;font-size:.9rem;font-weight:600;color:var(--ink-2);flex-shrink:0;}
.bar-track{flex:1;height:12px;border-radius:999px;background:var(--primary-soft);overflow:hidden;}
.bar-fill{height:100%;border-radius:999px;}
.bar-val{width:66px;text-align:right;font-variant-numeric:tabular-nums;font-size:.9rem;font-weight:700;}

[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:1rem 1.2rem;box-shadow:var(--shadow);}
[data-testid="stMetricLabel"] p{font-size:.72rem!important;font-weight:700!important;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)!important;margin-bottom:.3rem;}
[data-testid="stMetricValue"]{font-weight:800!important;}

.stButton>button{
  border-radius:999px;border:1.5px solid var(--line);background:var(--surface);color:var(--ink);
  font-weight:600;padding:.5rem 1.3rem;transition:all .18s ease;
}
.stButton>button:hover{border-color:var(--primary);color:var(--primary);transform:translateY(-1px);box-shadow:0 8px 18px rgba(232,89,12,.14);}
.stButton>button[kind="primary"]{background:var(--primary);border-color:var(--primary);color:#fff;box-shadow:0 8px 20px rgba(232,89,12,.28);}
.stButton>button[kind="primary"]:hover{background:var(--primary-dark);border-color:var(--primary-dark);color:#fff;transform:translateY(-1px);}

[data-baseweb="tab-list"]{gap:.5rem;border-bottom:1px solid var(--line);}
[data-baseweb="tab"]{
  border-radius:999px!important;padding:.45rem 1.15rem!important;font-weight:600;font-size:.9rem;
  color:var(--ink-2);background:var(--surface);border:1px solid var(--line);margin-bottom:.65rem;
}
[data-baseweb="tab"]:hover{color:var(--primary);border-color:var(--primary);}
[data-baseweb="tab"][aria-selected="true"]{background:var(--ink);color:#fff;border-color:var(--ink);}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important;}

[data-testid="stFileUploaderDropzone"]{border-radius:16px;border:2px dashed var(--line);background:var(--surface);padding:1.7rem 1rem;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--primary);}
[data-testid="stFileUploaderDropzoneInstructions"]{color:var(--muted);font-size:.9rem;}

[data-testid="stImage"] img{border-radius:14px;border:1px solid var(--line);}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;}
[data-testid="stAlert"]{border-radius:14px;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:16px;background:var(--surface);}

.note{
  background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--primary);
  border-radius:14px;padding:1rem 1.3rem;font-size:.9rem;color:var(--ink-2);line-height:1.6;
}

.app-footer{
  margin-top:3.2rem;padding-top:1.5rem;border-top:1px solid var(--line);
  color:var(--muted);font-size:.82rem;text-align:center;
}
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def page_head(title, sub):
    st.markdown(
        f"<div class='page-head'><h1>{title}</h1><p>{sub}</p></div>",
        unsafe_allow_html=True,
    )


def section_head(title, sub=None):
    sub_html = f"<p>{sub}</p>" if sub else ""
    st.markdown(
        f"<div class='section-head'><h2>{title}</h2>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        "<div class='app-footer'>Proyek UTS Deep Learning · Kelompok 12822 · 12828 · 12836 · 12846 · 12850<br>"
        "Model dilatih dengan TensorFlow 2.21 · Aplikasi dibangun dengan Streamlit</div>",
        unsafe_allow_html=True,
    )
