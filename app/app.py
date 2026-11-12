import sys
from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from app_config import inject_css

st.set_page_config(
    page_title="Klasifikasi Makanan Khas Yogyakarta",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

pages = [
    st.Page(str(BASE / "views" / "beranda.py"), title="Beranda", icon="🏠", default=True),
    st.Page(str(BASE / "views" / "prediksi.py"), title="Prediksi", icon="📸"),
    st.Page(str(BASE / "views" / "dashboard.py"), title="Dashboard", icon="📊"),
]

nav = st.navigation(pages)
nav.run()

with st.sidebar:
    st.markdown(
        "<div class='sidebar-note'>🍽️ <b>Klasifikasi Makanan Yogyakarta</b><br>"
        "Bakpia · Gudeg · Sate Klathak<br>Custom CNN vs MobileNetV2</div>",
        unsafe_allow_html=True,
    )
