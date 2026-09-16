"""ローカル検査画面の共通外観。実行状態やデータには触れない。"""

from html import escape

import streamlit as st
from src.ui.display_text import display_text


APP_CSS = """
<style>
.stApp { background: #0b1120; color: #e6edf7; }
[data-testid="stHeader"] { background: #0b1120; }
.stMainBlockContainer { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 1540px; }
h1 { font-size: 1.65rem !important; font-weight: 650 !important; letter-spacing: -.035em; }
h2 { font-size: 1.22rem !important; font-weight: 600 !important; padding-top: .3rem !important; }
h3 { font-size: 1.02rem !important; font-weight: 600 !important; }
[data-testid="stCaptionContainer"] { color: #9baec7; }
[data-testid="stCaptionContainer"] p { color: #9baec7; font-size: .79rem; }
[data-testid="stMetric"] {
    background: linear-gradient(120deg, #172439, #111b2c);
    border: 1px solid #26374e; border-radius: 10px; padding: 1rem 1.1rem;
    border-top: 2px solid #38bdf8;
}
[data-testid="stMetricLabel"] p { color: #aebfd5; font-size: .82rem; }
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 600; letter-spacing: -.035em; }
[data-testid="stMetricDelta"] { font-size: .78rem; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #26374e !important; }
[data-testid="stExpander"] { border-color: #26374e; background: #101a2a; }
[data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #26374e; }
[data-baseweb="tab"] { height: 2.9rem; }
[data-baseweb="tab"] p { font-size: .9rem; font-weight: 600; }
.stButton button { border-radius: 7px; font-size: .86rem; font-weight: 600; }
.stTextInput input { font-size: .83rem; }
.brand { display: flex; justify-content: space-between; align-items: center;
    gap: 1rem; padding-bottom: 1.3rem; margin-bottom: .3rem; border-bottom: 1px solid #26374e; }
.brand-mark { color: #38bdf8; font-size: .73rem; letter-spacing: .2em; font-weight: 700; }
.brand h1 { padding: .4rem 0 .3rem !important; margin: 0; color: #e6edf7; }
.brand p { margin: 0; color: #9baec7; font-size: .82rem; }
.local-badge { color: #9fcbdf; background: #12283a; border: 1px solid #264b61;
    border-radius: 6px; padding: .45rem .7rem; font-size: .69rem; letter-spacing: .1em; white-space: nowrap; }
.section-label { color: #71a4c8; font-size: .67rem; font-weight: 700; letter-spacing: .16em;
    margin: .1rem 0 .3rem; }
.empty-workspace { border: 1px dashed #30445f; border-radius: 10px;
    padding: 2.4rem 2rem; background: #0e1929; margin: 1rem 0; }
.empty-workspace h3 { color: #d0deee; margin: 0 0 .5rem; }
.empty-workspace p { color: #9baec7; font-size: .86rem; margin: 0; }
@media (max-width: 700px) {
    .stMainBlockContainer { padding: 1.5rem 1rem; }
    .brand { align-items: flex-start; }
    .brand h1 { font-size: 1.25rem !important; }
    .local-badge { display: none; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
}
</style>
"""


def apply_theme() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_brand() -> None:
    st.markdown(
        '<div class="brand"><div><h1>ASSEMBLY INSIGHT</h1></div>'
        '<span class="local-badge">LOCAL WORKSPACE</span></div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{escape(text)}</div>', unsafe_allow_html=True)


def empty_workspace(title: str, description: str) -> None:
    st.markdown(
        f'<div class="empty-workspace"><h3>{escape(display_text(title))}</h3>'
        f'<p>{escape(display_text(description))}</p></div>', unsafe_allow_html=True,
    )
