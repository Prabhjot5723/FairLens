"""FairLens visual theme: black / blue / gold in dark mode, paper / blue / gold in light mode.

Usage in app.py (right after st.set_page_config):

    from theme import inject_theme, hero, grade_card, verdict_card, balance_beam, apply_mpl_theme
    inject_theme()
    hero()            # also shows the Light mode switch at the top right
"""
import math
from string import Template
from urllib.parse import quote

import streamlit as st

DARK = dict(
    bg="#000000", panel="#0A1224", panel2="#0E1A33", line="#1B2A4A",
    blue="#2F6BFF", blue_soft="#7FA8FF", gold="#D4A84B", gold_soft="#F0D58A",
    text="#E8ECF5", muted="#8A97B3",
    pass_c="#4DA3FF", warn_c="#E3B341", fail_c="#FF6B6B",
    on_pass="#000000", on_warn="#000000", on_fail="#000000",
    grid="rgba(47,107,255,.10)", glow1="rgba(47,107,255,.20)", glow2="rgba(212,168,75,.08)",
    ring_gold="#D4A84B", ring_blue="#2F6BFF", ring_op="1",
    scheme="dark", dfilter="none", letter="#D4A84B",
)
LIGHT = dict(
    bg="#F6F3EC", panel="#FFFFFF", panel2="#EEF2FB", line="#D9DFEC",
    blue="#1F55E0", blue_soft="#5C83E8", gold="#B8860B", gold_soft="#D4A84B",
    text="#0B1220", muted="#586482",
    pass_c="#1F55E0", warn_c="#C98A0B", fail_c="#D03A3A",
    on_pass="#FFFFFF", on_warn="#FFFFFF", on_fail="#FFFFFF",
    grid="rgba(31,85,224,.08)", glow1="rgba(31,85,224,.12)", glow2="rgba(212,168,75,.20)",
    ring_gold="#B8860B", ring_blue="#1F55E0", ring_op=".8",
    scheme="light", dfilter="invert(1) hue-rotate(180deg)", letter="#A67A12",
)


def is_light() -> bool:
    return bool(st.session_state.get("fl_light", False))


def _palette():
    return LIGHT if is_light() else DARK


def _rings_svg(gold: str, blue: str) -> str:
    """Concentric lens rings + crosshair, the same motif as the logo."""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1000 1000' fill='none'>"
        f"<circle cx='500' cy='500' r='110' stroke='{gold}' stroke-opacity='.55' stroke-width='2'/>"
        f"<circle cx='500' cy='500' r='210' stroke='{blue}' stroke-opacity='.45' stroke-width='1.5' stroke-dasharray='3 14'/>"
        f"<circle cx='500' cy='500' r='320' stroke='{gold}' stroke-opacity='.32' stroke-width='1.5'/>"
        f"<circle cx='500' cy='500' r='410' stroke='{blue}' stroke-opacity='.30' stroke-width='1.5' stroke-dasharray='28 18'/>"
        f"<circle cx='500' cy='500' r='485' stroke='{gold}' stroke-opacity='.18' stroke-width='1'/>"
        f"<line x1='500' y1='10' x2='500' y2='990' stroke='{gold}' stroke-opacity='.22'/>"
        f"<line x1='10' y1='500' x2='990' y2='500' stroke='{blue}' stroke-opacity='.25'/>"
        f"<circle cx='500' cy='500' r='6' fill='{gold}' fill-opacity='.7'/>"
        "</svg>"
    )
    return "url(\"data:image/svg+xml;utf8," + quote(svg) + "\")"


CSS_T = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Manrope:wght@400;500;600&display=swap');

:root {
  --bg:${bg}; --panel:${panel}; --panel2:${panel2}; --line:${line};
  --blue:${blue}; --blue-soft:${blue_soft}; --gold:${gold}; --gold-soft:${gold_soft};
  --text:${text}; --muted:${muted};
  --pass:${pass_c}; --warn:${warn_c}; --fail:${fail_c};
  --on-pass:${on_pass}; --on-warn:${on_warn}; --on-fail:${on_fail};
  color-scheme:${scheme};
}

html, body, .stApp { font-family:'Manrope',sans-serif; background:var(--bg); color:var(--text); }

/* ---------- Background: grid + glow, rotating lens rings ---------- */
.stApp { position:relative; }
.stApp::before {
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(800px 500px at 6% -4%, ${glow1}, transparent 65%),
    radial-gradient(700px 500px at 100% 100%, ${glow2}, transparent 65%),
    linear-gradient(${grid} 1px, transparent 1px) 0 0 / 56px 56px,
    linear-gradient(90deg, ${grid} 1px, transparent 1px) 0 0 / 56px 56px;
  -webkit-mask-image:radial-gradient(ellipse 90% 80% at 60% 20%, #000 30%, transparent 100%);
          mask-image:radial-gradient(ellipse 90% 80% at 60% 20%, #000 30%, transparent 100%);
}
.stApp::after {
  content:""; position:fixed; z-index:0; pointer-events:none;
  width:1100px; height:1100px; top:-420px; right:-420px;
  background:${rings} center / contain no-repeat; opacity:${ring_op};
  animation:fl-spin 240s linear infinite;
}
[data-testid="stAppViewContainer"] { position:relative; z-index:1; background:transparent; }
[data-testid="stAppViewContainer"]::before {
  content:""; position:fixed; z-index:-1; pointer-events:none;
  width:760px; height:760px; bottom:-330px; left:-300px;
  background:${rings_b} center / contain no-repeat; opacity:${ring_op};
  animation:fl-spin 320s linear infinite reverse;
}
@keyframes fl-spin { to { transform:rotate(360deg); } }

header[data-testid="stHeader"] { background:transparent; color:var(--muted); }
header[data-testid="stHeader"] * { color:var(--muted); }
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:2.2rem; max-width:1150px; }

/* ---------- Text ---------- */
h1,h2,h3,h4 { font-family:'Bricolage Grotesque',sans-serif; letter-spacing:-.015em; color:var(--text); }
h2 { border-bottom:1px solid var(--line); padding-bottom:.4rem; }
.stApp p, .stApp li, .stApp label, .stApp span, [data-testid="stMarkdownContainer"] { color:var(--text); }
[data-testid="stCaptionContainer"], .stApp small { color:var(--muted); }
p, li, label { line-height:1.6; }
a { color:var(--blue-soft); }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }

/* ---------- Buttons: gold is the call-to-action ---------- */
.stButton > button, .stDownloadButton > button {
  background:var(--gold-soft); border:0; border-radius:8px; font-weight:600;
  padding:.55rem 1.2rem; transition:background .15s, transform .1s;
}
.stButton > button, .stButton > button *, .stDownloadButton > button, .stDownloadButton > button * { color:#0a0a0a !important; }
.stButton > button:hover, .stDownloadButton > button:hover { background:var(--gold); }
.stButton > button:active { transform:translateY(1px); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline:2px solid var(--blue-soft); outline-offset:2px; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap:1.4rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color:var(--muted); font-weight:600; background:transparent; }
.stTabs [data-baseweb="tab"] * { color:inherit; }
.stTabs [aria-selected="true"] { color:var(--gold); }
.stTabs [data-baseweb="tab-highlight"] { background:var(--gold); }

/* ---------- Inputs, menus, tables ---------- */
[data-testid="stFileUploader"] section { background:var(--panel); border:1px dashed var(--blue); border-radius:10px; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="textarea"], .stTextInput input, .stNumberInput input
  { background:var(--panel) !important; border-color:var(--line) !important; color:var(--text) !important; }
[data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"] { background:var(--panel) !important; color:var(--text) !important; }
li[role="option"] { color:var(--text) !important; }
li[role="option"]:hover, li[aria-selected="true"] { background:var(--panel2) !important; }
[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:10px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; filter:${dfilter}; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:.9rem 1rem; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { font-family:'Bricolage Grotesque',sans-serif; color:var(--gold) !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color:var(--muted) !important; }
[data-testid="stAlert"] { background:var(--panel); border:1px solid var(--line); border-radius:10px; }

/* ---------- Hero ---------- */
.fl-hero { display:flex; align-items:center; gap:1.1rem; margin:.2rem 0 1.4rem; }
.fl-hero svg { flex:none; }
.fl-title { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:3rem; line-height:1; margin:0; color:var(--text); }
.fl-title .fl-lens { color:var(--gold); }
.fl-tag { color:var(--muted) !important; font-size:1.05rem; margin:.45rem 0 0; max-width:52ch; }

/* SVG colour helpers (so SVGs follow the active mode) */
.s-gold { stroke:var(--gold); } .s-blue { stroke:var(--blue); } .s-bsoft { stroke:var(--blue-soft); } .s-line { stroke:var(--line); }
.f-gold { fill:var(--gold); } .f-blue { fill:var(--blue); }

/* ---------- Verdict cards ---------- */
.fl-card { background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--c); border-radius:10px; padding:1rem 1.1rem; height:100%; }
.fl-card .fl-name { color:var(--muted); font-size:.9rem; }
.fl-card .fl-val { font-family:'Bricolage Grotesque',sans-serif; font-size:2rem; font-weight:700; margin:.15rem 0; color:var(--text); }
.fl-card .fl-badge { display:inline-block; font-size:.8rem; font-weight:700; background:var(--c); border-radius:999px; padding:.1rem .65rem; color:var(--on); }
.fl-card .fl-note { color:var(--muted); font-size:.88rem; margin-top:.55rem; }

/* ---------- Grade card ---------- */
.fl-grade { display:flex; align-items:center; gap:1.4rem; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:1.2rem 1.5rem; }
.fl-grade .fl-ring-arc { stroke-dasharray:var(--len) 400; stroke-dashoffset:var(--len); animation:fl-draw 1.4s ease-out .2s forwards; }
.fl-grade .fl-letter { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:44px; fill:${letter}; }
.fl-grade .fl-gt { font-family:'Bricolage Grotesque',sans-serif; font-size:1.35rem; font-weight:700; color:var(--text); }
.fl-grade .fl-gs { color:var(--muted); margin-top:.3rem; max-width:46ch; }
@keyframes fl-draw { to { stroke-dashoffset:0; } }

/* ---------- Balance beam ---------- */
.fl-beam { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:1rem 1.2rem .6rem; }
.fl-beam text { font-family:'Manrope',sans-serif; fill:var(--muted); font-size:14px; }
.fl-beam text.fl-rate { font-family:'Bricolage Grotesque',sans-serif; font-size:26px; font-weight:700; fill:var(--text); }
.fl-beam .fl-cap { color:var(--muted); font-size:.88rem; text-align:center; margin:.1rem 0 .4rem; }

@media (prefers-reduced-motion: reduce) {
  .fl-grade .fl-ring-arc { animation:none; stroke-dashoffset:0; }
  .stApp::after, [data-testid="stAppViewContainer"]::before { animation:none; }
  .stButton > button { transition:none; }
}
@media (max-width:640px) {
  .fl-title { font-size:2.2rem; }
  .fl-grade { flex-direction:column; align-items:flex-start; }
  .stApp::after { opacity:.5; }
}
</style>
""")


def inject_theme():
    """Call once at the top of app.py. Reads the Light mode switch and styles the whole page."""
    p = dict(_palette())
    p["rings"] = _rings_svg(p["ring_gold"], p["ring_blue"])
    p["rings_b"] = _rings_svg(p["ring_blue"], p["ring_gold"])
    st.markdown(CSS_T.substitute(p), unsafe_allow_html=True)


def _html(s: str):
    # Strip indentation and blank lines, or Streamlit's markdown treats the block as code.
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


def hero(tagline="See which groups your model treats differently, then test fixes.", toggle=True):
    mark = (
        '<svg width="62" height="62" viewBox="0 0 62 62" aria-hidden="true">'
        '<circle class="s-blue" cx="31" cy="31" r="26" fill="none" stroke-width="3"/>'
        '<circle class="s-gold" cx="31" cy="31" r="15" fill="none" stroke-width="3"/>'
        '<line class="s-gold" x1="31" y1="5" x2="31" y2="57" stroke-width="2"/>'
        '<line class="s-bsoft" x1="9" y1="31" x2="53" y2="31" stroke-width="2"/>'
        '<circle class="f-gold" cx="31" cy="31" r="3.5"/></svg>'
    )
    left, right = st.columns([5, 1.3])
    with left:
        _html(
            f'<div class="fl-hero">{mark}<div>'
            f'<h1 class="fl-title">Fair<span class="fl-lens">Lens</span></h1>'
            f'<p class="fl-tag">{tagline}</p></div></div>'
        )
    if toggle:
        with right:
            st.write("")
            st.toggle("Light mode", key="fl_light")


def letter_grade(score: float) -> str:
    """score is 0-100. Adjust the cut-offs to match how you define fairness."""
    for cut, g in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= cut:
            return g
    return "F"


def grade_card(score: float, headline: str, detail: str = ""):
    """One overall fairness grade as a gold ring that draws itself once on load."""
    score = max(0, min(100, score))
    circ = 2 * math.pi * 54
    arc = circ * score / 100
    g = letter_grade(score)
    _html(
        f'<div class="fl-grade">'
        f'<svg width="140" height="140" viewBox="0 0 140 140" role="img" aria-label="Fairness grade {g}">'
        f'<circle class="s-line" cx="70" cy="70" r="54" fill="none" stroke-width="10"/>'
        f'<circle class="fl-ring-arc s-gold" style="--len:{arc:.1f}" cx="70" cy="70" r="54" fill="none" '
        f'stroke-width="10" stroke-linecap="round" transform="rotate(-90 70 70)"/>'
        f'<text class="fl-letter" x="70" y="86" text-anchor="middle">{g}</text></svg>'
        f'<div><div class="fl-gt">{headline}</div><div class="fl-gs">{detail}</div></div></div>'
    )


_STATUS = {
    "pass": ("var(--pass)", "var(--on-pass)", "✓ PASS"),
    "warn": ("var(--warn)", "var(--on-warn)", "! WARN"),
    "fail": ("var(--fail)", "var(--on-fail)", "✕ FAIL"),
}


def verdict_card(name: str, value: str, status: str, note: str = ""):
    """status: 'pass', 'warn' or 'fail'. The label always carries a symbol, so colour is never the only signal."""
    color, on, label = _STATUS[status]
    _html(
        f'<div class="fl-card" style="--c:{color};--on:{on}">'
        f'<div class="fl-name">{name}</div><div class="fl-val">{value}</div>'
        f'<span class="fl-badge">{label}</span><div class="fl-note">{note}</div></div>'
    )


def balance_beam(label_a: str, rate_a: float, label_b: str, rate_b: float):
    """Scales that tilt toward the group the model selects more often. Rates are 0-1."""
    angle = max(-14.0, min(14.0, -(rate_a - rate_b) * 30))  # negative = left side sinks
    th = math.radians(angle)
    px, py, half = 260, 60, 170
    lx, ly = px - half * math.cos(th), py - half * math.sin(th)
    rx, ry = px + half * math.cos(th), py + half * math.sin(th)

    def pan(x, y):
        return (
            f'<line class="s-bsoft" x1="{x:.1f}" y1="{y:.1f}" x2="{x-45:.1f}" y2="{y+75:.1f}" stroke-width="1.5"/>'
            f'<line class="s-bsoft" x1="{x:.1f}" y1="{y:.1f}" x2="{x+45:.1f}" y2="{y+75:.1f}" stroke-width="1.5"/>'
            f'<path class="f-blue" d="M{x-45:.1f} {y+75:.1f} Q{x:.1f} {y+108:.1f} {x+45:.1f} {y+75:.1f} Z"/>'
        )

    svg = (
        f'<svg viewBox="0 0 520 260" width="100%" role="img" '
        f'aria-label="{label_a} selected at {rate_a:.1%}, {label_b} at {rate_b:.1%}">'
        f'<line class="s-blue" x1="{px}" y1="{py}" x2="{px}" y2="225" stroke-width="4"/>'
        f'<rect class="f-blue" x="215" y="225" width="90" height="8" rx="3"/>'
        f'<line class="s-gold" x1="{lx:.1f}" y1="{ly:.1f}" x2="{rx:.1f}" y2="{ry:.1f}" stroke-width="5" stroke-linecap="round"/>'
        f'<circle class="f-gold" cx="{px}" cy="{py}" r="8"/>'
        f'{pan(lx, ly)}{pan(rx, ry)}'
        f'<text class="fl-rate" x="{lx:.1f}" y="{ly+145:.1f}" text-anchor="middle">{rate_a:.1%}</text>'
        f'<text x="{lx:.1f}" y="{ly+166:.1f}" text-anchor="middle">{label_a}</text>'
        f'<text class="fl-rate" x="{rx:.1f}" y="{ry+145:.1f}" text-anchor="middle">{rate_b:.1%}</text>'
        f'<text x="{rx:.1f}" y="{ry+166:.1f}" text-anchor="middle">{label_b}</text></svg>'
    )
    _html(
        f'<div class="fl-beam">{svg}'
        f'<div class="fl-cap">The beam tilts toward the group the model favours.</div></div>'
    )


def apply_mpl_theme():
    """Makes matplotlib charts match the active mode. Call it before plotting, on every run."""
    import matplotlib as mpl

    p = _palette()
    mpl.rcParams.update({
        "figure.facecolor": p["panel"], "axes.facecolor": p["panel"], "savefig.facecolor": p["panel"],
        "axes.edgecolor": p["line"], "axes.labelcolor": p["text"], "text.color": p["text"],
        "xtick.color": p["muted"], "ytick.color": p["muted"], "grid.color": p["line"],
        "axes.prop_cycle": mpl.cycler(color=[p["gold"], p["blue"], p["blue_soft"], p["gold_soft"], p["fail_c"]]),
        "axes.spines.top": False, "axes.spines.right": False,
    })
