"""FairLens visual theme: black base, blue structure, gold for what matters.

Usage in app.py (call once, right after st.set_page_config):

    from theme import inject_theme, hero, grade_card, verdict_card, balance_beam
    inject_theme()
    hero()
"""
import math
import streamlit as st

BLACK = "#000000"
NAVY = "#0A1224"
NAVY_2 = "#0E1A33"
LINE = "#1B2A4A"
BLUE = "#2F6BFF"
BLUE_SOFT = "#7FA8FF"
GOLD = "#D4A84B"
GOLD_SOFT = "#F0D58A"
TEXT = "#E8ECF5"
MUTED = "#8A97B3"
PASS_C = "#4DA3FF"
WARN_C = "#E3B341"
FAIL_C = "#FF6B6B"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Manrope:wght@400;500;600&display=swap');

:root {{
  --bg:{BLACK}; --panel:{NAVY}; --panel2:{NAVY_2}; --line:{LINE};
  --blue:{BLUE}; --blue-soft:{BLUE_SOFT}; --gold:{GOLD}; --gold-soft:{GOLD_SOFT};
  --text:{TEXT}; --muted:{MUTED};
  --pass:{PASS_C}; --warn:{WARN_C}; --fail:{FAIL_C};
}}

html, body, [class*="css"], .stApp {{ font-family:'Manrope',sans-serif; color:var(--text); }}
.stApp {{
  background:
    radial-gradient(900px 420px at 8% -8%, rgba(47,107,255,.16), transparent 60%),
    var(--bg);
}}
header[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:2.2rem; max-width:1150px; }}

h1,h2,h3,h4 {{ font-family:'Bricolage Grotesque',sans-serif; letter-spacing:-.015em; color:var(--text); }}
h2 {{ border-bottom:1px solid var(--line); padding-bottom:.4rem; }}
p, li, label, .stMarkdown {{ line-height:1.6; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background:var(--panel); border-right:1px solid var(--line); }}

/* Buttons: gold is the single call-to-action colour */
.stButton > button, .stDownloadButton > button {{
  background:var(--gold); color:#0a0a0a; border:0; border-radius:8px;
  font-weight:600; padding:.55rem 1.2rem; transition:background .15s, transform .1s;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ background:var(--gold-soft); color:#000; }}
.stButton > button:active {{ transform:translateY(1px); }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{ outline:2px solid var(--blue-soft); outline-offset:2px; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:1.4rem; border-bottom:1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ color:var(--muted); font-weight:600; background:transparent; }}
.stTabs [aria-selected="true"] {{ color:var(--gold); }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--gold); }}

/* Inputs, uploader, expander, tables */
[data-testid="stFileUploader"] section {{ background:var(--panel); border:1px dashed var(--blue); border-radius:10px; }}
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{ background:var(--panel); border-color:var(--line); }}
.streamlit-expanderHeader, [data-testid="stExpander"] {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
[data-testid="stMetric"] {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:.9rem 1rem; }}
[data-testid="stMetricValue"] {{ font-family:'Bricolage Grotesque',sans-serif; color:var(--gold); }}
[data-testid="stMetricLabel"] {{ color:var(--muted); }}

/* Hero */
.fl-hero {{ display:flex; align-items:center; gap:1.1rem; margin:.2rem 0 1.6rem; }}
.fl-hero svg {{ flex:none; }}
.fl-title {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:3rem; line-height:1; margin:0; }}
.fl-title .fl-fair {{ color:var(--text); }}
.fl-title .fl-lens {{ color:var(--gold); }}
.fl-tag {{ color:var(--muted); font-size:1.05rem; margin:.45rem 0 0; max-width:52ch; }}

/* Verdict cards */
.fl-card {{ background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--c); border-radius:10px; padding:1rem 1.1rem; height:100%; }}
.fl-card .fl-name {{ color:var(--muted); font-size:.9rem; }}
.fl-card .fl-val {{ font-family:'Bricolage Grotesque',sans-serif; font-size:2rem; font-weight:700; margin:.15rem 0; }}
.fl-card .fl-badge {{ display:inline-block; font-size:.8rem; font-weight:700; color:#000; background:var(--c); border-radius:999px; padding:.1rem .65rem; }}
.fl-card .fl-note {{ color:var(--muted); font-size:.88rem; margin-top:.55rem; }}

/* Grade card */
.fl-grade {{ display:flex; align-items:center; gap:1.4rem; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:1.2rem 1.5rem; }}
.fl-grade .fl-ring-arc {{ stroke-dasharray:var(--len) 400; stroke-dashoffset:var(--len); animation:fl-draw 1.4s ease-out .2s forwards; }}
.fl-grade .fl-letter {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:44px; fill:var(--gold); }}
.fl-grade .fl-gt {{ font-family:'Bricolage Grotesque',sans-serif; font-size:1.35rem; font-weight:700; }}
.fl-grade .fl-gs {{ color:var(--muted); margin-top:.3rem; max-width:46ch; }}
@keyframes fl-draw {{ to {{ stroke-dashoffset:0; }} }}

/* Balance beam */
.fl-beam {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:1rem 1.2rem .6rem; }}
.fl-beam text {{ font-family:'Manrope',sans-serif; fill:var(--muted); font-size:14px; }}
.fl-beam text.fl-rate {{ font-family:'Bricolage Grotesque',sans-serif; font-size:26px; font-weight:700; fill:var(--text); }}
.fl-beam .fl-cap {{ color:var(--muted); font-size:.88rem; text-align:center; margin:.1rem 0 .4rem; }}

@media (prefers-reduced-motion: reduce) {{
  .fl-grade .fl-ring-arc {{ animation:none; stroke-dashoffset:0; }}
  .stButton > button {{ transition:none; }}
}}
@media (max-width:640px) {{
  .fl-title {{ font-size:2.2rem; }}
  .fl-grade {{ flex-direction:column; align-items:flex-start; }}
}}
</style>
"""


def inject_theme():
    """Call once at the top of app.py."""
    st.markdown(CSS, unsafe_allow_html=True)


def _html(s: str):
    # Strip leading spaces and blank lines, or Streamlit's markdown treats the block as code.
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


def hero(tagline="See which groups your model treats differently, then test fixes."):
    mark = (
        f'<svg width="62" height="62" viewBox="0 0 62 62" aria-hidden="true">'
        f'<circle cx="31" cy="31" r="26" fill="none" stroke="{BLUE}" stroke-width="3"/>'
        f'<circle cx="31" cy="31" r="15" fill="none" stroke="{GOLD}" stroke-width="3"/>'
        f'<line x1="31" y1="5" x2="31" y2="57" stroke="{GOLD}" stroke-width="2"/>'
        f'<line x1="9" y1="31" x2="53" y2="31" stroke="{BLUE_SOFT}" stroke-width="2"/>'
        f'<circle cx="31" cy="31" r="3.5" fill="{GOLD}"/></svg>'
    )
    _html(
        f'<div class="fl-hero">{mark}<div>'
        f'<h1 class="fl-title"><span class="fl-fair">Fair</span><span class="fl-lens">Lens</span></h1>'
        f'<p class="fl-tag">{tagline}</p></div></div>'
    )


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
    _html(
        f'<div class="fl-grade">'
        f'<svg width="140" height="140" viewBox="0 0 140 140" role="img" aria-label="Fairness grade {letter_grade(score)}">'
        f'<circle cx="70" cy="70" r="54" fill="none" stroke="{LINE}" stroke-width="10"/>'
        f'<circle class="fl-ring-arc" style="--len:{arc:.1f}" cx="70" cy="70" r="54" fill="none" '
        f'stroke="{GOLD}" stroke-width="10" stroke-linecap="round" transform="rotate(-90 70 70)"/>'
        f'<text class="fl-letter" x="70" y="86" text-anchor="middle">{letter_grade(score)}</text></svg>'
        f'<div><div class="fl-gt">{headline}</div><div class="fl-gs">{detail}</div></div></div>'
    )


_STATUS = {
    "pass": (PASS_C, "✓ PASS"),
    "warn": (WARN_C, "! WARN"),
    "fail": (FAIL_C, "✕ FAIL"),
}


def verdict_card(name: str, value: str, status: str, note: str = ""):
    """status: 'pass', 'warn' or 'fail'. The label always carries a symbol, so colour is never the only signal."""
    color, label = _STATUS[status]
    _html(
        f'<div class="fl-card" style="--c:{color}">'
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
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x-45:.1f}" y2="{y+75:.1f}" stroke="{BLUE_SOFT}" stroke-width="1.5"/>'
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x+45:.1f}" y2="{y+75:.1f}" stroke="{BLUE_SOFT}" stroke-width="1.5"/>'
            f'<path d="M{x-45:.1f} {y+75:.1f} Q{x:.1f} {y+108:.1f} {x+45:.1f} {y+75:.1f} Z" fill="{BLUE}"/>'
        )

    svg = (
        f'<svg viewBox="0 0 520 260" width="100%" role="img" '
        f'aria-label="{label_a} selected at {rate_a:.1%}, {label_b} at {rate_b:.1%}">'
        f'<line x1="{px}" y1="{py}" x2="{px}" y2="225" stroke="{BLUE}" stroke-width="4"/>'
        f'<rect x="215" y="225" width="90" height="8" rx="3" fill="{BLUE}"/>'
        f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{rx:.1f}" y2="{ry:.1f}" stroke="{GOLD}" stroke-width="5" stroke-linecap="round"/>'
        f'<circle cx="{px}" cy="{py}" r="8" fill="{GOLD}"/>'
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
    """Makes matplotlib charts match the dashboard. Call once before plotting."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": NAVY, "axes.facecolor": NAVY, "savefig.facecolor": NAVY,
        "axes.edgecolor": LINE, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": LINE,
        "axes.prop_cycle": mpl.cycler(color=[GOLD, BLUE, BLUE_SOFT, GOLD_SOFT, FAIL_C]),
        "axes.spines.top": False, "axes.spines.right": False,
    })
