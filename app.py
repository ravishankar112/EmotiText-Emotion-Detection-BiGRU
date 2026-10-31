import html
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EmotiText | Emotion detection from text",
    page_icon="🎭",
    layout="wide",
)

ARTIFACT_DIR = "Artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "BiGRU_Model.keras")
TOKENIZER_PATH = os.path.join(ARTIFACT_DIR, "tokenizer.pkl")

MAX_LEN = 50
MAX_BATCH_ROWS = 2000

LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]

MODEL_RESULTS = {
    "SimpleRNN": None,
    "LSTM": None,
    "GRU": None,
    "BiLSTM": None,
    "BiGRU": None,
}

ALGOS = [
    ("SimpleRNN", "Baseline recurrent model", False),
    ("LSTM", "Long-term memory with gates", False),
    ("GRU", "Lighter and faster than LSTM", False),
    ("BiLSTM", "LSTM reading both directions", False),
    ("BiGRU", "GRU reading both directions (final model)", True),
]

EMOTIONS = {
    "joy": {
        "emoji": "😄",
        "color": "#FFC21A",
        "on": "#12141F",
        "line": "Positive energy. This text sounds happy and upbeat.",
    },
    "sadness": {
        "emoji": "😢",
        "color": "#2F6BFF",
        "on": "#FFFFFF",
        "line": "Low mood. This text sounds sad or lonely.",
    },
    "anger": {
        "emoji": "😠",
        "color": "#E5384C",
        "on": "#FFFFFF",
        "line": "Strong frustration. This text sounds angry or irritated.",
    },
    "fear": {
        "emoji": "😨",
        "color": "#7C4DEB",
        "on": "#FFFFFF",
        "line": "Worry and unease. This text sounds scared or anxious.",
    },
    "love": {
        "emoji": "🥰",
        "color": "#E0468C",
        "on": "#FFFFFF",
        "line": "Warm feelings. This text sounds caring and affectionate.",
    },
    "surprise": {
        "emoji": "😲",
        "color": "#00897B",
        "on": "#FFFFFF",
        "line": "Unexpected event. This text sounds shocked or amazed.",
    },
}

LEGEND_ORDER = [
    "joy",
    "sadness",
    "love",
    "anger",
    "fear",
    "surprise",
]

EXAMPLES = {
    "😄 Happy": "I can't believe how happy I am right now, this is amazing!",
    "😢 Lonely": "I feel so alone and hopeless today.",
    "😠 Angry": "I am furious that they cancelled the trip at the last minute.",
    "😨 Scared": "I feel terrified when walking down dark alleyways alone.",
    "😲 Shocked": "I was shocked and completely surprised by the unexpected gift!",
    "🥰 Caring": "I feel so lucky to have you, I really care about you a lot.",
}


# STYLE

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --ink:#12141F;
    --muted:#5E6275;
    --line:#E3E6EF;
    --bg:#F8F9FC;
    --acc:#12141F;
    --on:#FFFFFF;
}

html, body, [class*="css"], .stApp {
    font-family: 'Manrope', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--bg);
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding-top: 2.4rem;
    padding-bottom: 3rem;
    max-width: 1060px;
}

/* Header */

.brand {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--ink);
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: .55rem;
}

.brand-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--acc);
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.5rem;
    line-height: 1.02;
    letter-spacing: -0.035em;
    margin: 0 0 1rem 0;
    color: var(--ink);
    max-width: 720px;
}

.hero-sub {
    color: var(--muted);
    font-size: 1.08rem;
    line-height: 1.7;
    max-width: 600px;
    margin: 0 0 1.3rem 0;
}

.legend {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
    margin-bottom: 1.8rem;
}

.lg {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .35rem .8rem .35rem .6rem;
    border-radius: 999px;
    background: #fff;
    border: 1px solid var(--line);
    font-size: .86rem;
    font-weight: 600;
    color: var(--ink);
    text-transform: capitalize;
}

.lg i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

.sec-h {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    color: var(--ink);
    margin: 1.7rem 0 .6rem 0;
    letter-spacing: -0.01em;
}

.muted {
    color: var(--muted);
    font-size: .98rem;
    line-height: 1.7;
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: .4rem;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    height: 46px;
    padding: 0 1.1rem;
    color: var(--muted);
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: var(--ink) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: var(--acc);
    height: 3px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.6rem;
}

.dev-card {
    background: var(--ink);
    border-radius: 20px;
    padding: 1.2rem;
}

.dev-avatar {
    width: 50px;
    height: 50px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.2rem;
    color: var(--ink);
    background: #fff;
    margin-bottom: .8rem;
}

.dev-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #fff;
}

.dev-role {
    color: #B9BDD3;
    font-size: .88rem;
    margin-top: .2rem;
    line-height: 1.5;
}

.dots {
    display: flex;
    gap: .35rem;
    margin-top: .9rem;
}

.dots i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

.side-h {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--ink);
    margin: 1.5rem 0 .6rem 0;
}

.chip {
    display: inline-block;
    padding: .28rem .7rem;
    margin: 0 .35rem .4rem 0;
    border-radius: 999px;
    background: var(--bg);
    border: 1px solid var(--line);
    color: var(--ink);
    font-size: .8rem;
    font-weight: 600;
}

.kv {
    display: flex;
    justify-content: space-between;
    gap: .6rem;
    padding: .45rem 0;
    border-bottom: 1px solid var(--line);
    font-size: .86rem;
}

.kv span:first-child {
    color: var(--muted);
}

.kv span:last-child {
    color: var(--ink);
    font-weight: 600;
    text-align: right;
}

.kv.final {
    background: var(--ink);
    border-radius: 10px;
    padding: .45rem .7rem;
    border-bottom: none;
}

.kv.final span:first-child,
.kv.final span:last-child {
    color: #fff;
    font-weight: 700;
}

/* Inputs */

.stTextArea textarea {
    background: #fff !important;
    border: 1.5px solid var(--line) !important;
    border-radius: 16px !important;
    color: var(--ink) !important;
    font-size: 1.06rem !important;
    padding: 1rem 1.1rem !important;
    line-height: 1.6 !important;
}

.stTextArea textarea:focus {
    border-color: var(--acc) !important;
    box-shadow: 0 0 0 3px rgba(18,20,31,.07) !important;
}

.stTextArea label {
    display: none;
}

div.stButton > button,
div.stDownloadButton > button {
    border-radius: 999px;
    border: 1.5px solid var(--line);
    background: #fff;
    color: var(--ink);
    padding: .4rem .9rem;
    font-weight: 600;
    transition: border-color .15s, background .15s;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    border-color: var(--ink);
    color: var(--ink);
    background: #fff;
}

div.stButton > button:focus-visible {
    outline: 2px solid var(--ink);
    outline-offset: 2px;
}

div.stButton > button[kind="primary"] {
    background: var(--acc);
    color: var(--on);
    border: none;
    border-radius: 14px;
    font-weight: 700;
    padding: .7rem 2rem;
    font-size: 1rem;
}

div.stButton > button[kind="primary"]:hover {
    background: var(--acc);
    color: var(--on);
    filter: brightness(.93);
}

/* Mood panel */

.mood {
    display: flex;
    align-items: center;
    gap: 1.8rem;
    border-radius: 28px;
    padding: 2rem 2.2rem;
    margin-top: 1.6rem;
    background: var(--c);
    color: var(--on);
}

.mood-emoji {
    width: 104px;
    height: 104px;
    border-radius: 30px;
    background: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.6rem;
    flex-shrink: 0;
    box-shadow: 0 10px 24px rgba(0,0,0,.18);
}

.mood-mid {
    flex: 1;
    min-width: 0;
}

.mood-word {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.4rem;
    line-height: 1;
    letter-spacing: -0.03em;
    text-transform: capitalize;
}

.mood-line {
    font-size: 1.05rem;
    margin-top: .7rem;
    opacity: .95;
    font-weight: 500;
}

.mood-quote {
    font-style: italic;
    margin-top: .9rem;
    font-size: .95rem;
    opacity: .85;
    padding-top: .8rem;
    border-top: 1px solid currentColor;
    border-top-color: rgba(127,127,127,.45);
}

.mood-warn {
    display: inline-block;
    margin-top: .8rem;
    padding: .3rem .8rem;
    border-radius: 999px;
    background: rgba(0,0,0,.16);
    font-size: .86rem;
    font-weight: 600;
}

.mood-right {
    text-align: right;
    flex-shrink: 0;
}

.mood-pct {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 4rem;
    line-height: 1;
    letter-spacing: -0.03em;
}

.mood-pct span {
    font-size: 1.8rem;
}

.mood-cap {
    font-size: .95rem;
    font-weight: 600;
    opacity: .9;
    margin-top: .2rem;
}

/* Probability bars */

.bars {
    margin-top: 1.2rem;
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1.3rem 1.6rem;
}

.bars-h {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--ink);
    margin-bottom: .9rem;
}

.bar-row {
    display: flex;
    align-items: center;
    gap: .8rem;
    margin: .7rem 0;
}

.bar-name {
    width: 118px;
    color: var(--ink);
    font-size: .95rem;
    text-transform: capitalize;
    font-weight: 600;
}

.bar-track {
    flex: 1;
    height: 12px;
    background: #EEF0F7;
    border-radius: 999px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
}

.bar-pct {
    width: 58px;
    text-align: right;
    color: var(--ink);
    font-size: .92rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}

/* History */

.hist-h {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--ink);
    margin: 2rem 0 .6rem 0;
}

.hist-row {
    display: flex;
    align-items: center;
    gap: .8rem;
    padding: .7rem 0;
    border-bottom: 1px solid var(--line);
}

.hist-pill {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    padding: .18rem .75rem .18rem .55rem;
    border-radius: 999px;
    background: #fff;
    border: 1px solid var(--line);
    font-size: .82rem;
    font-weight: 700;
    color: var(--ink);
    text-transform: capitalize;
}

.hist-pill i {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}

.hist-text {
    color: var(--ink);
    font-size: .93rem;
    flex: 1;
}

.hist-pct {
    color: var(--muted);
    font-size: .88rem;
    font-weight: 600;
}

/* About tab */

.card {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1.3rem 1.4rem;
    height: 100%;
}

.card-h {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--ink);
    margin-bottom: .45rem;
}

.algo-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: .85rem 1.1rem;
    border-radius: 14px;
    margin-bottom: .5rem;
    background: #fff;
    border: 1px solid var(--line);
}

.algo-row.final {
    background: var(--ink);
    border-color: var(--ink);
}

.algo-row.final .algo-name,
.algo-row.final .algo-desc,
.algo-row.final .algo-acc {
    color: #fff;
}

.algo-name {
    width: 100px;
    font-weight: 800;
    color: var(--ink);
}

.algo-desc {
    flex: 1;
    color: var(--muted);
    font-size: .93rem;
}

.algo-acc {
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    font-weight: 700;
}

.flow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: .5rem;
    margin-top: .4rem;
}

.flow-step {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: .5rem .85rem;
    color: var(--ink);
    font-size: .9rem;
    font-weight: 600;
}

.flow-sep {
    color: var(--muted);
    font-weight: 700;
}

.footer {
    margin-top: 3rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: .9rem;
}

.footer b {
    color: var(--ink);
}

@media (max-width: 760px) {

    .hero-title {
        font-size: 2.4rem;
    }

    .mood {
        flex-direction: column;
        align-items: flex-start;
        padding: 1.5rem;
        gap: 1.1rem;
    }

    .mood-right {
        text-align: left;
    }

    .mood-word {
        font-size: 2.6rem;
    }

    .mood-pct {
        font-size: 3rem;
    }

    .bar-name {
        width: 88px;
    }

    .algo-row {
        flex-wrap: wrap;
    }

    .algo-name {
        width: 100%;
    }
}

@media (prefers-reduced-motion: reduce) {
    * {
        transition: none !important;
    }
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def h(markup: str) -> str:
    """Flatten HTML so Markdown does not treat indented lines as code blocks."""
    return "".join(line.strip() for line in markup.splitlines())


def accent_css(label) -> str:
    """Switch the app accent colour to the detected emotion."""
    if not label:
        return ""

    m = EMOTIONS[label]

    return f"<style>:root{{--acc:{m['color']};--on:{m['on']};}}</style>"


# MODEL
def files_exist() -> bool:
    return (
        os.path.exists(MODEL_PATH)
        and os.path.exists(TOKENIZER_PATH)
    )


@st.cache_resource(show_spinner="Loading BiGRU model...")
def load_artifacts():

    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


def encode(texts, tokenizer) -> np.ndarray:
    """Tokenize + pad (padding='post', truncating='post') exactly like training."""

    seqs = tokenizer.texts_to_sequences(list(texts))

    arr = np.zeros(
        (len(seqs), MAX_LEN),
        dtype="int32"
    )

    for i, s in enumerate(seqs):

        s = s[:MAX_LEN]
        arr[i, :len(s)] = s

    return arr


def predict_probs(texts, model, tokenizer) -> np.ndarray:

    return model.predict(
        encode(texts, tokenizer),
        batch_size=256,
        verbose=0
    )


def bars_html(probs, title="Score for each emotion") -> str:

    order = np.argsort(probs)[::-1]
    rows = ""

    for rank, i in enumerate(order):

        m = EMOTIONS[LABELS[i]]
        p = float(probs[i]) * 100
        opacity = 1 if rank == 0 else 0.4

        rows += (
            f'<div class="bar-row">'
            f'<div class="bar-name">{m["emoji"]} {LABELS[i]}</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill" '
            f'style="width:{p:.1f}%;background:{m["color"]};opacity:{opacity}">'
            f'</div></div>'
            f'<div class="bar-pct">{p:.1f}%</div>'
            f'</div>'
        )

    return (
        f'<div class="bars">'
        f'<div class="bars-h">{title}</div>'
        f'{rows}'
        f'</div>'
    )


#STATE 

for key, default in (
    ("user_text", ""),
    ("history", []),
    ("result", None),
):

    if key not in st.session_state:
        st.session_state[key] = default


_res0 = st.session_state.result

st.markdown(
    accent_css(LABELS[_res0["top"]] if _res0 else None),
    unsafe_allow_html=True
)


# SIDEBAR

with st.sidebar:

    dots = "".join(
        f'<i style="background:{EMOTIONS[l]["color"]}"></i>'
        for l in LEGEND_ORDER
    )

    st.markdown(
        h(f"""
        <div class="dev-card">

            <div class="dev-avatar">RK</div>

            <div class="dev-name">
                Ravi Shankar Kumar
            </div>

            <div class="dev-role">
                Developer and designer<br>
                Data Science and ML
            </div>

            <div class="dots">
                {dots}
            </div>

        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-h">Built with</div>',
        unsafe_allow_html=True
    )

    stack = [
        "Python",
        "TensorFlow",
        "Keras",
        "Streamlit",
        "NumPy",
        "Pandas",
        "Scikit-learn",
        "Seaborn",
        "Matplotlib",
        "Hugging Face Datasets",
    ]

    st.markdown(
        "".join(f'<span class="chip">{s}</span>' for s in stack),
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="side-h">Algorithms compared</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "".join(
            f'<div class="kv{" final" if fin else ""}">'
            f'<span>{n}</span>'
            f'<span>{"Final model" if fin else "Compared"}</span>'
            f'</div>'
            for n, _d, fin in ALGOS
        ),
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="side-h">Final model (BiGRU)</div>',
        unsafe_allow_html=True
    )

    info = [
        ("Architecture", "2 x Bidirectional GRU"),
        ("Embedding", "300 dimensions"),
        ("Vocabulary", "10,000 words"),
        ("Max length", "50 tokens"),
        ("Dataset", "dair-ai/emotion"),
        ("Classes", "6 emotions"),
        ("Optimizer", "Adam"),
        ("Loss", "Sparse categorical CE"),
        ("Imbalance fix", "Class weights"),
        ("Training", "Early stopping"),
        ("Evaluation", "Confusion matrix"),
    ]

    st.markdown(
        "".join(
            f'<div class="kv"><span>{k}</span><span>{v}</span></div>'
            for k, v in info
        ),
        unsafe_allow_html=True
    )


# HEADER 

legend = "".join(
    f'<span class="lg">'
    f'<i style="background:{EMOTIONS[l]["color"]}"></i>'
    f'{EMOTIONS[l]["emoji"]} {l}'
    f'</span>'
    for l in LEGEND_ORDER
)

st.markdown(
    f'<div class="brand">'
    f'<span class="brand-dot"></span>EmotiText</div>'

    f'<div class="hero-title">'
    f'What is this text feeling?</div>'

    f'<p class="hero-sub">'
    f'Type a sentence or upload a CSV. A BiGRU deep learning model reads the text and tells you '
    f'which of six emotions it expresses, with a confidence score for each.</p>'

    f'<div class="legend">{legend}</div>',
    unsafe_allow_html=True,
)


def set_example(txt: str):
    st.session_state.user_text = txt


tab_detect, tab_batch, tab_about = st.tabs(
    ["Detect", "Batch (CSV)", "About project"]
)


#  TAB 1: DETECT 

with tab_detect:

    st.text_area(
        "Your text",
        key="user_text",
        height=130,
        placeholder="Example: I was so nervous before the interview, my hands were shaking.",
    )

    st.caption("Try an example")

    cols = st.columns(len(EXAMPLES))

    for col, (name, txt) in zip(cols, EXAMPLES.items()):

        col.button(
            name,
            key=f"ex_{name}",
            on_click=set_example,
            args=(txt,),
            use_container_width=True
        )

    if st.button("Detect emotion", type="primary"):

        text = st.session_state.user_text.strip()

        if not text:

            st.warning(
                "Write a sentence first, then press Detect emotion."
            )

        elif not files_exist():

            st.error(
                f"Model files not found. Put `BiGRU_Model.keras` and `tokenizer.pkl` inside the "
                f"`{ARTIFACT_DIR}/` folder next to app.py."
            )

        else:

            try:

                model, tokenizer = load_artifacts()

                probs = predict_probs(
                    [text],
                    model,
                    tokenizer
                )[0]

                top = int(np.argmax(probs))

                st.session_state.result = {
                    "text": text,
                    "probs": probs.tolist(),
                    "top": top
                }

                st.session_state.history.insert(
                    0,
                    {
                        "text": text,
                        "label": LABELS[top],
                        "conf": float(probs[top])
                    }
                )

                st.session_state.history = (
                    st.session_state.history[:6]
                )

            except Exception as e:  # noqa: BLE001

                st.error(
                    "Could not load the model. Make sure the TensorFlow version matches the one used for "
                    f"training. Details: {e}"
                )

    res = st.session_state.result

    if res:

        probs = res["probs"]
        label = LABELS[res["top"]]
        meta = EMOTIONS[label]
        conf = probs[res["top"]] * 100

        st.markdown(
            accent_css(label),
            unsafe_allow_html=True
        )

        warn = ""

        if conf < 50:

            warn = (
                '<div class="mood-warn">'
                'Not very sure. A longer, clearer sentence usually helps.'
                '</div>'
            )

        st.markdown(
            h(f"""
            <div class="mood" style="--c:{meta['color']};--on:{meta['on']}">

                <div class="mood-emoji">
                    {meta['emoji']}
                </div>

                <div class="mood-mid">

                    <div class="mood-word">
                        {label}
                    </div>

                    <div class="mood-line">
                        {meta['line']}
                    </div>

                    {warn}

                    <div class="mood-quote">
                        "{html.escape(res['text'])}"
                    </div>

                </div>

                <div class="mood-right">

                    <div class="mood-pct">
                        {conf:.0f}<span>%</span>
                    </div>

                    <div class="mood-cap">
                        confidence
                    </div>

                </div>

            </div>
            """),
            unsafe_allow_html=True,
        )

        st.markdown(
            bars_html(probs),
            unsafe_allow_html=True
        )

    if st.session_state.history:

        st.markdown(
            '<div class="hist-h">Recent checks</div>',
            unsafe_allow_html=True
        )

        rows = ""

        for item in st.session_state.history:

            c = EMOTIONS[item["label"]]["color"]

            short = html.escape(
                item["text"][:90]
                + ("..." if len(item["text"]) > 90 else "")
            )

            rows += (
                f'<div class="hist-row">'
                f'<span class="hist-pill">'
                f'<i style="background:{c}"></i>'
                f'{item["label"]}</span>'
                f'<span class="hist-text">{short}</span>'
                f'<span class="hist-pct">{item["conf"]*100:.0f}%</span>'
                f'</div>'
            )

        st.markdown(
            rows,
            unsafe_allow_html=True
        )


# TAB 2: BATCH 

with tab_batch:

    st.markdown(
        '<div class="sec-h">Check many texts at once</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="muted">Upload a CSV file with a column of text (reviews, tweets, messages). '
        f'EmotiText predicts the emotion for each row, shows the overall mix and lets you download the results. '
        f'Up to {MAX_BATCH_ROWS} rows are processed.</p>',
        unsafe_allow_html=True,
    )

    up = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed"
    )

    if up is not None:

        try:

            df = pd.read_csv(up)

        except Exception as e:  # noqa: BLE001

            df = None

            st.error(
                f"Could not read this file as CSV. Details: {e}"
            )

        if df is not None:

            if df.empty:

                st.warning("This CSV has no rows.")

            else:

                obj_cols = [
                    c for c in df.columns
                    if df[c].dtype == object
                    or str(df[c].dtype).startswith("str")
                ]

                text_cols = obj_cols or list(df.columns)

                col_name = st.selectbox(
                    "Which column has the text?",
                    text_cols
                )

                if st.button("Analyze file", type="primary"):

                    if not files_exist():

                        st.error(
                            f"Model files not found in the `{ARTIFACT_DIR}/` folder."
                        )

                    else:

                        try:

                            model, tokenizer = load_artifacts()

                            sub = df.head(MAX_BATCH_ROWS).copy()

                            texts = (
                                sub[col_name]
                                .fillna("")
                                .astype(str)
                                .tolist()
                            )

                            with st.spinner(
                                f"Analyzing {len(texts)} rows..."
                            ):

                                P = predict_probs(
                                    texts,
                                    model,
                                    tokenizer
                                )

                            sub["emotion"] = [
                                LABELS[i]
                                for i in P.argmax(axis=1)
                            ]

                            sub["confidence_%"] = (
                                P.max(axis=1) * 100
                            ).round(1)

                            st.session_state.batch = sub

                            if len(df) > MAX_BATCH_ROWS:

                                st.info(
                                    f"Only the first {MAX_BATCH_ROWS} of {len(df)} rows were analyzed."
                                )

                        except Exception as e:  # noqa: BLE001

                            st.error(
                                f"Could not run the model. Details: {e}"
                            )

    out = st.session_state.get("batch")

    if out is not None:

        counts = out["emotion"].value_counts()

        mix = np.array(
            [counts.get(l, 0) for l in LABELS],
            dtype=float
        )

        mix = mix / mix.sum()

        st.markdown(
            bars_html(
                mix,
                title=f"Emotion mix across {len(out)} rows"
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sec-h">Results</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            out,
            use_container_width=True,
            height=320
        )

        st.download_button(
            "Download results as CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="emotitext_results.csv",
            mime="text/csv",
        )


#  TAB 3: ABOUT 

with tab_about:

    st.markdown(
        '<div class="sec-h">Algorithms compared</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="muted">Five recurrent neural networks were trained on the same data and compared. '
        'BiGRU was kept as the final model and powers this app.</p>',
        unsafe_allow_html=True,
    )

    rows = ""

    for name, desc, fin in ALGOS:

        acc = MODEL_RESULTS.get(name)

        acc_txt = (
            f"{acc:.1f}%"
            if isinstance(acc, (int, float))
            else "-"
        )

        rows += (
            f'<div class="algo-row{" final" if fin else ""}">'
            f'<div class="algo-name">{name}</div>'
            f'<div class="algo-desc">{desc}</div>'
            f'<div class="algo-acc">{acc_txt}</div>'
            f'</div>'
        )

    st.markdown(
        rows,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sec-h">How it works</div>',
        unsafe_allow_html=True
    )

    steps = [
        "Raw text",
        "Tokenizer (10,000 words)",
        "Pad to 50 tokens",
        "Embedding (300-dim)",
        "BiGRU 128",
        "BiGRU 64",
        "Softmax (6 classes)",
        "Emotion"
    ]

    st.markdown(
        '<div class="flow">'
        + '<span class="flow-sep">&gt;</span>'.join(
            f'<span class="flow-step">{s}</span>'
            for s in steps
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sec-h">Project details</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        h("""
        <div class="card">

            <div class="card-h">
                Data
            </div>

            <div class="muted">
                dair-ai/emotion from Hugging Face. Short English sentences labelled with sadness, joy, love, anger, fear
                or surprise. Class counts are uneven, so class weights were used.
            </div>

        </div>
        """),
        unsafe_allow_html=True,
    )

    c2.markdown(
        h("""
        <div class="card">

            <div class="card-h">
                Training
            </div>

            <div class="muted">
                Adam optimizer, sparse categorical crossentropy, dropout 0.5 and early stopping on validation loss.
                Results were checked with a confusion matrix.
            </div>

        </div>
        """),
        unsafe_allow_html=True,
    )

    c3.markdown(
        h("""
        <div class="card">

            <div class="card-h">
                Limits
            </div>

            <div class="muted">
                Works best on short English sentences. Mixed or neutral text is forced into the nearest of the six
                emotions, so confidence can be low.
            </div>

        </div>
        """),
        unsafe_allow_html=True,
    )


# FOOTER 

st.markdown(
    '<div class="footer">Designed and developed by <b>Ravi Shankar Kumar</b>, Data Science and ML. '
    'Built with TensorFlow and Streamlit.</div>',
    unsafe_allow_html=True,
)