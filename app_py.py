# app.py
# ============================================================
# Fake News Detection (RoBERTa) - Single-file Streamlit App
# Navigation: Intro -> Predict -> Result (via Next / Back buttons)
# Model source: Hugging Face Hub  (fmfahim6/fake-news-roberta)
# XAI: Integrated Gradients (Captum) + token bar chart + attribution line plot
# ============================================================

import os
import base64
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients


# =========================
# App Config
# =========================
st.set_page_config(page_title="Fake News Detection", layout="wide")

MODEL_ID = "fmfahim6/fake-news-roberta"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256

ASSETS_DIR = "assets"
BG_INTRO = os.path.join(ASSETS_DIR, "bg_intro.png")    # optional
BG_PRED  = os.path.join(ASSETS_DIR, "bg_input.png")    # optional
BG_RES   = os.path.join(ASSETS_DIR, "bg_result.png")   # optional


# =========================
# Session State
# =========================
if "page" not in st.session_state:
    st.session_state.page = "intro"  # intro | predict | result

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "pred_label" not in st.session_state:
    st.session_state.pred_label = None  # 0/1

if "probs" not in st.session_state:
    st.session_state.probs = None  # {"FAKE": float, "REAL": float}

if "xai" not in st.session_state:
    st.session_state.xai = None  # dict with tokens + attributions


# =========================
# UI Styling
# =========================
def _set_bg(image_path: str) -> None:
    """Sets a full-page background. If image missing, uses a dark gradient."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        bg_css = f"""
        <style>
          .stApp {{
            background:
              radial-gradient(1100px 500px at 18% 0%, rgba(56,189,248,0.18), transparent 60%),
              radial-gradient(900px 500px at 82% 8%, rgba(168,85,247,0.14), transparent 60%),
              linear-gradient(rgba(0,0,0,0.64), rgba(0,0,0,0.72)),
              url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
          }}
        </style>
        """
    else:
        bg_css = """
        <style>
          .stApp {
            background:
              radial-gradient(1100px 500px at 18% 0%, rgba(56,189,248,0.18), transparent 60%),
              radial-gradient(900px 500px at 82% 8%, rgba(168,85,247,0.14), transparent 60%),
              #0b1220;
          }
        </style>
        """
    st.markdown(bg_css, unsafe_allow_html=True)


def _inject_css() -> None:
    """A different styling approach than previous versions: neon border + layout bar."""
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }

          /* Top header bar */
          .topbar {
            width: 100%;
            background: rgba(4, 10, 20, 0.55);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 14px 18px;
            box-shadow: 0 18px 60px rgba(0,0,0,0.45);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            display:flex;
            align-items:center;
            justify-content:space-between;
            margin-bottom: 18px;
          }

          .brand {
            font-size: 18px;
            font-weight: 850;
            letter-spacing: 0.2px;
            color: rgba(248,250,252,0.95);
          }

          .tag {
            font-size: 12px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(56,189,248,0.35);
            background: rgba(56,189,248,0.12);
            color: rgba(248,250,252,0.92);
          }

          /* Main cards */
          .card {
            background: rgba(7, 14, 26, 0.66);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 20px 70px rgba(0,0,0,0.50);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            color: rgba(248,250,252,0.95);
          }

          .card-neon {
            border: 1px solid rgba(56,189,248,0.22);
            position: relative;
          }
          .card-neon:before {
            content: "";
            position: absolute;
            inset: -1px;
            border-radius: 18px;
            padding: 1px;
            background: linear-gradient(120deg, rgba(56,189,248,0.42), rgba(168,85,247,0.26), rgba(34,197,94,0.20));
            -webkit-mask:
              linear-gradient(#000 0 0) content-box,
              linear-gradient(#000 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
          }

          .h1 {
            font-size: 44px;
            font-weight: 900;
            margin: 0 0 10px 0;
            color: rgba(248,250,252,0.98);
            text-shadow: 0 2px 18px rgba(0,0,0,0.55);
          }

          .p {
            font-size: 15.5px;
            line-height: 1.78;
            color: rgba(248,250,252,0.90);
            margin: 0;
          }

          .muted {
            font-size: 13px;
            line-height: 1.6;
            color: rgba(248,250,252,0.74);
          }

          .chip {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.06);
            color: rgba(248,250,252,0.92);
            font-size: 12px;
            margin-right: 8px;
            margin-top: 8px;
          }

          .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 10px;
          }

          .metric {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px;
            padding: 14px;
          }
          .metric .k { font-size: 12px; color: rgba(248,250,252,0.78); }
          .metric .v { font-size: 26px; font-weight: 900; margin-top: 6px; }

          /* Token highlight */
          .tokwrap {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 14px;
            line-height: 2.1;
            font-size: 14px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _topbar() -> None:
    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">Fake News Detection</div>
          <div class="tag">Model: {MODEL_ID}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Model + Inference
# =========================
@st.cache_resource
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def _predict(text: str) -> Tuple[int, float, float]:
    tokenizer, model = _load_model()
    text = "" if text is None else str(text).strip()

    enc = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred = int(np.argmax(probs))  # 0=FAKE, 1=REAL
    return pred, float(probs[0]), float(probs[1])


# =========================
# XAI: Integrated Gradients
# =========================
def _forward_logits(input_ids, attention_mask):
    _, model = _load_model()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits


@st.cache_resource
def _lig():
    tokenizer, model = _load_model()
    return LayerIntegratedGradients(_forward_logits, model.roberta.embeddings)


def _compute_ig(text: str, target_class: int, n_steps: int = 24) -> dict:
    """
    Returns:
      {
        "tokens": [...],
        "scores": [...],   # normalized token attributions (signed)
      }
    """
    tokenizer, _ = _load_model()
    lig = _lig()

    text = "" if text is None else str(text).strip()

    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE).long()
    attention_mask = enc["attention_mask"].to(DEVICE).long()

    pad_id = tokenizer.pad_token_id
    baseline_ids = torch.full_like(input_ids, pad_id).to(DEVICE).long()

    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=int(target_class),
        n_steps=int(n_steps),
    )

    # (1, seq, hidden) -> (seq,)
    token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())

    # remove padding tokens influence by masking
    mask = attention_mask.squeeze(0).detach().cpu().numpy().astype(bool)
    tokens = [t for t, m in zip(tokens, mask) if m]
    token_attr = token_attr[mask]

    # normalize (signed)
    denom = np.max(np.abs(token_attr)) if np.max(np.abs(token_attr)) > 0 else 1.0
    scores = (token_attr / denom).astype(np.float32)

    return {"tokens": tokens, "scores": scores.tolist()}


def _html_token_highlight(tokens: List[str], scores: List[float], top_k: int = 40) -> str:
    """
    Creates an HTML block where important tokens are highlighted.
    Positive scores push toward predicted class; negative push away.
    """
    # Exclude RoBERTa special tokens from highlighting logic
    ignore = {"<s>", "</s>", "<pad>"}

    # choose tokens with largest absolute values for stronger highlighting
    pairs = [(i, t, float(s)) for i, (t, s) in enumerate(zip(tokens, scores)) if t not in ignore]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:top_k]
    keep_idx = set([i for i, _, _ in pairs_sorted])

    def color_for(s: float) -> str:
        # green-ish for positive, red-ish for negative
        s = max(-1.0, min(1.0, s))
        if s >= 0:
            alpha = 0.10 + 0.35 * abs(s)
            return f"rgba(34,197,94,{alpha:.3f})"   # green
        alpha = 0.10 + 0.35 * abs(s)
        return f"rgba(244,63,94,{alpha:.3f})"      # red

    out = []
    for i, (t, s) in enumerate(zip(tokens, scores)):
        # Fix RoBERTa token spacing (Ġ indicates a space before token)
        token_text = t.replace("Ġ", " ")
        if i in keep_idx and t not in ignore:
            bg = color_for(float(s))
            out.append(
                f"<span style='padding:3px 6px;border-radius:10px;background:{bg};"
                f"border:1px solid rgba(255,255,255,0.10);margin-right:3px;'>"
                f"{token_text}</span>"
            )
        else:
            out.append(f"<span style='margin-right:3px;opacity:0.92;'>{token_text}</span>")
    return "<div class='tokwrap'>" + "".join(out) + "</div>"


def _plot_top_tokens(tokens: List[str], scores: List[float], top_n: int = 18):
    ignore = {"<s>", "</s>", "<pad>"}
    rows = [(t.replace("Ġ", " "), float(s)) for t, s in zip(tokens, scores) if t not in ignore]
    rows = sorted(rows, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    rows = list(reversed(rows))  # for nicer bars (largest at top)

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.barh(labels, vals)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title("Top token attributions (Integrated Gradients)")
    ax.set_xlabel("Attribution (signed, normalized)")
    ax.grid(True, axis="x", alpha=0.25)
    st.pyplot(fig)


# =========================
# Pages
# =========================
def page_intro():
    _set_bg(BG_INTRO)
    _inject_css()
    _topbar()

    c1, c2 = st.columns([1.25, 1])

    with c1:
        st.markdown("<div class='card card-neon'>", unsafe_allow_html=True)
        st.markdown("<div class='h1'>Fake News Detection</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <p class="p">
              This application demonstrates a transformer-based NLP classifier that predicts whether a given
              news text is <b>FAKE</b> or <b>REAL</b>.
              The model is served from <b>Hugging Face Hub</b>, so the GitHub repository stays under the 25MB limit.
            </p>
            <div style="height:12px;"></div>
            <p class="p"><b>Steps</b></p>
            <ol class="p">
              <li>Click <b>Next</b></li>
              <li>Paste a news headline or full article text</li>
              <li>Click <b>Predict</b></li>
              <li>View prediction and <b>Explainable AI</b> on the result page</li>
            </ol>
            <p class="muted">
              Disclaimer: This tool is for educational purposes. Always verify information using trusted sources.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Next →", type="primary"):
            st.session_state.page = "predict"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 style='margin:0;color:rgba(248,250,252,0.96)'>Outputs</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="p">
              <li>Clear FAKE/REAL decision</li>
              <li>Confidence probabilities for both classes</li>
              <li>Explainable AI: token-level importance (Integrated Gradients)</li>
              <li>Attribution charts (bar + line) to show model behavior</li>
            </ul>
            <div class="muted">
              Label mapping: <b>FAKE → 0</b>, <b>REAL → 1</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def page_predict():
    _set_bg(BG_PRED)
    _inject_css()
    _topbar()

    top_left, top_right = st.columns([1, 4])
    with top_left:
        if st.button("← Back"):
            st.session_state.page = "intro"
            st.rerun()

    st.markdown("<div class='card card-neon'>", unsafe_allow_html=True)
    st.markdown("<div class='h1' style='font-size:38px;'>Paste News Text & Predict</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <p class="muted">
          Tip: Provide a full paragraph or article text for better confidence. Very short text may be unreliable.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Settings", expanded=False):
        max_len = st.slider("Max token length (truncation)", 128, 512, MAX_LEN, 64)
        st.caption("Higher max length may increase compute time. 256 is usually sufficient.")

    # store chosen max_len into session, used for prediction only (optional)
    st.session_state["max_len_runtime"] = int(max_len)

    default_text = st.session_state.get("input_text", "")
    text = st.text_area("Input text", value=default_text, height=240, placeholder="Paste the news text here...")

    a, b, c = st.columns([1, 1, 1])
    with a:
        do_predict = st.button("Predict", type="primary", use_container_width=True)
    with b:
        use_example = st.button("Use Example", use_container_width=True)
    with c:
        clear = st.button("Clear", use_container_width=True)

    if use_example:
        st.session_state.input_text = (
            "WASHINGTON (Reuters) - The administration announced new measures on Monday to strengthen enforcement "
            "and improve transparency, according to officials."
        )
        st.session_state.pred_label = None
        st.session_state.probs = None
        st.session_state.xai = None
        st.rerun()

    if clear:
        st.session_state.input_text = ""
        st.session_state.pred_label = None
        st.session_state.probs = None
        st.session_state.xai = None
        st.rerun()

    if do_predict:
        st.session_state.input_text = text
        st.session_state.pred_label = None
        st.session_state.probs = None
        st.session_state.xai = None

        if len(text.strip()) < 20:
            st.warning("Please enter at least 20 characters.")
        else:
            with st.spinner("Loading model and running inference..."):
                try:
                    # Use runtime max_len from settings
                    runtime_max_len = int(st.session_state.get("max_len_runtime", MAX_LEN))

                    tokenizer, model = _load_model()
                    enc = tokenizer(
                        [text.strip()],
                        padding=True,
                        truncation=True,
                        max_length=runtime_max_len,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(DEVICE) for k, v in enc.items()}
                    with torch.no_grad():
                        logits = model(**enc).logits
                        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

                    pred = int(np.argmax(probs))
                    st.session_state.pred_label = pred
                    st.session_state.probs = {"FAKE": float(probs[0]), "REAL": float(probs[1])}

                    st.success("Prediction complete. Click **Result →** to view output and Explainable AI.")
                except Exception as e:
                    st.error("Prediction failed.")
                    st.code(str(e))

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer navigation
    nav1, nav2 = st.columns([5, 1])
    with nav2:
        if st.button("Result →", use_container_width=True):
            st.session_state.page = "result"
            st.rerun()


def page_result():
    _set_bg(BG_RES)
    _inject_css()
    _topbar()

    top_left, top_right = st.columns([1, 4])
    with top_left:
        if st.button("← Back"):
            st.session_state.page = "predict"
            st.rerun()

    text = st.session_state.get("input_text", "")
    pred_label = st.session_state.get("pred_label", None)
    probs = st.session_state.get("probs", None)

    if pred_label is None or probs is None or len(text.strip()) == 0:
        st.warning("No prediction found. Go back to the Predict page and click Predict.")
        return

    pred_label = int(pred_label)
    p_fake = float(probs["FAKE"])
    p_real = float(probs["REAL"])

    if pred_label == 0:
        final = "FAKE"
        conf = p_fake
    else:
        final = "REAL"
        conf = p_real

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("<div class='card card-neon'>", unsafe_allow_html=True)
        st.markdown("<div class='h1' style='font-size:38px;'>Result</div>", unsafe_allow_html=True)

        # decision chip
        if final == "REAL":
            st.markdown(
                f"<span class='chip' style='border-color:rgba(34,197,94,0.35);background:rgba(34,197,94,0.10);'>"
                f"Prediction: REAL (1)</span>"
                f"<span class='chip'>Confidence: {conf*100:.2f}%</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<span class='chip' style='border-color:rgba(244,63,94,0.35);background:rgba(244,63,94,0.10);'>"
                f"Prediction: FAKE (0)</span>"
                f"<span class='chip'>Confidence: {conf*100:.2f}%</span>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="metric">
              <div class="k">FAKE Probability</div>
              <div class="v">{p_fake*100:.2f}%</div>
            </div>
            <div class="metric">
              <div class="k">REAL Probability</div>
              <div class="v">{p_real*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Show input text"):
            st.write(text)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 style='margin:0;color:rgba(248,250,252,0.96)'>Explainable AI</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <p class="muted">
              Integrated Gradients highlights tokens that influenced the prediction.
              Green tokens support the predicted class; red tokens oppose it.
            </p>
            """,
            unsafe_allow_html=True,
        )

        compute = st.button("Generate XAI (Integrated Gradients)", type="primary", use_container_width=True)
        st.caption("This may take some seconds depending on device.")

        if compute:
            with st.spinner("Computing attributions..."):
                try:
                    xai = _compute_ig(text, target_class=pred_label, n_steps=24)
                    st.session_state.xai = xai
                    st.success("XAI generated.")
                except Exception as e:
                    st.error("XAI failed.")
                    st.code(str(e))

        st.markdown("</div>", unsafe_allow_html=True)

    # XAI visualizations (bottom, full width)
    if st.session_state.get("xai") is not None:
        xai = st.session_state.xai
        tokens = xai["tokens"]
        scores = xai["scores"]

        st.markdown("<div class='card card-neon'>", unsafe_allow_html=True)
        st.markdown("<h2 style='margin:0;color:rgba(248,250,252,0.96)'>How the model found this</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='muted'>Below are token highlights plus two charts: a bar chart (top tokens) and a line chart (attribution across sequence).</p>",
            unsafe_allow_html=True,
        )

        # Token highlight
        html = _html_token_highlight(tokens, scores, top_k=44)
        st.markdown(html, unsafe_allow_html=True)

        st.write("")
        c1, c2 = st.columns([1.1, 1])

        with c1:
            _plot_top_tokens(tokens, scores, top_n=18)

        with c2:
            # Line chart: attribution across sequence (shape/line requested)
            ignore = {"<s>", "</s>", "<pad>"}
            seq_scores = [float(s) for t, s in zip(tokens, scores) if t not in ignore]
            df_line = pd.DataFrame({"token_index": np.arange(len(seq_scores)), "attribution": seq_scores}).set_index("token_index")
            st.subheader("Attribution shape across text")
            st.line_chart(df_line)

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Router
# =========================
if st.session_state.page == "intro":
    page_intro()
elif st.session_state.page == "predict":
    page_predict()
else:
    page_result()
