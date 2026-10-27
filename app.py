"""AgroScan — Streamlit entry point.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="AgroScan · Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.artifacts import (
    demo_options,
    demo_zip,
    load_class_distribution,
    load_classification_report,
    load_test_metrics,
    load_training_history,
)
from src.config import (
    CONFIDENCE_LIKELY,
    CONFIDENCE_UNCLEAR,
    CONFUSION_MATRIX_PATH,
    COURSE_LINE,
    DEPLOY_TEMPLATE_URL,
    DISCLAIMER,
    GITHUB_REPO_URL,
    HF_REPO_ID,
    LIVE_APP_URL,
    MAX_BATCH_FILES,
    MAX_UPLOAD_MB,
    SUPPORTED_IMAGE_TYPES,
)
from src.disease_info import by_crop, get_info
from src.labels import parse_label, triage_prediction
from src.model import (
    LoadedModel,
    ModelLoadError,
    generate_gradcam,
    load_model,
    predict,
    to_rgb,
    warm_up,
)
from src.quality import assess_image_quality

APP_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CSS = """
<style>
:root {
    --agro-green: #1f6b3b;
    --agro-green-deep: #123522;
    --agro-lime: #8cc26d;
    --agro-paper: #f7f9f3;
    --agro-border: #dde6d7;
    --agro-ink: #172a1d;
    --agro-muted: #5e6f61;
    --agro-amber: #b7791f;
    --agro-danger: #b3402e;
}

.stApp {
    background:
        radial-gradient(1100px 500px at 85% -10%, rgba(140, 194, 109, 0.20), transparent 60%),
        radial-gradient(900px 500px at 0% 30%, rgba(214, 230, 202, 0.35), transparent 65%),
        var(--agro-paper);
}
.block-container {
    max-width: 1240px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}
header[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stToolbar"] {
    visibility: hidden;
}
footer { visibility: hidden; }

h1, h2, h3, h4 {
    color: var(--agro-ink);
    letter-spacing: -0.015em;
}

/* Hero --------------------------------------------------------------- */
.agro-hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 2rem 2.2rem 1.9rem;
    margin-bottom: 1.4rem;
    color: #f6faf2;
    background:
        radial-gradient(600px 300px at 95% -20%, rgba(181, 220, 150, 0.35), transparent 65%),
        linear-gradient(120deg, #102d1c 0%, #1c5933 55%, #2c7a48 100%);
    box-shadow: 0 18px 40px rgba(18, 53, 34, 0.18);
}
.agro-hero::after {
    content: "";
    position: absolute;
    width: 340px;
    height: 340px;
    right: -120px;
    bottom: -210px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.16), transparent 68%);
}
.agro-hero-top {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.agro-logo {
    width: 58px;
    height: 58px;
    flex: 0 0 58px;
    filter: drop-shadow(0 6px 14px rgba(0, 0, 0, 0.22));
}
.agro-kicker {
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 650;
    color: #cfe5c4;
}
.agro-title {
    font-size: 2.9rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.035em;
    margin: 0.15rem 0 0.55rem;
}
.agro-subtitle {
    max-width: 780px;
    color: #dcebdd;
    font-size: 1.02rem;
    line-height: 1.58;
    margin: 0.35rem 0 0;
}
.agro-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.15rem;
    position: relative;
    z-index: 2;
}
.agro-pills span {
    border: 1px solid rgba(255, 255, 255, 0.22);
    background: rgba(255, 255, 255, 0.09);
    padding: 0.3rem 0.72rem;
    border-radius: 999px;
    font-size: 0.76rem;
    color: #eef7e9;
    white-space: nowrap;
}
.agro-pills span strong { color: #c9e8b5; }

/* Sidebar ------------------------------------------------------------ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef4e8 0%, #f7f9f3 100%);
    border-right: 1px solid var(--agro-border);
}
.agro-side-brand {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--agro-green-deep);
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem;
}
.agro-side-brand span {
    font-size: 1.5rem;
}
.agro-status-card {
    border: 1px solid #cfe0c6;
    background: #eaf3e3;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    margin: 0.7rem 0 1rem;
    font-size: 0.82rem;
    color: #345041;
    line-height: 1.55;
}
.agro-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2f9b58;
    box-shadow: 0 0 0 3px rgba(47, 155, 88, 0.15);
    margin-right: 0.45rem;
}
.agro-threshold {
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
    font-size: 0.8rem;
    line-height: 1.45;
    color: var(--agro-muted);
    margin: 0.5rem 0;
}
.agro-threshold b { color: var(--agro-ink); }
.agro-threshold-dot {
    width: 9px;
    height: 9px;
    flex: 0 0 9px;
    border-radius: 50%;
    margin-top: 0.3rem;
}

/* Cards & results ---------------------------------------------------- */
.agro-result {
    border-radius: 18px;
    padding: 1.3rem 1.4rem 1.15rem;
    border: 1px solid;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(25, 50, 30, 0.06);
}
.agro-result.danger { background: #fff4f0; border-color: #efc4b8; }
.agro-result.warning { background: #fff9e9; border-color: #ecd39c; }
.agro-result.success { background: #f1f8ec; border-color: #c9dfbd; }
.agro-result.neutral { background: #f4f6f2; border-color: #d7ded2; }
.agro-status {
    font-size: 0.7rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    font-weight: 750;
    color: var(--agro-muted);
}
.agro-result.danger .agro-status { color: var(--agro-danger); }
.agro-result.warning .agro-status { color: var(--agro-amber); }
.agro-result.success .agro-status { color: var(--agro-green); }
.agro-prediction {
    font-size: 1.95rem;
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -0.03em;
    color: var(--agro-ink);
    margin: 0.28rem 0 0.3rem;
}
.agro-prediction small {
    font-weight: 500;
    color: var(--agro-muted);
    font-size: 1.15rem;
}
.agro-confidence {
    font-size: 0.92rem;
    color: var(--agro-muted);
}
.agro-confidence b { color: var(--agro-ink); }

.agro-bar-row { margin: 0.55rem 0 0.8rem; }
.agro-bar-label {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
    color: var(--agro-ink);
}
.agro-bar-label b { font-variant-numeric: tabular-nums; color: var(--agro-muted); font-weight: 600; }
.agro-bar {
    height: 9px;
    background: #e7eee2;
    border-radius: 999px;
    overflow: hidden;
}
.agro-bar span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2f7d4f, #8cc26d);
}
.agro-bar.dim span { background: #b9cbb0; }

.agro-kpi {
    background: #ffffff;
    border: 1px solid var(--agro-border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    height: 100%;
    box-shadow: 0 6px 18px rgba(25, 50, 30, 0.04);
}
.agro-kpi b {
    display: block;
    font-size: 1.7rem;
    line-height: 1.15;
    letter-spacing: -0.03em;
    color: var(--agro-green);
}
.agro-kpi span {
    display: block;
    margin-top: 0.25rem;
    font-size: 0.74rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--agro-muted);
    font-weight: 650;
}

.agro-note {
    border-left: 3px solid var(--agro-lime);
    background: rgba(255, 255, 255, 0.65);
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 0.9rem;
    color: var(--agro-muted);
    font-size: 0.85rem;
    line-height: 1.55;
}
.agro-footer {
    margin-top: 2.4rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--agro-border);
    color: var(--agro-muted);
    font-size: 0.78rem;
    line-height: 1.6;
}
.agro-footer a { color: var(--agro-green); }

/* Native widget polish ---------------------------------------------- */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--agro-border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 6px 18px rgba(25, 50, 30, 0.04);
}
[data-testid="stFileUploaderDropzone"] {
    background: #fbfcf8;
    border: 1.5px dashed #b9cbb0;
    border-radius: 16px;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--agro-green);
    background: #f5f9f1;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
    border-bottom: 1px solid var(--agro-border);
}
.stTabs [data-baseweb="tab"] {
    height: 2.7rem;
    border-radius: 10px 10px 0 0;
    padding: 0 1rem;
    color: var(--agro-muted);
    font-weight: 650;
}
.stTabs [aria-selected="true"] {
    color: var(--agro-green) !important;
    background: rgba(140, 194, 109, 0.10);
}
.stTabs [class*="SelectionIndicator"] {
    background: var(--agro-green) !important;
}
.stButton > button, .stDownloadButton > button, .stLinkButton > a {
    border-radius: 10px;
    font-weight: 650;
}
div[data-testid="stExpander"] details {
    border-radius: 12px;
    border-color: var(--agro-border);
    background: rgba(255, 255, 255, 0.55);
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        f"""
        <section class="agro-hero">
          <div class="agro-hero-top">
            <svg class="agro-logo" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect width="64" height="64" rx="19" fill="url(#agroGradient)"/>
              <path d="M49 14C31 14 16 23 16 39c0 4 1 7 3 10 2-11 9-18 20-22-8 6-13 14-14 24 3 2 6 3 10 3 15 0 22-12 22-28 0-5-1-12-1-12z" fill="#F5FAF0"/>
              <path d="M21 51c4-14 12-22 25-27" stroke="#2F7D4F" stroke-width="2.8" stroke-linecap="round"/>
              <defs>
                <linearGradient id="agroGradient" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
                  <stop stop-color="#2E7D4F"/>
                  <stop offset="1" stop-color="#173D26"/>
                </linearGradient>
              </defs>
            </svg>
            <div>
              <div class="agro-kicker">{COURSE_LINE}</div>
              <div class="agro-title">AgroScan</div>
            </div>
          </div>
          <p class="agro-subtitle">
            Screen a close-up leaf photo for 26 diseases and 12 healthy states across 14 crops.
            Get a calibrated confidence, the visual evidence behind the call, and cautious next steps.
          </p>
          <div class="agro-pills">
            <span><strong>EfficientNet-B0</strong> fine-tuned</span>
            <span>39 classes</span>
            <span>99.56% test top-1</span>
            <span>99.40% macro F1</span>
            <span>SDG 2 · Zero Hunger</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data / model helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_model() -> LoadedModel:
    loaded = load_model()
    warm_up(loaded)
    return loaded


def decode_upload(uploaded) -> Image.Image | None:
    if uploaded is None:
        return None
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"{uploaded.name} is larger than {MAX_UPLOAD_MB} MB. Please resize it first.")
        return None
    try:
        image = Image.open(io.BytesIO(uploaded.getvalue()))
        image.load()
        return to_rgb(image)
    except (UnidentifiedImageError, OSError, ValueError):
        st.error(f"Could not read {uploaded.name}. Please use a JPG, PNG or WEBP image.")
        return None


def _label_parts(label: str) -> tuple[str, str]:
    parsed = parse_label(label)
    if parsed.is_unknown:
        return "Unsupported", parsed.condition_display
    return parsed.crop_display, parsed.condition_display


def format_confidence(value: float) -> str:
    """Avoid reporting a near-certain model as a misleading 100.0%."""
    if value >= 0.99995:
        return ">99.99%"
    if value >= 0.999:
        return f"{value:.2%}"
    if 0 < value < 0.001:
        return "<0.1%"
    return f"{value:.1%}"


def top_k_html(prediction, loaded: LoadedModel, k: int = 3) -> str:
    blocks = []
    for rank, (index, probability) in enumerate(prediction.top_k(k)):
        parsed = parse_label(loaded.labels[index])
        name = parsed.display
        width = max(1.0, probability * 100)
        dim = " dim" if rank else ""
        blocks.append(
            f"""
            <div class="agro-bar-row">
              <div class="agro-bar-label"><span>{html.escape(name)}</span><b>{format_confidence(probability)}</b></div>
              <div class="agro-bar{dim}"><span style="width:{width:.2f}%"></span></div>
            </div>
            """
        )
    return "".join(blocks)


def quality_line(quality) -> None:
    if quality.warnings:
        st.warning("Image check: " + " ".join(quality.warnings))
    else:
        st.caption(
            f"Image check passed · {quality.width}×{quality.height} px · "
            f"brightness {quality.brightness:.0f}/255 · sharpness {quality.sharpness:.0f}"
        )


def build_screening_report(
    filename: str,
    prediction,
    loaded: LoadedModel,
) -> str:
    parsed = parse_label(prediction.class_name)
    triage = triage_prediction(prediction.class_name, prediction.confidence)
    info = get_info(prediction.class_name)
    alternatives = []
    for index, probability in prediction.top_k(3):
        alt_crop, alt_condition = _label_parts(loaded.labels[index])
        alternatives.append(f"{alt_crop} · {alt_condition} ({format_confidence(probability)})")
    return "\n".join(
        [
            "# AgroScan screening summary",
            "",
            f"- **Date:** {datetime.now().astimezone().strftime('%d %b %Y, %H:%M %Z')}",
            f"- **Image:** {filename}",
            f"- **Crop:** {parsed.crop_display}",
            f"- **Condition:** {parsed.condition_display}",
            f"- **Confidence:** {format_confidence(prediction.confidence)}",
            f"- **Status:** {triage.title}",
            f"- **Model:** EfficientNet-B0 · {HF_REPO_ID}",
            "",
            "## Top-3 possibilities",
            *[f"- {item}" for item in alternatives],
            "",
            "## What to look for",
            info["symptoms"],
            "",
            "## Suggested next steps",
            info["management"],
            "",
            "## Prevention",
            info["prevention"],
            "",
            "---",
            DISCLAIMER,
            "",
            "Generated by AgroScan for educational screening only.",
        ]
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar(loaded: LoadedModel) -> None:
    with st.sidebar:
        st.markdown('<div class="agro-side-brand"><span>🌿</span> AgroScan</div>', unsafe_allow_html=True)
        st.caption("Plant disease screening · v" + APP_VERSION)
        st.markdown(
            f"""
            <div class="agro-status-card">
              <b><span class="agro-dot"></span>Model ready</b><br/>
              EfficientNet-B0 · {loaded.num_classes} classes<br/>
              {html.escape(loaded.device_label)} · {loaded.model_size_mb:.1f} MB checkpoint
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Confidence bands")
        st.markdown(
            f"""
            <div class="agro-threshold"><span class="agro-threshold-dot" style="background:#2f9b58"></span>
              <div><b>≥ {CONFIDENCE_LIKELY:.0%}</b> — likely match</div></div>
            <div class="agro-threshold"><span class="agro-threshold-dot" style="background:#d7a33a"></span>
              <div><b>{CONFIDENCE_UNCLEAR:.0%}–{CONFIDENCE_LIKELY:.0%}</b> — possible, confirm with an expert</div></div>
            <div class="agro-threshold"><span class="agro-threshold-dot" style="background:#9aa79b"></span>
              <div><b>&lt; {CONFIDENCE_UNCLEAR:.0%}</b> — unclear, retake the photo</div></div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("##### Explore")
        st.link_button("Hugging Face model card", f"https://huggingface.co/{HF_REPO_ID}", width="stretch")
        st.link_button("GitHub repository", GITHUB_REPO_URL, width="stretch")
        if LIVE_APP_URL:
            st.link_button("Live application", LIVE_APP_URL, width="stretch")
        st.link_button("Deploy your own copy", DEPLOY_TEMPLATE_URL, width="stretch")
        st.download_button(
            "Download demo images",
            data=demo_zip(),
            file_name="agroscan_demo_images.zip",
            mime="application/zip",
            width="stretch",
        )

        st.divider()
        st.caption(
            "**Responsible use.** AgroScan was trained on lab-style PlantVillage "
            "close-ups. It can be wrong on field photos, unsupported crops or "
            "unseen diseases. It is a screening aid, not a diagnosis."
        )
        st.caption(f"Model by [{HF_REPO_ID.split('/')[0]}](https://huggingface.co/{HF_REPO_ID.split('/')[0]}), MIT licence.")


# ---------------------------------------------------------------------------
# Diagnose tab
# ---------------------------------------------------------------------------


def render_diagnose(loaded: LoadedModel) -> None:
    st.markdown("### Screen one leaf")
    st.caption(
        "Use one clear, close-up leaf that fills most of the frame. "
        "The model supports apple, blueberry, cherry, corn, grape, orange, peach, "
        "pepper, potato, raspberry, soybean, squash, strawberry and tomato."
    )

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload a leaf photo",
            type=SUPPORTED_IMAGE_TYPES,
            help=f"JPG, PNG or WEBP up to {MAX_UPLOAD_MB} MB. One leaf, even daylight, in focus.",
        )
        with st.expander("Demo image or camera", expanded=False):
            options = demo_options()
            demo_path: Path | None = None
            if options:
                choice = st.selectbox("Try a PlantVillage demo", list(options), index=0)
                demo_path = options[choice]
            camera = st.camera_input("Or take a photo", key="camera")

        image: Image.Image | None = None
        source_label = ""
        if uploaded is not None:
            image = decode_upload(uploaded)
            source_label = f"Uploaded · {uploaded.name}"
        elif camera is not None:
            image = decode_upload(camera)
            source_label = "Camera capture"
        elif demo_path is not None:
            image = to_rgb(Image.open(demo_path))
            source_label = "Demo image · PlantVillage"

        if image is None:
            st.markdown(
                '<div class="agro-note">Upload a leaf photo or open the demo panel to try a sample image.</div>',
                unsafe_allow_html=True,
            )
            return

        st.image(image, caption=source_label, width="stretch")
        if uploaded is None and camera is None:
            st.caption("A demo image is shown automatically. Upload your own photo to replace it.")
        quality = assess_image_quality(image)
        quality_line(quality)

    with right:
        st.markdown("**Screening result**")
        with st.spinner("Running EfficientNet-B0 inference…"):
            prediction = predict(image, loaded)
        parsed = parse_label(prediction.class_name)
        triage = triage_prediction(prediction.class_name, prediction.confidence)
        info = get_info(prediction.class_name)

        confidence_text = f"{format_confidence(prediction.confidence)} model confidence"
        if parsed.is_unknown:
            prediction_title = html.escape(parsed.condition_display)
        else:
            prediction_title = (
                f"{html.escape(parsed.crop_display)} <small>·</small> "
                f"{html.escape(parsed.condition_display)}"
            )
        st.markdown(
            f"""
            <div class="agro-result {triage.tone}">
              <div class="agro-status">{html.escape(triage.title)}</div>
              <div class="agro-prediction">{prediction_title}</div>
              <div class="agro-confidence"><b>{confidence_text}</b> · {html.escape(info['category'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(triage.message)

        st.markdown("##### Top 3 possibilities")
        st.markdown(top_k_html(prediction, loaded), unsafe_allow_html=True)

        guidance_left, guidance_right = st.columns(2, gap="medium")
        with guidance_left, st.container(border=True):
            st.markdown("**What to look for**")
            st.write(info["symptoms"])
        with guidance_right, st.container(border=True):
            st.markdown("**Suggested next steps**")
            st.write(info["management"])

        with st.expander("Prevention habits for this crop"):
            st.write(info["prevention"])

        action_left, action_right = st.columns([1, 1], gap="medium")
        with action_left:
            report = build_screening_report(
                source_label.replace("Uploaded · ", "").replace("Demo image · ", "demo_"),
                prediction,
                loaded,
            )
            st.download_button(
                "Download screening summary",
                data=report,
                file_name="agroscan_screening_summary.md",
                mime="text/markdown",
                width="stretch",
            )
        with action_right:
            show_cam = st.toggle("Explain with Grad-CAM", value=False, help="Highlights the leaf regions that most influenced the prediction.")

        if show_cam:
            with st.expander("Why did the model decide?", expanded=True):
                with st.spinner("Computing Grad-CAM…"):
                    gradcam = generate_gradcam(image, loaded, prediction.top_indices[0])
                original_col, cam_col = st.columns(2)
                with original_col:
                    st.image(image, caption="Original", width="stretch")
                with cam_col:
                    st.image(gradcam.overlay, caption="Model attention", width="stretch")
                st.caption(
                    "Warm colours mark regions that increased the predicted class score. "
                    "Grad-CAM is an explanation aid, not a lesion segmentation or a proof of cause."
                )

        st.markdown(
            f'<div class="agro-note">{html.escape(DISCLAIMER)}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Batch tab
# ---------------------------------------------------------------------------


def render_batch(loaded: LoadedModel) -> None:
    st.markdown("### Screen a folder of leaves")
    st.caption(
        f"Upload up to {MAX_BATCH_FILES} images. Each image is screened independently; "
        "the CSV keeps the top prediction, confidence and triage status."
    )

    files = st.file_uploader(
        "Upload multiple leaf photos",
        type=SUPPORTED_IMAGE_TYPES,
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if not files:
        st.markdown(
            f"""
            <div class="agro-note">
              <b>How to use batch screening:</b> drop 2–{MAX_BATCH_FILES} close-up leaf photos above.
              Results are shown as a table and can be downloaded as CSV for the lab report.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if len(files) > MAX_BATCH_FILES:
        st.error(f"Please upload at most {MAX_BATCH_FILES} images at a time.")
        return

    rows: list[dict] = []
    previews: list[tuple[str, Image.Image, str, str]] = []
    progress = st.progress(0.0, text="Screening images…")
    for number, uploaded in enumerate(files, start=1):
        image = decode_upload(uploaded)
        if image is None:
            continue
        prediction = predict(image, loaded)
        parsed = parse_label(prediction.class_name)
        triage = triage_prediction(prediction.class_name, prediction.confidence)
        flag = "🟢" if parsed.is_healthy else "🔴" if triage.level == "likely" else "🟡" if triage.level == "possible" else "⚪"
        alt_index, alt_probability = prediction.top_k(2)[1]
        alt_crop, alt_condition = _label_parts(loaded.labels[alt_index])
        rows.append(
            {
                "File": uploaded.name,
                "Crop": parsed.crop_display,
                "Condition": parsed.condition_display,
                "Confidence (%)": round(prediction.confidence * 100, 2),
                "Status": triage.title,
                "Flag": flag,
                "Canonical label": prediction.class_name,
                "Alternative": f"{alt_crop} · {alt_condition} ({alt_probability:.1%})",
            }
        )
        previews.append((uploaded.name, image, f"{parsed.crop_display} · {parsed.condition_display}", triage.title))
        progress.progress(number / len(files), text=f"Screening {number}/{len(files)} · {uploaded.name}")
    progress.empty()

    if not rows:
        st.error("None of the uploaded files could be read.")
        return

    data = pd.DataFrame(rows)
    disease_likely = int(((data["Status"] == "Likely disease detected")).sum())
    needs_review = int(data["Status"].isin(["Unknown or unclear", "Possible disease detected", "Not a supported leaf"]).sum())

    metric_left, metric_mid, metric_right = st.columns(3)
    with metric_left:
        st.metric("Images screened", len(data))
    with metric_mid:
        st.metric("Likely disease", disease_likely)
    with metric_right:
        st.metric("Needs a second look", needs_review)

    st.markdown("##### Results")
    st.dataframe(
        data[["Flag", "File", "Crop", "Condition", "Confidence (%)", "Status", "Alternative"]],
        hide_index=True,
        width="stretch",
        column_config={
            "Confidence (%)": st.column_config.ProgressColumn(
                "Confidence (%)",
                help="Softmax confidence for the top prediction.",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
            "Status": st.column_config.TextColumn("Triage", width="medium"),
        },
    )

    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv_data,
        file_name="agroscan_batch_results.csv",
        mime="text/csv",
        width="content",
    )

    st.markdown("##### Preview")
    columns = st.columns(4)
    for index, (name, image, label, status) in enumerate(previews[:8]):
        with columns[index % 4]:
            st.image(image, caption=f"{name}\n{label} · {status}", width="stretch")
    if len(previews) > 8:
        st.caption(f"Showing the first 8 of {len(previews)} images. All results are in the table and CSV.")

    st.caption(DISCLAIMER)


# ---------------------------------------------------------------------------
# Model tab
# ---------------------------------------------------------------------------


def _style_figure(figure: go.Figure, height: int = 340) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        font={"color": "#33483a", "size": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        title_font={"size": 14, "color": "#172a1d"},
    )
    figure.update_xaxes(gridcolor="#e7eee2", linecolor="#dde6d7")
    figure.update_yaxes(gridcolor="#e7eee2", linecolor="#dde6d7")
    return figure


def render_model_tab(loaded: LoadedModel) -> None:
    metrics = load_test_metrics()
    history = load_training_history()
    distribution = load_class_distribution()
    report = load_classification_report()

    st.markdown("### Model & evaluation")
    st.caption(
        "Held-out PlantVillage test split from the published model card. "
        "These numbers describe lab-style close-up images, not field conditions."
    )

    columns = st.columns(4)
    figures = [
        (f"{metrics['accuracy_top1']:.2%}", "Top-1 accuracy"),
        (f"{metrics['accuracy_top3']:.2%}", "Top-3 accuracy"),
        (f"{metrics['f1_macro']:.2%}", "Macro F1"),
        (f"{metrics['num_images']:,}", "Test images"),
    ]
    for column, (value, label) in zip(columns, figures, strict=True):
        with column:
            st.markdown(
                f'<div class="agro-kpi"><b>{value}</b><span>{label}</span></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("##### Cross-entropy loss")
        loss_long = history.melt(
            id_vars="epoch",
            value_vars=["train_loss", "val_loss"],
            var_name="split",
            value_name="loss",
        )
        loss_long["split"] = loss_long["split"].str.replace("_loss", "", regex=False).str.title()
        loss_figure = px.line(
            loss_long,
            x="epoch",
            y="loss",
            color="split",
            markers=True,
            color_discrete_sequence=["#8cc26d", "#1f6b3b"],
            labels={"epoch": "Epoch", "loss": "Loss", "split": "Split"},
        )
        st.plotly_chart(_style_figure(loss_figure), width="stretch", config={"displayModeBar": False})

    with right:
        st.markdown("##### Accuracy and macro F1")
        score_long = history.melt(
            id_vars="epoch",
            value_vars=["train_accuracy", "val_accuracy", "val_macro_f1"],
            var_name="metric",
            value_name="score",
        )
        labels = {
            "train_accuracy": "Train accuracy",
            "val_accuracy": "Validation accuracy",
            "val_macro_f1": "Validation macro F1",
        }
        score_long["metric"] = score_long["metric"].map(labels)
        score_figure = px.line(
            score_long,
            x="epoch",
            y="score",
            color="metric",
            markers=True,
            color_discrete_sequence=["#a9c89a", "#1f6b3b", "#d7a33a"],
            labels={"epoch": "Epoch", "score": "Score", "metric": ""},
        )
        score_figure.update_yaxes(range=[0.85, 1.005], tickformat=".0%")
        st.plotly_chart(_style_figure(score_figure), width="stretch", config={"displayModeBar": False})

    best_epoch = int(history.loc[history["val_macro_f1"].idxmax(), "epoch"])
    st.markdown(
        f"""
        <div class="agro-note">
          The {len(history)}-epoch training run selected its best checkpoint at epoch
          <b>{best_epoch}</b> by validation macro F1 (<b>{history['val_macro_f1'].max():.4f}</b>).
          The deployed checkpoint contains <b>{loaded.parameter_count:,}</b> parameters and is
          <b>{loaded.model_size_mb:.1f} MB</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    distribution["crop"] = distribution["class_name"].apply(lambda label: parse_label(label).crop_display)
    distribution["kind"] = distribution["class_name"].apply(
        lambda label: "Unknown" if parse_label(label).is_unknown else ("Healthy" if parse_label(label).is_healthy else "Disease")
    )
    distribution["display"] = distribution["class_name"].apply(
        lambda label: parse_label(label).display
    )
    top_classes = distribution.nlargest(14, "train_count").sort_values("train_count")
    figure = px.bar(
        top_classes,
        x="train_count",
        y="display",
        orientation="h",
        color="kind",
        color_discrete_map={"Disease": "#b7791f", "Healthy": "#2f7d4f", "Unknown": "#8b9a8d"},
        title="Most represented training classes",
        labels={"train_count": "Training images", "display": "", "kind": ""},
    )
    figure.update_layout(height=430, legend_title_text="")
    st.plotly_chart(_style_figure(figure, height=430), width="stretch", config={"displayModeBar": False})
    st.caption(
        "PlantVillage is imbalanced: orange huanglongbing has 3,854 training images, while "
        "potato healthy has only 106. The published training used class-weighted cross-entropy."
    )

    st.markdown("##### Per-class test performance")
    report["display"] = report["class_name"].apply(lambda label: parse_label(label).display)
    report = report.sort_values("f1", ascending=True)
    st.dataframe(
        report[["display", "precision", "recall", "f1", "support"]],
        hide_index=True,
        width="stretch",
        column_config={
            "display": st.column_config.TextColumn("Class", width="large"),
            "precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1, format="%.3f"),
            "recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.3f"),
            "f1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f"),
            "support": st.column_config.NumberColumn("Test images", format="%d"),
        },
    )
    st.caption("Weakest classes first. Even the lowest per-class F1, Cercospora / gray leaf spot, is above 0.92.")

    with st.expander("Full class distribution and test confusion matrix"):
        st.dataframe(
            distribution.sort_values("train_count", ascending=False)[["display", "kind", "train_count", "class_weight"]],
            hide_index=True,
            width="stretch",
            column_config={
                "display": st.column_config.TextColumn("Class", width="large"),
                "kind": st.column_config.TextColumn("Type"),
                "train_count": st.column_config.NumberColumn("Training images", format="%d"),
                "class_weight": st.column_config.NumberColumn("Class weight", format="%.3f"),
            },
        )
        st.image(
            str(CONFUSION_MATRIX_PATH),
            caption="PlantVillage test confusion matrix published with the checkpoint.",
            width="stretch",
        )

    with st.expander("Model card, intended use and limitations", expanded=False):
        recipe_left, recipe_right = st.columns(2, gap="large")
        with recipe_left:
            st.markdown(
                """
                **Training recipe**
                - Architecture: torchvision **EfficientNet-B0**, ImageNet initialisation
                - Head: 39-way linear layer (1,280 → 39)
                - Input: 224 × 224 RGB, ImageNet mean/std
                - Optimiser: AdamW, learning rate 1e-4
                - Loss: class-weighted cross-entropy
                - Epochs: 15 recorded in `training_history.csv`; best checkpoint at epoch 12
                - Selection metric: validation macro F1
                """
            )
        with recipe_right:
            st.markdown(
                """
                **Intended use**
                - Close-up, leaf-level screening of one leaf
                - The 14 supported crops and 38 known classes plus unknown
                - Education, triage and early warning

                **Not intended for**
                - Whole-field or far-away whole-plant photos
                - Unsupported crops or diseases
                - A guaranteed diagnosis or a pesticide prescription
                """
            )
        st.warning(
            "Softmax confidence can be overconfident on out-of-distribution images. "
            "The 99.56% test accuracy reflects PlantVillage's controlled backgrounds and "
            "does not transfer automatically to phone photos from a farm."
        )

    st.markdown(
        f"""
        <div class="agro-note">
          Model provenance: <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank">{HF_REPO_ID}</a>
          · MIT licence · author <b>{HF_REPO_ID.split('/')[0]}</b>. Test metrics are copied
          unchanged from the published <code>plantvillage_test_metrics.json</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Crop guide tab
# ---------------------------------------------------------------------------


def render_crop_guide() -> None:
    st.markdown("### Crop and disease guide")
    st.caption(
        "Plain-language notes for every class the model can predict. "
        "Confirm locally before treating a crop."
    )
    groups = by_crop()
    crop = st.selectbox("Choose a crop", list(groups), index=list(groups).index("Tomato"))
    labels = groups[crop]
    disease_labels = [label for label in labels if not parse_label(label).is_healthy]
    st.markdown(
        f"**{crop}** · {len(disease_labels)} disease class{'es' if len(disease_labels) != 1 else ''} "
        f"and {len(labels) - len(disease_labels)} healthy class"
    )

    for label in labels:
        parsed = parse_label(label)
        info = get_info(label)
        icon = "🌿" if parsed.is_healthy else "🔎"
        title = "Healthy leaf · no disease detected" if parsed.is_healthy else f"{parsed.condition_display} · {info['category']}"
        with st.expander(f"{icon}  {title}"):
            symptoms_col, management_col = st.columns(2, gap="large")
            with symptoms_col:
                st.markdown("**Symptoms**")
                st.write(info["symptoms"])
            with management_col:
                st.markdown("**Management**")
                st.write(info["management"])
            st.caption("Prevention: " + info["prevention"])
            st.caption("Canonical model label: `" + label + "`")

    st.info(
        "If your crop is not in the list, AgroScan cannot screen it reliably. "
        "Use a local extension service or a model trained for that crop."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def footer() -> None:
    st.markdown(
        f"""
        <div class="agro-footer">
          <b>AgroScan v{APP_VERSION}</b> · {COURSE_LINE} · Federal Institute of Science and Technology,
          Department of Computer Science and Engineering.<br/>
          Model: <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank">AgroScan EfficientNet-B0</a>
          by {HF_REPO_ID.split('/')[0]} (MIT). Code: <a href="{GITHUB_REPO_URL}" target="_blank">GitHub</a>.
          Dataset: PlantVillage (Mohanty et al., 2016).<br/>
          {DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    hero()

    with st.spinner("Loading the AgroScan model…"):
        try:
            loaded = get_model()
        except ModelLoadError as exc:
            st.error(str(exc))
            st.info(
                "If you are running locally, the model should be at "
                "`models/agroscan_efficientnet_b0.pth`. You can download it with "
                "`python scripts/download_model.py`."
            )
            st.stop()

    sidebar(loaded)

    diagnose_tab, batch_tab, model_tab, guide_tab = st.tabs(
        ["🔬 Diagnose", "🗂️ Batch screening", "📊 Model & metrics", "🌱 Crop guide"]
    )
    with diagnose_tab:
        render_diagnose(loaded)
    with batch_tab:
        render_batch(loaded)
    with model_tab:
        render_model_tab(loaded)
    with guide_tab:
        render_crop_guide()

    footer()


if __name__ == "__main__":
    main()
