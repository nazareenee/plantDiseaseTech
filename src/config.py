"""Application-wide configuration for AgroScan.

Values here are shared by the Streamlit app, tests and helper scripts so that
the model contract (input size, normalisation, label order, thresholds) is
defined in exactly one place.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DIR = PROJECT_ROOT / "sample_images"

APP_NAME = "AgroScan"
APP_TAGLINE = "AI-assisted plant disease screening from a close-up leaf photo"
COURSE_LINE = "PCCSL 508 · Machine Learning Lab · Semester V · Project 1"

GITHUB_REPO_URL = "https://github.com/countluss/agroscan-plant-disease-detection"
# Set after the Streamlit Community Cloud deployment is live.
LIVE_APP_URL = ""
DEPLOY_TEMPLATE_URL = (
    "https://share.streamlit.io/deploy?"
    f"repository={GITHUB_REPO_URL}&branch=main&mainModule=app.py"
)

HF_REPO_ID = "tahmidalbi/agroscan-efficientnet-b0"
MODEL_FILENAME = "agroscan_efficientnet_b0.pth"
LOCAL_MODEL_PATH = MODELS_DIR / MODEL_FILENAME
MODEL_LICENSE = "MIT"
MODEL_AUTHOR = "tahmidalbi"

LABELS_PATH = ASSETS_DIR / "labels.json"
MANIFEST_PATH = ASSETS_DIR / "model_manifest.json"
TEST_METRICS_PATH = ASSETS_DIR / "plantvillage_test_metrics.json"
METRICS_PATH = ASSETS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = ASSETS_DIR / "plantvillage_test_classification_report.json"
CLASS_DISTRIBUTION_PATH = ASSETS_DIR / "class_distribution.csv"
TRAINING_HISTORY_PATH = ASSETS_DIR / "training_history.csv"
CONFUSION_MATRIX_PATH = ASSETS_DIR / "plantvillage_test_confusion_matrix.png"
DISEASE_INFO_PATH = ASSETS_DIR / "disease_info.json"

# Pre-processing contract published in the model manifest.
IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Confidence triage thresholds from the model card.
CONFIDENCE_UNCLEAR = 0.70
CONFIDENCE_LIKELY = 0.90

TOP_K = 3
MAX_BATCH_FILES = 30
MAX_UPLOAD_MB = 15

# Optional override, mainly for tests and CPU-only machines.
DEVICE_OVERRIDE = os.getenv("AGROSCAN_DEVICE", "").strip().lower()

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]

# Crops covered by the model (used for labels and the crop guide).
SUPPORTED_CROPS = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn",
    "Grape",
    "Orange",
    "Peach",
    "Pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]

DISCLAIMER = (
    "AgroScan is an AI screening aid, not an agricultural diagnosis. "
    "Consult a qualified agricultural extension officer before applying "
    "pesticides or making economic decisions, especially when symptoms are "
    "spreading or the crop is valuable."
)
