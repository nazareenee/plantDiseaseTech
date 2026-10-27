"""Cached access to the model card artifacts shipped in ``assets/``."""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import pandas as pd

from .config import (
    ASSETS_DIR,
    CLASSIFICATION_REPORT_PATH,
    CLASS_DISTRIBUTION_PATH,
    DISEASE_INFO_PATH,
    LABELS_PATH,
    MANIFEST_PATH,
    METRICS_PATH,
    SAMPLE_DIR,
    TEST_METRICS_PATH,
    TRAINING_HISTORY_PATH,
)

DEMO_IMAGES = [
    {
        "file": "tomato_late_blight.jpg",
        "label": "Tomato___Late_blight",
        "caption": "Tomato · late blight",
    },
    {
        "file": "tomato_healthy.jpg",
        "label": "Tomato___healthy",
        "caption": "Tomato · healthy",
    },
    {
        "file": "apple_scab.jpg",
        "label": "Apple___Apple_scab",
        "caption": "Apple · apple scab",
    },
    {
        "file": "grape_black_rot.jpg",
        "label": "Grape___Black_rot",
        "caption": "Grape · black rot",
    },
    {
        "file": "corn_common_rust.jpg",
        "label": "Corn_(maize)___Common_rust",
        "caption": "Corn · common rust",
    },
    {
        "file": "potato_early_blight.jpg",
        "label": "Potato___Early_blight",
        "caption": "Potato · early blight",
    },
]


@lru_cache(maxsize=1)
def load_labels() -> list[str]:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_test_metrics() -> dict:
    return json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_training_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_training_history() -> pd.DataFrame:
    return pd.read_csv(TRAINING_HISTORY_PATH)


@lru_cache(maxsize=1)
def load_class_distribution() -> pd.DataFrame:
    return pd.read_csv(CLASS_DISTRIBUTION_PATH)


@lru_cache(maxsize=1)
def load_classification_report() -> pd.DataFrame:
    raw = json.loads(CLASSIFICATION_REPORT_PATH.read_text(encoding="utf-8"))
    rows = [
        {"class_name": name, **values}
        for name, values in raw.items()
        if isinstance(values, dict) and name not in {"macro avg", "weighted avg"}
    ]
    df = pd.DataFrame(rows).rename(
        columns={
            "precision": "precision",
            "recall": "recall",
            "f1-score": "f1",
            "support": "support",
        }
    )
    return df[["class_name", "precision", "recall", "f1", "support"]]


@lru_cache(maxsize=1)
def load_disease_info() -> dict[str, dict]:
    raw = json.loads(DISEASE_INFO_PATH.read_text(encoding="utf-8"))
    return {key: value for key, value in raw.items() if not key.startswith("_")}


def demo_options() -> dict[str, Path]:
    """Return ``caption -> path`` for the bundled demonstration images."""
    return {
        item["caption"]: SAMPLE_DIR / item["file"]
        for item in DEMO_IMAGES
        if (SAMPLE_DIR / item["file"]).exists()
    }


@lru_cache(maxsize=1)
def demo_zip() -> bytes:
    """Bundle the demo images so reviewers can try the app offline."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in DEMO_IMAGES:
            path = SAMPLE_DIR / item["file"]
            if path.exists():
                archive.write(path, arcname=path.name)
        readme = SAMPLE_DIR / "README.md"
        if readme.exists():
            archive.write(readme, arcname="README.md")
    return buffer.getvalue()


def artifact_inventory() -> list[dict[str, str]]:
    """Small inventory used on the model page and in smoke tests."""
    files = [
        ("Class labels", LABELS_PATH),
        ("Model manifest", MANIFEST_PATH),
        ("Test metrics", TEST_METRICS_PATH),
        ("Training history", TRAINING_HISTORY_PATH),
        ("Class distribution", CLASS_DISTRIBUTION_PATH),
        ("Classification report", CLASSIFICATION_REPORT_PATH),
        ("Disease knowledge base", DISEASE_INFO_PATH),
    ]
    return [
        {"Artifact": name, "File": path.name, "Available": "yes" if path.exists() else "no"}
        for name, path in files
    ]


def assets_present() -> bool:
    return ASSETS_DIR.exists()
