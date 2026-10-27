"""Label parsing and confidence triage.

The checkpoint stores canonical PlantVillage labels such as
``Tomato___Late_blight``.  This module turns them into human-friendly crop and
condition names and implements the confidence bands published in the model
card.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import CONFIDENCE_LIKELY, CONFIDENCE_UNCLEAR

UNKNOWN_CLASS = "Non_leaf_or_unknown"

CROP_DISPLAY = {
    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Cherry_(including_sour)": "Cherry (incl. sour)",
    "Corn_(maize)": "Corn (maize)",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Pepper": "Pepper (bell)",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Squash": "Squash",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}

CONDITION_DISPLAY = {
    "Apple_scab": "Apple scab",
    "Black_rot": "Black rot",
    "Cedar_apple_rust": "Cedar apple rust",
    "healthy": "Healthy",
    "Powdery_mildew": "Powdery mildew",
    "Cercospora_leaf_spot_Gray_leaf_spot": "Cercospora / gray leaf spot",
    "Common_rust": "Common rust",
    "Northern_Leaf_Blight": "Northern leaf blight",
    "Esca_(Black_Measles)": "Esca (black measles)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf blight (Isariopsis leaf spot)",
    "Non_leaf_or_unknown": "Non-leaf or unknown",
    "Haunglongbing_(Citrus_greening)": "Huanglongbing (citrus greening)",
    "Bacterial_spot": "Bacterial spot",
    "Early_blight": "Early blight",
    "Late_blight": "Late blight",
    "Leaf_Mold": "Leaf mold",
    "Septoria_leaf_spot": "Septoria leaf spot",
    "Spider_mites_Two-spotted_spider_mite": "Spider mites (two-spotted)",
    "Target_Spot": "Target spot",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato yellow leaf curl virus",
    "Tomato_mosaic_virus": "Tomato mosaic virus",
    "Leaf_scorch": "Leaf scorch",
}


@dataclass(frozen=True)
class ParsedLabel:
    """Structured view of a canonical label."""

    canonical: str
    crop: str
    crop_display: str
    condition: str
    condition_display: str
    is_healthy: bool
    is_unknown: bool

    @property
    def display(self) -> str:
        if self.is_unknown:
            return self.condition_display
        return f"{self.crop_display} · {self.condition_display}"


@dataclass(frozen=True)
class Triage:
    """Interpretation of a prediction under the published thresholds."""

    level: str  # "likely", "possible", "unclear" or "out_of_scope"
    title: str
    message: str
    needs_expert: bool
    tone: str  # "danger", "warning", "success" or "neutral"


def parse_label(label: str) -> ParsedLabel:
    """Parse ``Crop___Condition`` without losing the canonical string."""
    if label == UNKNOWN_CLASS:
        return ParsedLabel(
            canonical=label,
            crop="Unknown",
            crop_display="Not a supported leaf",
            condition="Non_leaf_or_unknown",
            condition_display="Non-leaf or unknown",
            is_healthy=False,
            is_unknown=True,
        )
    if "___" not in label:
        raise ValueError(f"Unexpected label format: {label!r}")
    crop, condition = label.split("___", 1)
    crop_display = CROP_DISPLAY.get(crop, crop.replace("_", " "))
    condition_display = CONDITION_DISPLAY.get(condition, condition.replace("_", " ").strip())
    return ParsedLabel(
        canonical=label,
        crop=crop,
        crop_display=crop_display,
        condition=condition,
        condition_display=condition_display,
        is_healthy=condition == "healthy",
        is_unknown=condition == "Non_leaf_or_unknown",
    )


def confidence_triage(confidence: float) -> str:
    """Return the confidence band: ``likely``, ``possible`` or ``unclear``."""
    if confidence >= CONFIDENCE_LIKELY:
        return "likely"
    if confidence >= CONFIDENCE_UNCLEAR:
        return "possible"
    return "unclear"


def triage_prediction(label: str, confidence: float) -> Triage:
    """Translate a model output into a cautious, user-facing status.

    The model card defines confidence bands only.  We additionally handle the
    healthy and out-of-scope classes so that a high-confidence healthy leaf is
    not described as a "likely disease".
    """
    parsed = parse_label(label)
    band = confidence_triage(confidence)

    if parsed.is_unknown:
        return Triage(
            level="out_of_scope",
            title="Not a supported leaf",
            message=(
                "The model does not see a clear close-up of a supported leaf. "
                "Retake the photo with one leaf filling most of the frame, in "
                "even daylight, or consult an extension officer."
            ),
            needs_expert=True,
            tone="neutral",
        )

    if parsed.is_healthy and band != "unclear":
        return Triage(
            level=band,
            title="Likely healthy" if band == "likely" else "Possibly healthy",
            message=(
                "No disease pattern was detected in this image. Keep monitoring "
                "the crop and re-check leaves that develop new symptoms."
            ),
            needs_expert=False,
            tone="success",
        )

    if band == "likely":
        return Triage(
            level=band,
            title="Likely disease detected",
            message=(
                "The model's confidence is high. Confirm the symptoms against "
                "the description below and consult an agricultural expert before "
                "treating the crop."
            ),
            needs_expert=False,
            tone="danger" if not parsed.is_healthy else "success",
        )
    if band == "possible":
        return Triage(
            level=band,
            title="Possible disease detected",
            message=(
                "The model is not confident enough for a strong call. Take a "
                "sharper close-up and ask an agricultural expert to confirm."
            ),
            needs_expert=True,
            tone="warning",
        )
    return Triage(
        level=band,
        title="Unknown or unclear",
        message=(
            "Confidence is below 70%. The image may be blurry, poorly lit, a "
            "different crop, or outside the training distribution. Retake the "
            "photo or consult an expert."
        ),
        needs_expert=True,
        tone="neutral",
    )
