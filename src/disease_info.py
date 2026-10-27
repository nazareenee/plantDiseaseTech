"""Look-ups for the extension-style disease knowledge base."""

from __future__ import annotations

from functools import lru_cache

from .artifacts import load_disease_info, load_labels
from .labels import ParsedLabel, parse_label


@lru_cache(maxsize=1)
def class_info() -> dict[str, dict]:
    return load_disease_info()


def get_info(label: str) -> dict:
    """Return guidance for a canonical label, with a safe fallback."""
    return class_info().get(
        label,
        {
            "category": "Unclassified",
            "symptoms": "No reference description is available for this class.",
            "management": "Consult a local agricultural extension officer.",
            "prevention": "Practise good crop sanitation and monitor the field.",
        },
    )


@lru_cache(maxsize=1)
def by_crop() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for label in load_labels():
        parsed = parse_label(label)
        if parsed.is_unknown:
            continue
        grouped.setdefault(parsed.crop_display, []).append(label)
    for labels in grouped.values():
        labels.sort(key=lambda value: (parse_label(value).is_healthy, value))
    return dict(sorted(grouped.items()))


def disease_count() -> int:
    return sum(1 for label in load_labels() if not parse_label(label).is_healthy)


def healthy_count() -> int:
    return sum(1 for label in load_labels() if parse_label(label).is_healthy)


def resolved_info(label: str) -> tuple[ParsedLabel, dict]:
    return parse_label(label), get_info(label)
