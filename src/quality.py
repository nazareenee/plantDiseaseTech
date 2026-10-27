"""Lightweight image-quality checks.

These heuristics are deliberately conservative: they only warn about obvious
problems (very dark, washed-out, soft or extremely wide images).  They do not
gate prediction, because a blurry image can still be worth screening.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps


@dataclass
class ImageQuality:
    width: int
    height: int
    aspect_ratio: float
    brightness: float
    contrast: float
    sharpness: float
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clear(self) -> bool:
        return not self.warnings


def assess_image_quality(image: Image.Image) -> ImageQuality:
    image = ImageOps.exif_transpose(image)
    width, height = image.size
    aspect_ratio = width / height if height else 0.0

    gray = image.convert("L").resize((320, 320), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32)
    brightness = float(pixels.mean())
    contrast = float(pixels.std())

    # Variance of a 4-neighbour discrete Laplacian: low values mean few edges.
    laplacian = (
        -4.0 * pixels
        + np.roll(pixels, 1, axis=0)
        + np.roll(pixels, -1, axis=0)
        + np.roll(pixels, 1, axis=1)
        + np.roll(pixels, -1, axis=1)
    )
    sharpness = float(laplacian.var())

    warnings: list[str] = []
    if brightness < 38:
        warnings.append("The photo looks dark; use brighter, even daylight.")
    elif brightness > 238:
        warnings.append("The photo looks washed out; avoid direct glare or flash.")
    if contrast < 12:
        warnings.append("Contrast is low; make sure the leaf is clearly separated from the background.")
    if sharpness < 40:
        warnings.append("The photo may be soft or blurry; hold the camera steady and focus on the leaf.")
    if aspect_ratio > 2.2 or aspect_ratio < 0.45:
        warnings.append("The frame is very wide or tall; crop to a single leaf for best results.")
    if min(width, height) < 180:
        warnings.append("The image is small; a higher-resolution photo will preserve more detail.")

    return ImageQuality(
        width=width,
        height=height,
        aspect_ratio=round(aspect_ratio, 2),
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        sharpness=round(sharpness, 1),
        warnings=warnings,
    )
