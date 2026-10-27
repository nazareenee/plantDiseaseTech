"""Model loading, inference, Grad-CAM and model metadata.

The AgroScan checkpoint is a wrapped PyTorch dictionary:

``{"model_state_dict": ..., "class_names": ..., "num_classes": ..., ...}``

The architecture is torchvision EfficientNet-B0 with the standard classifier
head replaced by a 39-way linear layer.  No custom layers are required, which
keeps deployment simple and makes the checkpoint easy to audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

from .artifacts import load_labels, load_manifest
from .config import (
    DEVICE_OVERRIDE,
    HF_REPO_ID,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LOCAL_MODEL_PATH,
    MODEL_FILENAME,
    MODELS_DIR,
    RESIZE_SIZE,
)


class ModelLoadError(RuntimeError):
    """Raised when the checkpoint cannot be located or loaded."""


@dataclass
class LoadedModel:
    model: nn.Module
    labels: list[str]
    device: torch.device
    checkpoint_path: Path
    metadata: dict

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def model_size_mb(self) -> float:
        return self.checkpoint_path.stat().st_size / (1024 * 1024)

    @property
    def device_label(self) -> str:
        if self.device.type == "cuda":
            return f"CUDA · {torch.cuda.get_device_name(0)}"
        if self.device.type == "mps":
            return "Apple Metal (MPS)"
        return "CPU"


@dataclass(frozen=True)
class Prediction:
    class_name: str
    confidence: float
    probabilities: np.ndarray
    top_indices: tuple[int, ...]
    top_probabilities: tuple[float, ...]

    def top_k(self, k: int = 3) -> list[tuple[int, float]]:
        return list(zip(self.top_indices[:k], self.top_probabilities[:k], strict=True))


@dataclass(frozen=True)
class GradCAMResult:
    overlay: Image.Image
    heatmap: np.ndarray
    target_index: int


def resolve_device() -> torch.device:
    if DEVICE_OVERRIDE:
        if DEVICE_OVERRIDE.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        if DEVICE_OVERRIDE == "mps" and not _mps_available():
            return torch.device("cpu")
        return torch.device(DEVICE_OVERRIDE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_model_path() -> Path:
    """Prefer the bundled checkpoint, then fall back to the Hugging Face Hub."""
    if LOCAL_MODEL_PATH.exists():
        return LOCAL_MODEL_PATH

    try:
        from huggingface_hub import hf_hub_download

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=MODELS_DIR,
        )
        return Path(downloaded)
    except Exception as exc:  # pragma: no cover - depends on network
        raise ModelLoadError(
            "The AgroScan checkpoint was not found locally and could not be "
            "downloaded from the Hugging Face Hub. Run "
            "`python scripts/download_model.py` once with an internet "
            "connection, or place the .pth file in `models/`."
        ) from exc


def load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.6 has no weights_only argument
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ModelLoadError("The checkpoint did not contain a dictionary.")
    return checkpoint


def _extract_state_dict(checkpoint: dict) -> dict:
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value
    if any(str(key).startswith(("features.", "classifier.")) for key in checkpoint):
        return checkpoint
    raise ModelLoadError("Could not find a state dict in the checkpoint.")


def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: Path | None = None) -> LoadedModel:
    """Load the checkpoint exactly once and prepare it for inference."""
    path = Path(model_path) if model_path else resolve_model_path()
    if not path.exists():
        raise ModelLoadError(f"Checkpoint not found: {path}")

    checkpoint = load_checkpoint(path)
    state_dict = _extract_state_dict(checkpoint)
    checkpoint_labels = checkpoint.get("class_names")
    labels = list(checkpoint_labels) if checkpoint_labels else load_labels()
    num_classes = int(checkpoint.get("num_classes", len(labels)))
    if num_classes != len(labels):
        raise ModelLoadError("Checkpoint class count does not match its label list.")

    head = state_dict.get("classifier.1.weight")
    if head is None or tuple(head.shape) != (num_classes, 1280):
        raise ModelLoadError(
            "Unexpected classifier head in checkpoint; expected "
            f"({num_classes}, 1280), got "
            f"{None if head is None else tuple(head.shape)}."
        )

    model = build_model(num_classes)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(f"Checkpoint does not match EfficientNet-B0: {exc}") from exc

    device = resolve_device()
    model.to(device)
    model.eval()

    manifest = load_manifest()
    mean = tuple(checkpoint.get("normalization_mean", IMAGENET_MEAN))
    std = tuple(checkpoint.get("normalization_std", IMAGENET_STD))
    metadata = {
        "architecture": checkpoint.get("architecture", "efficientnet_b0"),
        "image_size": int(checkpoint.get("image_size", manifest.get("input_size", IMAGE_SIZE))),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_f1": checkpoint.get("best_val_f1"),
        "normalization_mean": mean,
        "normalization_std": std,
        "source": "Hugging Face · " + HF_REPO_ID,
        "license": "MIT",
    }
    return LoadedModel(
        model=model,
        labels=labels,
        device=device,
        checkpoint_path=path,
        metadata=metadata,
    )


def to_rgb(image: Image.Image) -> Image.Image:
    """Normalise EXIF rotation, transparency and colour mode."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")


def build_transform(loaded: LoadedModel | None = None) -> transforms.Compose:
    mean = loaded.metadata["normalization_mean"] if loaded else IMAGENET_MEAN
    std = loaded.metadata["normalization_std"] if loaded else IMAGENET_STD
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE, antialias=True),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def preprocess(image: Image.Image, loaded: LoadedModel | None = None) -> torch.Tensor:
    return build_transform(loaded)(to_rgb(image))


def predict(image: Image.Image, loaded: LoadedModel, top_k: int = 3) -> Prediction:
    tensor = preprocess(image, loaded).unsqueeze(0).to(loaded.device)
    with torch.inference_mode():
        logits = loaded.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].float().cpu().numpy()

    order = np.argsort(probabilities)[::-1]
    top_indices = tuple(int(index) for index in order[:top_k])
    top_probabilities = tuple(float(probabilities[index]) for index in top_indices)
    return Prediction(
        class_name=loaded.labels[int(order[0])],
        confidence=float(probabilities[int(order[0])]),
        probabilities=probabilities,
        top_indices=top_indices,
        top_probabilities=top_probabilities,
    )


def predict_batch(images: list[Image.Image], loaded: LoadedModel, top_k: int = 3) -> list[Prediction]:
    return [predict(image, loaded, top_k=top_k) for image in images]


def _colourise(cam: np.ndarray) -> np.ndarray:
    """Map a 0-1 array to a readable blue → green → yellow → red heatmap."""
    stops = np.array(
        [
            [18, 44, 120],
            [0, 150, 199],
            [60, 190, 90],
            [255, 221, 51],
            [220, 60, 35],
        ],
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, len(stops))
    channels = [np.interp(cam, positions, stops[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def generate_gradcam(
    image: Image.Image,
    loaded: LoadedModel,
    target_index: int | None = None,
) -> GradCAMResult:
    """Generate a Grad-CAM overlay for the predicted (or supplied) class."""
    rgb = to_rgb(image)
    tensor = preprocess(rgb, loaded).unsqueeze(0).to(loaded.device)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output: torch.Tensor) -> None:
        activations.append(output)

        def save_gradient(gradient: torch.Tensor) -> None:
            gradients.append(gradient.detach())

        output.register_hook(save_gradient)

    handle = loaded.model.features.register_forward_hook(forward_hook)
    was_training = loaded.model.training
    try:
        loaded.model.eval()
        loaded.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = loaded.model(tensor)
            index = int(logits.argmax(dim=1).item()) if target_index is None else int(target_index)
            score = logits[0, index]
            score.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations.")

        activation = activations[-1].detach()[0]
        gradient = gradients[-1][0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=0))
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam_np = cam.detach().float().cpu().numpy()
    finally:
        handle.remove()
        loaded.model.zero_grad(set_to_none=True)
        if was_training:
            loaded.model.train()

    low, high = np.percentile(cam_np, [1, 99])
    if high - low < 1e-8:
        cam_np = np.zeros_like(cam_np)
    else:
        cam_np = np.clip((cam_np - low) / (high - low), 0.0, 1.0)

    base = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    base_array = np.asarray(base, dtype=np.float32)
    heat_array = _colourise(cam_np).astype(np.float32)
    alpha = (cam_np**0.8 * 0.62)[..., None]
    blended = base_array * (1.0 - alpha) + heat_array * alpha
    overlay = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
    overlay = overlay.resize(rgb.size, Image.Resampling.LANCZOS)
    return GradCAMResult(overlay=overlay, heatmap=cam_np, target_index=index)


def warm_up(loaded: LoadedModel) -> None:
    """Run one dummy inference so the first real request is not slow."""
    dummy = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (120, 150, 100))
    predict(dummy, loaded)
