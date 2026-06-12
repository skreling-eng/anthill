"""SigLIP 2 image embedding for $image2embedding."""

from __future__ import annotations

from pathlib import Path

from externals.image2embedding.embedding_format import pack_siglip_embedding
from externals.image2embedding.model_list import Image2EmbeddingModel

_MODEL_CACHE: dict[tuple[str, str, bool], tuple[object, object]] = {}


def _resolve_device(use_gpu: bool) -> str:
    import torch

    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _require_pillow() -> None:
    try:
        import PIL  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "$image2embedding requires Pillow (SiglipImageProcessor).\n"
            "  tools\\setup_external_venvs.ps1   (creates .venvs/media)\n"
            "  or: uv sync --extra image2embedding\n"
            "  or: uv sync --extra media"
        ) from exc


def load_model(
    profile: Image2EmbeddingModel,
    model_dir: Path,
    *,
    use_gpu: bool,
) -> tuple[object, object]:
    key = (profile.name, str(model_dir.resolve()), use_gpu)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    _require_pillow()
    import torch
    from transformers import AutoModel, AutoProcessor

    device = _resolve_device(use_gpu)
    load_kwargs: dict = {}
    if device == "cuda":
        load_kwargs = {"dtype": torch.float16, "device_map": "auto"}
    else:
        load_kwargs = {"dtype": torch.float32}

    print(
        f"$image2embedding: loading {profile.hf_repo} on {device}",
        flush=True,
    )
    model = AutoModel.from_pretrained(str(model_dir), **load_kwargs)
    processor = AutoProcessor.from_pretrained(str(model_dir), use_fast=True)
    if device == "cpu":
        model = model.to(device)
    model.eval()

    _MODEL_CACHE[key] = (model, processor)
    return model, processor


def encode_image(
    profile: Image2EmbeddingModel,
    model: object,
    processor: object,
    image_path: Path,
    *,
    use_gpu: bool,
) -> str:
    import torch
    from PIL import Image

    if use_gpu:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = _resolve_device(use_gpu)
    else:
        device = _resolve_device(use_gpu)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    model_inputs = {
        k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
    }

    with torch.no_grad():
        features = model.get_image_features(**model_inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        vec = features[0].detach().float().cpu().numpy()

    return pack_siglip_embedding(vec, source_dim=profile.source_dim)


def encode_text(
    profile: Image2EmbeddingModel,
    model: object,
    processor: object,
    text: str,
    *,
    use_gpu: bool,
) -> str:
    import torch

    if use_gpu:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = _resolve_device(use_gpu)
    else:
        device = _resolve_device(use_gpu)

    inputs = processor(
        text=[text],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    model_inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
        if k in ("input_ids", "attention_mask")
    }

    with torch.no_grad():
        features = model.get_text_features(**model_inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        vec = features[0].detach().float().cpu().numpy()

    return pack_siglip_embedding(vec, source_dim=profile.source_dim)


def _model_device(model: object, *, use_gpu: bool):
    if use_gpu:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return _resolve_device(use_gpu)


def image_feature_vectors(
    profile: Image2EmbeddingModel,
    model: object,
    processor: object,
    images: list[object],
    *,
    use_gpu: bool,
) -> list:
    """Return L2-normalized SigLIP vectors for a batch of PIL RGB images."""
    import torch

    if not images:
        return []

    device = _model_device(model, use_gpu=use_gpu)
    inputs = processor(images=images, return_tensors="pt")
    model_inputs = {
        k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
    }

    with torch.no_grad():
        features = model.get_image_features(**model_inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        arr = features.detach().float().cpu().numpy()

    return [arr[i] for i in range(arr.shape[0])]


def encode_pil_images_averaged(
    profile: Image2EmbeddingModel,
    model: object,
    processor: object,
    images: list[object],
    *,
    use_gpu: bool,
) -> str:
    from externals.image2embedding.embedding_format import pack_averaged_siglip_embeddings

    vecs = image_feature_vectors(profile, model, processor, images, use_gpu=use_gpu)
    return pack_averaged_siglip_embeddings(vecs, source_dim=profile.source_dim)
