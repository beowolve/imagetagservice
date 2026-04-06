"""
RAM model loading and inference.

Model file must be placed at:  models/ram_swin_large_14m.pth

Download from HuggingFace:
  https://huggingface.co/xinyu1205/recognize_anything_model/resolve/main/ram_swin_large_14m.pth
"""

from __future__ import annotations

import functools
import logging
import sys
from typing import Optional

import torch
from PIL import Image

from config import (
    CACHE_SIZE,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MODEL_PATH,
)

logger = logging.getLogger(__name__)

_model = None
_transform = None
_device: str = "cpu"


def load_model() -> None:
    """Load the RAM model into memory (call once at startup)."""
    global _model, _transform, _device

    if _model is not None:
        return  # already loaded

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RAM model not found at '{MODEL_PATH}'.\n"
            "Download it from:\n"
            "  https://huggingface.co/xinyu1205/recognize_anything_model/resolve/main/ram_swin_large_14m.pth\n"
            f"and place it in the '{MODEL_PATH.parent}' directory."
        )

    try:
        from ram import get_transform
        from ram.models import ram
    except ImportError as exc:
        raise ImportError(
            "The 'ram' package is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/xinyu1205/recognize-anything.git"
        ) from exc

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading RAM model on %s from %s …", _device, MODEL_PATH)

    import io, os
    _transform = get_transform(image_size=IMAGE_SIZE)
    # RAM prints internal debug info to stdout during loading — suppress it
    _devnull = open(os.devnull, "w")
    _old_stdout, _old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = _devnull
    try:
        _model = ram(
            pretrained=str(MODEL_PATH),
            image_size=IMAGE_SIZE,
            vit="swin_l",
        )
    finally:
        sys.stdout, sys.stderr = _old_stdout, _old_stderr
        _devnull.close()
    # generate_tag() ignores its threshold parameter and always uses
    # model.class_threshold — so we set it here after loading.
    _model.class_threshold = torch.ones(_model.num_class) * CONFIDENCE_THRESHOLD
    _model.eval().to(_device)
    logger.info("RAM model loaded successfully.")


# ---------------------------------------------------------------------------
# Internal cached inference helper
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=CACHE_SIZE)
def _cached_inference(image_hash: str, known_tags_key: Optional[tuple]) -> dict:
    """
    Not called directly – see tag_image().
    image_hash and known_tags_key are used as cache keys only;
    the actual image tensor is re-computed inside tag_image() which holds
    the bytes reference.  This cache is keyed on the hash so identical
    uploads never hit the model twice.
    """
    # This function body is intentionally empty – the real work is done in
    # tag_image() which stores its result here via _store_result().
    raise RuntimeError("Should never be called directly.")


# We can't pass tensors/PIL images to lru_cache, so we use a dict-based cache
# alongside a separate lru_cache for eviction ordering.

_result_cache: dict[tuple, dict] = {}
_cache_order: list[tuple] = []


def _cache_get(key: tuple) -> Optional[dict]:
    return _result_cache.get(key)


def _cache_set(key: tuple, value: dict) -> None:
    if key not in _result_cache:
        _cache_order.append(key)
        if len(_cache_order) > CACHE_SIZE:
            evict = _cache_order.pop(0)
            _result_cache.pop(evict, None)
    _result_cache[key] = value


# ---------------------------------------------------------------------------
# Tag matching
# ---------------------------------------------------------------------------

def _match_tags(
    all_tags: list[str],
    known_tags: list[str],
) -> tuple[list[str], list[str]]:
    """
    Returns (matched_known_tags, suggested_new_tags).

    A known tag is "matched" when it appears (case-insensitive substring) in
    any RAM tag, OR when any RAM tag appears inside a known tag.
    suggested_new_tags are RAM tags that did not match any known tag.
    """
    all_lower = [t.lower() for t in all_tags]
    known_lower = [t.lower() for t in known_tags]

    matched_known: list[str] = []
    unmatched_ram: list[str] = []

    for ram_tag, ram_low in zip(all_tags, all_lower):
        matched = False
        for k_orig, k_low in zip(known_tags, known_lower):
            if k_low in ram_low or ram_low in k_low:
                if k_orig not in matched_known:
                    matched_known.append(k_orig)
                matched = True
        if not matched:
            unmatched_ram.append(ram_tag)

    return matched_known, unmatched_ram


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tag_image(
    image: Image.Image,
    known_tags: Optional[list[str]] = None,
) -> dict:
    """
    Run RAM inference on a PIL image.

    Returns:
        {
          "matched_tags":   [...],   # known tags found in the image
          "suggested_tags": [...],   # RAM tags not in the known list
          "all_tags":       [...],   # every tag RAM returned
        }

    If known_tags is None/empty, matched_tags is [] and all RAM tags are
    returned as suggested_tags.
    """
    if _model is None or _transform is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Build cache key
    known_key = tuple(sorted(known_tags)) if known_tags else None

    # We need a stable image identity for the cache.  The caller (app.py /
    # cli.py) should pass image bytes to image_hash(); here we accept a PIL
    # image so we serialise it lazily only on a cache miss.
    # The cache key is injected from outside via _tag_image_cached().
    return _tag_image_impl(image, known_key)


def tag_image_with_hash(
    image: Image.Image,
    image_hash_str: str,
    known_tags: Optional[list[str]] = None,
) -> dict:
    """Same as tag_image() but uses image_hash_str for cache lookup."""
    known_key = tuple(sorted(known_tags)) if known_tags else None
    cache_key = (image_hash_str, known_key)

    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    result = _tag_image_impl(image, known_key)
    _cache_set(cache_key, result)
    return result


def _tag_image_impl(image: Image.Image, known_key: Optional[tuple]) -> dict:
    try:
        image_tensor = _transform(image).unsqueeze(0).to(_device)

        with torch.no_grad():
            tags_en, _tags_cn = _model.generate_tag(image_tensor)

        # generate_tag returns a list with one entry per batch item.
        # Each entry is a " | "-separated string of tag names.
        raw = tags_en[0] if tags_en else ""
        all_tags = [t.strip() for t in raw.split("|") if t.strip()]

        if known_key:
            matched, suggested = _match_tags(all_tags, list(known_key))
        else:
            matched, suggested = [], all_tags

        return {
            "matched_tags": matched,
            "suggested_tags": suggested,
            "all_tags": all_tags,
        }

    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise RuntimeError(f"Inference failed: {exc}") from exc
