# image_worker.py
import os
import re
import uuid
from io import BytesIO
from PIL import Image
from google import genai
from nano_banana_2 import NanoBanana2
from config import CURRENT_DATE

STOCK_DIR = "stock_photos"
os.makedirs(STOCK_DIR, exist_ok=True)


def crop_to_16_9(img: Image.Image) -> Image.Image:
    """Center-crops any image to exactly 16:9 and forces a 1920x1080 resolution so gradient compositing math covers edge-to-edge flawlessly."""
    w, h = img.size
    target_ratio = 16 / 9
    current_ratio = w / h
    
    # Pre-crop dimension matching
    if abs(current_ratio - target_ratio) > 0.01:
        if current_ratio > target_ratio:
            # Image is too wide (e.g., 21:9), crop the left and right sides
            new_w = int(h * target_ratio)
            offset = (w - new_w) // 2
            img = img.crop((offset, 0, offset + new_w, h))
        else:
            # Image is too tall (e.g., Portrait/Square), crop the top and bottom
            new_h = int(w / target_ratio)
            offset = (h - new_h) // 2
            img = img.crop((0, offset, w, offset + new_h))
            
    # Guarantee backwards compatibility for older Pillow environments
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS

    # By strictly forcing 1920x1080 here, the bounding box overlays in NanoBanana2 
    # will perfectly match up mathematically, solving the gradient seam issues.
    return img.resize((1920, 1080), resample_filter)


def generate_single_image(args):
    """
    Worker for individual AND parallel image generation.

    args: (index, topic_data, config, api_key, theme_labels)
        topic_data   : dict with keys 'topic', 'category', etc.
        config       : dict containing 'meta' (edited texts) and 'img_cfg' (source type/files)
        theme_labels : dict mapping theme key → page-label string.

    Returns: (index, result_dict)
    """
    i, topic_data, config, api_key, theme_labels = args
    thread_client = genai.Client(api_key=api_key)

    meta    = config["meta"]
    img_cfg = config["img_cfg"]

    # ── unpack metadata ───────────────────────────────────────────────────────
    headline  = meta.get("headline",  topic_data["topic"][:50].upper())
    subtext   = meta.get("subtext",   "Read the full post for complete details.")
    date_text = meta.get("date",      CURRENT_DATE)
    scene     = meta.get("scene",     "dramatic urban cityscape at dusk")
    category  = meta.get("category",  topic_data.get("category", "news"))

    imagen_prompt = NanoBanana2.generate_imagen_prompt(headline, scene, category)
    src_type = img_cfg["type"]

    # ── resolve image based on selected source ───────────────────────────────
    if src_type == "AI Generation":
        background = NanoBanana2.generate_background_with_imagen(imagen_prompt, thread_client)
        background = crop_to_16_9(background) # Standardize size first
        
        # Automatically save AI generated image to stock folder
        safe_name = re.sub(r"[^\w\s-]", "", scene).strip().replace(" ", "_")[:40]
        if not safe_name: safe_name = "ai_generated"
        filename = f"{safe_name}_{uuid.uuid4().hex[:6]}.png"
        filepath = os.path.join(STOCK_DIR, filename)
        background.save(filepath)
        
    elif src_type == "Upload" and img_cfg.get("upload_bytes"):
        background = Image.open(BytesIO(img_cfg["upload_bytes"])).convert("RGB")
        background = crop_to_16_9(background)
        
    elif src_type == "Stock / Web" and img_cfg.get("stock_path"):
        try:
            background = Image.open(img_cfg["stock_path"]).convert("RGB")
            background = crop_to_16_9(background)
        except Exception:
            # Failsafe fallback if file was deleted or corrupted
            background = NanoBanana2.generate_background_with_imagen(imagen_prompt, thread_client)
            background = crop_to_16_9(background)
            src_type = "AI Generation (Fallback)"
        
    else:
        # Failsafe fallback to AI
        background = NanoBanana2.generate_background_with_imagen(imagen_prompt, thread_client)
        background = crop_to_16_9(background)
        src_type = "AI Generation (Fallback)"

    # Keep a pure copy of the background to return for preview purposes
    base_bg = background.copy()

    # ── composite three themed variants ──────────────────────────────────────
    images = {
        theme_key: NanoBanana2.composite_by_theme(
            theme_key,
            background,
            headline,
            subtext,
            date_text,
            theme_labels[theme_key]
        )
        for theme_key in ("dark", "light", "black_yellow")
    }

    # Attach base image so the UI can show the uncomposited preview
    images["base_background"] = base_bg

    return i, {
        "topic":        topic_data["topic"],
        "headline":     headline,
        "subtext":      subtext,
        "date":         date_text,
        "scene":        scene,
        "images":       images,
        "imagen_prompt": imagen_prompt,
        "source_type":  src_type
    }