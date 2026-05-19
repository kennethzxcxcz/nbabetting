# nano_banana_2.py
from PIL import Image, ImageDraw, ImageFont
import textwrap
from io import BytesIO
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class NanoBanana2:
    """Generate photorealistic news backgrounds and composite banners in three visually distinct themes."""

    @staticmethod
    @retry(
        stop=stop_after_attempt(6), # Lowered from 15: If it fails 6 times, it's a safety block. Better to fail fast so you can just upload a stock photo!
        wait=wait_exponential(multiplier=2, min=5, max=40),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def generate_background_with_imagen(image_prompt: str, client) -> Image.Image:
        """Call gemini-3.1-flash-image-preview, fallback to gemini-2.5-flash-image flawlessly."""
        try:
            # ── PRIMARY MODEL ──
            result = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=image_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="MINIMAL",
                    ),
                    image_config=types.ImageConfig(
                        aspect_ratio="16:9",
                        image_size="512",
                    ),
                    response_modalities=["IMAGE"],
                ),
            )
            
            # Safe extraction
            if result and getattr(result, "parts", None):
                for part in result.parts:
                    if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                        return Image.open(BytesIO(part.inline_data.data)).convert("RGB")
                        
            raise Exception("Primary model returned no valid image data (Possible safety block).")
            
        except Exception as e:
            print(f"Primary API error: {e}. Falling back to gemini-2.5-flash-image...")
            
            # ── FALLBACK MODEL ──
            fallback_response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=image_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="16:9"),
                ),
            )
            
            # Flawless safe extraction (Prevents TypeError if response is blocked)
            if fallback_response and getattr(fallback_response, "parts", None):
                for part in fallback_response.parts:
                    if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                        return Image.open(BytesIO(part.inline_data.data)).convert("RGB")
                        
            raise Exception(f"Fallback model failed to return an image (Likely safety block). Original error: {e}")

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
        for p in (path, "Arial.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
        # Fallback safeguard
        font = ImageFont.load_default()
        font.size = size 
        return font

    @staticmethod
    def _wrap_text(draw, text: str, font, max_width: int):
        """Safely wrap text by exact pixel length."""
        words = text.replace('\n', ' ').split()
        if not words:
            return []
        lines = []
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + " " + word
            if draw.textlength(test_line, font=font) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines

    @staticmethod
    def _shadow_text(draw, xy, text, font, fill,
                     shadow=(0, 0, 0, 180), offset=(2, 3), anchor="la"):
        """Draw text with a drop shadow using anchor."""
        draw.text((xy[0] + offset[0], xy[1] + offset[1]), text, font=font, fill=shadow, anchor=anchor)
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)

    @staticmethod
    def _vgrad(draw, x0, y0, x1, y1, c0, c1):
        """Vertical gradient from color c0 (top) to c1 (bottom) with eased transition."""
        h = y1 - y0
        if h <= 0:
            return
        for i in range(h):
            t = i / h
            t_color = t ** 1.5  # Eased alpha transition for perfect blending without harsh clash
            r = int(c0[0] + (c1[0] - c0[0]) * t_color)
            g = int(c0[1] + (c1[1] - c0[1]) * t_color)
            b = int(c0[2] + (c1[2] - c0[2]) * t_color)
            a = int(c0[3] + (c1[3] - c0[3]) * t_color)
            
            # FIX: We use draw.rectangle and overshoot x0/x1 by 10 pixels to completely 
            # bleed off the edges. This strictly solves the antialiasing/edge gap problem in Pillow.
            draw.rectangle([(x0 - 10, y0 + i), (x1 + 10, y0 + i)], fill=(r, g, b, a))

    # ── DARK THEME — Cinematic Noir ──────────────────────────────────────────
    @staticmethod
    def composite_dark_theme(
        background: Image.Image,
        headline: str,
        subtext: str,
        date_text: str,
        source_label: str = "NOW, TODAY"
    ) -> Image.Image:
        # Scale 16:9 image to perfectly match 1080 width and paste at the top
        W, H = 1080, 1080
        bg_w, bg_h = background.size
        new_h = int(W * bg_h / bg_w)
        
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
            
        img_resized = background.resize((W, new_h), resample)
        
        base_color = (8, 8, 14, 255)
        img = Image.new("RGBA", (W, H), base_color)
        img.paste(img_resized, (0, 0))
        
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)
        
        # Load fonts
        font_lbl = NanoBanana2._font("arialbd.ttf", 26)
        font_h = NanoBanana2._font("arialbd.ttf", 56)
        font_s = NanoBanana2._font("arial.ttf", 32)
        font_dt = NanoBanana2._font("arialbd.ttf", 24)

        # Dynamic layout bounds mapping
        max_w = 930
        hl_lines = NanoBanana2._wrap_text(draw, headline.upper(), font_h, max_w)[:4]
        st_lines = NanoBanana2._wrap_text(draw, subtext, font_s, max_w)[:3]

        h_pill = getattr(font_lbl, "size", 26) + 20
        h_space_pill_rule = 16
        h_rule = 2
        h_space_rule_hl = 20
        h_hl = len(hl_lines) * int(getattr(font_h, "size", 56) * 1.25)
        h_space_hl_st = 16
        h_st = len(st_lines) * int(getattr(font_s, "size", 32) * 1.35)

        total_h = h_pill + h_space_pill_rule + h_rule + h_space_rule_hl + h_hl + h_space_hl_st + h_st
        start_y = H - 60 - total_h # Anchor securely to bottom

        # Background Gradient dynamically conforming to perfectly hide and blend the image edge
        fade_end = int(min(max(0, start_y - 40), new_h))
        fade_start = int(max(0, fade_end - 160))
        
        # Applying the bled vgrad mathematically to lock side-to-side without gaps
        NanoBanana2._vgrad(draw, 0, fade_start, W, fade_end, c0=(8, 8, 14, 0), c1=base_color)
        
        # Using strict rectangle boundaries offset by -10 to force the block to bleed past the canvas
        draw.rectangle([(-10, fade_end), (W + 10, H + 10)], fill=base_color)

        # Top Hood & Vignette
        for y in range(220):
            ratio = y / 220.0
            a = int(160 * (1.0 - ratio))
            # Offset by -10 right/left to force bleed
            draw.rectangle([(-10, y), (W + 10, y)], fill=(0, 0, 0, a))
            
        # Increase the pixel width (e.g., to 200) to extend the left & right gradients further inward
        GRADIENT_WIDTH = 20 
        
        for i in range(GRADIENT_WIDTH):
            a = int(120 * (1 - i / float(GRADIENT_WIDTH)) ** 1.5)
            # Offset by -10 top/bottom to force bleed
            draw.rectangle([(i, -10), (i, H + 10)], fill=(0, 0, 0, a))
            draw.rectangle([(W - i, -10), (W - i, H + 10)], fill=(0, 0, 0, a))

        # Date Badge (top right)
        ds = date_text.upper()
        if hasattr(draw, "textlength"):
            badge_w = draw.textlength(ds, font=font_dt) + 32
        else:
            badge_w = draw.textsize(ds, font=font_dt)[0] + 32
            
        badge_h = getattr(font_dt, "size", 24) + 18
        badge_x = W - 40 - badge_w
        badge_y = 40
        draw.rectangle([(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)], fill=(210, 28, 28, 255))
        draw.text((badge_x + badge_w / 2.0, badge_y + badge_h / 2.0), ds, font=font_dt, fill=(255, 255, 255, 255), anchor="mm")

        # Top-down composite of the text block
        curr_y = start_y
        
        # Source Pill
        lbl = source_label.upper()
        if hasattr(draw, "textlength"):
            pill_text_w = draw.textlength(lbl, font=font_lbl)
        else:
            pill_text_w = draw.textsize(lbl, font=font_lbl)[0]
        
        # Box stretches to the left edge (0) but let's bleed it just in case
        draw.rectangle([(-10, curr_y), (75 + pill_text_w + 24, curr_y + h_pill)], fill=(210, 28, 28, 255))
        draw.text((75, curr_y + h_pill / 2.0), lbl, font=font_lbl, fill=(255, 255, 255, 255), anchor="lm")
        curr_y += h_pill + h_space_pill_rule

        # Horizontal Rule (Connects directly to the left vertical stripe at 40)
        draw.rectangle([(40, curr_y), (W - 60, curr_y + h_rule)], fill=(210, 28, 28, 255))
        curr_y += h_rule + h_space_rule_hl

        # Headline
        for line in hl_lines:
            NanoBanana2._shadow_text(draw, (75, curr_y), line, font_h, fill=(255, 212, 0, 255), shadow=(0, 0, 0, 220), offset=(3, 3), anchor="lt")
            curr_y += int(getattr(font_h, "size", 56) * 1.25)
        curr_y += h_space_hl_st

        # Subtext
        for line in st_lines:
            draw.text((75, curr_y), line, font=font_s, fill=(210, 210, 215, 255), anchor="lt")
            curr_y += int(getattr(font_s, "size", 32) * 1.35)

        # Left Vertical Stripe mapping to exact block bounds
        draw.rectangle([(40, start_y), (48, curr_y)], fill=(210, 28, 28, 255))

        out = Image.alpha_composite(img.convert("RGBA"), ov)
        return out.convert("RGB")

    # ── LIGHT THEME — Editorial Clean ────────────────────────────────────────
    @staticmethod
    def composite_light_theme(
        background: Image.Image,
        headline: str,
        subtext: str,
        date_text: str,
        source_label: str = "THE BRIEF"
    ) -> Image.Image:
        bg_w, bg_h = background.size
        new_h = int(1080 * bg_h / bg_w)
        img_resized = background.resize((1080, new_h), Image.LANCZOS)
        
        base_color = (245, 245, 248, 255)
        img = Image.new("RGBA", (1080, 1080), base_color)
        img.paste(img_resized, (0, 0))
        
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)

        # Load fonts
        font_lbl = NanoBanana2._font("arialbd.ttf", 26)
        font_dt_sm = NanoBanana2._font("arial.ttf", 20)
        font_h = NanoBanana2._font("arialbd.ttf", 54)
        font_s = NanoBanana2._font("arial.ttf", 30)

        # Dynamic layout bounds mapping
        max_w = 940
        hl_lines = NanoBanana2._wrap_text(draw, headline.upper(), font_h, max_w)[:4]
        st_lines = NanoBanana2._wrap_text(draw, subtext, font_s, max_w)[:3]

        h_source = getattr(font_lbl, "size", 26)
        h_date = getattr(font_dt_sm, "size", 20)
        h_rule = 2
        h_hl = len(hl_lines) * int(getattr(font_h, "size", 54) * 1.25)
        h_st = len(st_lines) * int(getattr(font_s, "size", 30) * 1.35)

        total_h = h_source + 6 + h_date + 24 + h_rule + 20 + h_hl + 16 + h_st
        start_y = 1080 - 60 - total_h

        fade_end = int(min(max(0, start_y - 30), new_h))
        fade_start = int(max(0, fade_end - 150))
        
        NanoBanana2._vgrad(draw, 0, fade_start, 1080, fade_end, c0=(245, 245, 248, 0), c1=base_color)
        # Bleed edge block
        draw.rectangle([(-10, fade_end), (1080 + 10, 1080 + 10)], fill=base_color)
        
        # Dark Hood to contrast image peak
        NanoBanana2._vgrad(draw, 0, 0, 1080, 200, c0=(0, 0, 0, 120), c1=(0, 0, 0, 0))

        # Top-down composite of the text block
        curr_y = start_y
        
        draw.text((80, curr_y), source_label.upper(), font=font_lbl, fill=(18, 42, 100, 255), anchor="lt")
        curr_y += h_source + 6
        
        draw.text((80, curr_y), date_text.upper(), font=font_dt_sm, fill=(90, 90, 100, 240), anchor="lt")
        curr_y += h_date + 24
        
        # Horizontal Rule connected to the left vertical stripe
        draw.rectangle([(48, curr_y), (1080 - 60, curr_y + h_rule)], fill=(18, 42, 100, 240))
        curr_y += h_rule + 20
        
        for line in hl_lines:
            draw.text((80, curr_y), line, font=font_h, fill=(12, 18, 52, 255), anchor="lt")
            curr_y += int(getattr(font_h, "size", 54) * 1.25)
        curr_y += 16
        
        for line in st_lines:
            draw.text((80, curr_y), line, font=font_s, fill=(60, 60, 70, 255), anchor="lt")
            curr_y += int(getattr(font_s, "size", 30) * 1.35)
            
        # Left Vertical Accent
        draw.rectangle([(48, start_y), (56, curr_y)], fill=(18, 42, 100, 240))

        out = Image.alpha_composite(img.convert("RGBA"), ov)
        return out.convert("RGB")

    # ── BLACK & YELLOW THEME — Tabloid Impact ────────────────────────────────
    @staticmethod
    def composite_black_yellow_theme(
        background: Image.Image,
        headline: str,
        subtext: str,
        date_text: str,
        source_label: str = "RAW TAKE"
    ) -> Image.Image:
        bg_w, bg_h = background.size
        new_h = int(1080 * bg_h / bg_w)
        img_resized = background.resize((1080, new_h), Image.LANCZOS)
        
        base_color = (0, 0, 0, 255)
        img = Image.new("RGBA", (1080, 1080), base_color)
        img.paste(img_resized, (0, 0))
        
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)

        # Load fonts
        font_dt2 = NanoBanana2._font("arialbd.ttf", 24)
        font_lbl = NanoBanana2._font("arialbd.ttf", 36)
        font_h = NanoBanana2._font("arialbd.ttf", 62)
        font_s = NanoBanana2._font("arialbd.ttf", 32)
        
        # Dynamic layout bounds mapping 
        max_w = 930
        hl_lines = NanoBanana2._wrap_text(draw, headline.upper(), font_h, max_w)[:4]
        st_lines = NanoBanana2._wrap_text(draw, subtext, font_s, max_w)[:3]

        h_hl = len(hl_lines) * int(getattr(font_h, "size", 62) * 1.15)
        h_st = len(st_lines) * int(getattr(font_s, "size", 32) * 1.3)
        
        total_h = h_hl + 20 + h_st
        start_y = 1080 - 60 - total_h
        
        # Yellow Ribbon (Bled past edge for 100% borderless fill)
        RH = 90
        SLOPE = 30
        draw.polygon([(-10, 0), (1080 + 10, 0), (1080 + 10, RH - SLOPE), (-10, RH)], fill=(255, 212, 0, 255))
        draw.polygon([(-10, RH), (1080 + 10, RH - SLOPE), (1080 + 10, RH - SLOPE + 6), (-10, RH + 6)], fill=(0, 0, 0, 220))
        
        # Top-Right Ribbon Date Badge
        ds = date_text.upper()
        badge_w = draw.textlength(ds, font=font_dt2) + 32
        badge_h = getattr(font_dt2, "size", 24) + 18
        badge_x = 1080 - 40 - badge_w
        badge_y = 16
        draw.rectangle([(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)], fill=(0, 0, 0, 245))
        draw.text((badge_x + badge_w / 2, badge_y + badge_h / 2), ds, font=font_dt2, fill=(255, 212, 0, 255), anchor="mm")
        
        # Source Label (Left side on ribbon)
        draw.text((40, 26), source_label.upper(), font=font_lbl, fill=(0, 0, 0, 255), anchor="lt")

        fade_end = int(min(max(0, start_y - 40), new_h))
        fade_start = int(max(0, fade_end - 140))
        NanoBanana2._vgrad(draw, 0, fade_start, 1080, fade_end, c0=(0, 0, 0, 0), c1=base_color)
        
        # Bleed edge block
        draw.rectangle([(-10, fade_end), (1080 + 10, 1080 + 10)], fill=base_color)

        # Top-down composite of the text block
        curr_y = start_y
        for line in hl_lines:
            NanoBanana2._shadow_text(draw, (75, curr_y), line, font_h, fill=(255, 255, 255, 255), shadow=(0, 0, 0, 255), offset=(4, 4), anchor="lt")
            curr_y += int(getattr(font_h, "size", 62) * 1.15)
        curr_y += 20
        
        for line in st_lines:
            draw.text((75, curr_y), line, font=font_s, fill=(255, 212, 0, 255), anchor="lt")
            curr_y += int(getattr(font_s, "size", 32) * 1.3)
            
        # Yellow Border Anchor Accent
        draw.rectangle([(40, start_y), (48, curr_y)], fill=(255, 212, 0, 255))

        out = Image.alpha_composite(img.convert("RGBA"), ov)
        return out.convert("RGB")

    # ── dispatcher ───────────────────────────────────────────────────────────

    @staticmethod
    def composite_by_theme(
        theme: str,
        background: Image.Image,
        headline: str,
        subtext: str,
        date_text: str,
        source_label: str
    ) -> Image.Image:
        if theme == "dark":
            return NanoBanana2.composite_dark_theme(
                background, headline, subtext, date_text, source_label)
        elif theme == "light":
            return NanoBanana2.composite_light_theme(
                background, headline, subtext, date_text, source_label)
        elif theme == "black_yellow":
            return NanoBanana2.composite_black_yellow_theme(
                background, headline, subtext, date_text, source_label)
        else:
            raise ValueError(f"Unknown theme: {theme!r}")

    # ── prompt generator ─────────────────────────────────────────────────────

    @staticmethod
    def generate_imagen_prompt(headline: str, subtext: str, category: str = "news") -> str:
        return (
            f"A purely photographic, wide cinematic establishing shot of the scene depicting a suitable background for the headline: {headline} and its subtext: {subtext}. "
            f"Category: {category}. "
            f"Photorealistic, high resolution, dramatic and moody cinematic lighting, 4K quality. "
            f"STRICTLY NO TEXT, NO LOGOS, NO PEOPLE IN THE FOREGROUND. "
            f"Clean composition focusing entirely on the environment and atmosphere. "
            f"All surfaces, walls, billboards, backgrounds, and objects are completely pristine, "
            f"blank, featureless, and untouched. Pure visual space only. "
            f"Any people present must be in the distant background, out of focus, "
            f"with faces completely obscured."
        )