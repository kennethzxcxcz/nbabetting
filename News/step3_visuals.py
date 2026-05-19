# step3_visuals.py
import streamlit as st
import os
import requests
import urllib.parse
from google import genai
from google.genai import types
import concurrent.futures
import json
import re
import uuid
from io import BytesIO
from utils import generate_with_retry, extract_json_from_text
from image_worker import generate_single_image
from config import API_KEY
from step2_rewriter import TONE_CONFIG


# Ensure stock directory exists
STOCK_DIR = "stock_photos"
os.makedirs(STOCK_DIR, exist_ok=True)


def build_config_from_state(i):
    """Helper to cleanly extract the current UI widget values for a specific topic."""
    meta = st.session_state["meta_edits"][i]
    hl = st.session_state.get(f"hl_{i}", meta.get("headline", ""))
    st_text = st.session_state.get(f"st_{i}", meta.get("subtext", ""))
    dt = st.session_state.get(f"dt_{i}", meta.get("date", ""))
    sc = st.session_state.get(f"sc_{i}", meta.get("scene", ""))
    cat = meta.get("category", "news")
    
    # Set default image source to Stock / Web
    src_type = st.session_state.get(f"src_{i}", "Stock / Web")
    
    uploaded = st.session_state.get(f"up_{i}")
    up_bytes = uploaded.getvalue() if uploaded else None
    
    selected_stock = st.session_state.get(f"stock_{i}")
    stock_path = os.path.join(STOCK_DIR, selected_stock) if selected_stock and src_type == "Stock / Web" else None
    
    return {
        "meta": {"headline": hl, "subtext": st_text, "date": dt, "scene": sc, "category": cat},
        "img_cfg": {"type": src_type, "upload_bytes": up_bytes, "stock_path": stock_path}
    }


def _search_pixabay(query, count, api_key):
    """Pixabay free API — 5,000 req/hour. Get key at pixabay.com/api/docs/"""
    params = {"key": api_key, "q": query, "image_type": "photo",
              "safesearch": "true", "per_page": min(count, 200), "page": 1}
    resp = requests.get("https://pixabay.com/api/", params=params, timeout=10)
    resp.raise_for_status()
    hits = resp.json().get("hits", [])
    return [h["largeImageURL"] for h in hits if h.get("largeImageURL")][:count]


def _search_pexels(query, count, api_key):
    """Pexels free API — 200 req/hour. Get key at pexels.com/api/"""
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": min(count, 80), "page": 1}
    resp = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    photos = resp.json().get("photos", [])
    return [p["src"]["large"] for p in photos if p.get("src")][:count]


def _search_unsplash(query, count, api_key):
    """Unsplash free API — 50 req/hour. Get key at unsplash.com/developers"""
    params = {"query": query, "per_page": min(count, 30), "page": 1, "client_id": api_key}
    resp = requests.get("https://api.unsplash.com/search/photos", params=params, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [r["urls"]["regular"] for r in results if r.get("urls")][:count]


def _search_duckduckgo(query, count):
    """DuckDuckGo fallback — no key needed but aggressively rate limited."""
    from duckduckgo_search import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.images(keywords=query, region="wt-wt", safesearch="moderate", max_results=count):
            url = r.get("image", "")
            if url.startswith("http"):
                results.append(url)
            if len(results) >= count:
                break
    return results


def scrape_web_images(query, count=12):
    """
    Multi-source fallback chain: Pixabay → Pexels → Unsplash → DuckDuckGo.
    Configure API keys in the sidebar under ⚙️ Image Search API Keys.
    Free signups: pixabay.com/api/docs | pexels.com/api | unsplash.com/developers
    """
    keys = st.session_state.get("img_api_keys", {})
    sources = [
        ("Pixabay",  keys.get("pixabay"),  lambda k: _search_pixabay(query, count, k)),
        ("Pexels",   keys.get("pexels"),   lambda k: _search_pexels(query, count, k)),
        ("Unsplash", keys.get("unsplash"), lambda k: _search_unsplash(query, count, k)),
    ]

    for name, key, fn in sources:
        if not key:
            continue
        try:
            results = fn(key)
            if results:
                print(f"[Image Search] ✅ {name} → {len(results)} results")
                return results
        except Exception as e:
            print(f"[Image Search] ⚠️ {name} failed ({e}) — trying next...")

    # Last resort: DuckDuckGo (no key, but rate limited)
    try:
        results = _search_duckduckgo(query, count)
        if results:
            return results
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "duckduckgo_search", "-q"])
    except Exception as e:
        print(f"[Image Search] ❌ DuckDuckGo also failed: {e}")

    return []


@st.dialog("Select Background Image", width="large")
def image_selector_dialog(i, default_query):
    """Pop-up modal to visually preview and select local stock or scraped web images."""
    tab_stock, tab_web = st.tabs(["🖼️ Stock Photos", "🌐 Web Search"])
    
    with tab_stock:
        stock_files = sorted(
            [f for f in os.listdir(STOCK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))],
            key=lambda x: os.path.getmtime(os.path.join(STOCK_DIR, x)),
            reverse=True
        )
        if not stock_files:
            st.info("No stock photos found. Search the web to add some!")
        else:
            cols = st.columns(3)
            for idx, file in enumerate(stock_files):
                with cols[idx % 3]:
                    st.image(os.path.join(STOCK_DIR, file), use_container_width=True)
                    if st.button("Select", key=f"sel_stk_{i}_{file}"):
                        st.session_state[f"stock_{i}"] = file
                        st.rerun()

    with tab_web:
        col_sq, col_sb = st.columns([3, 1])
        with col_sq:
            search_q = st.text_input("Search query", value=default_query, key=f"sq_{i}", label_visibility="collapsed")
        with col_sb:
            do_search = st.button("Search Web", use_container_width=True, key=f"sb_{i}")
        
        if do_search or f"web_res_{i}" not in st.session_state:
            with st.spinner("Fetching images via DuckDuckGo..."):
                st.session_state[f"web_res_{i}"] = scrape_web_images(search_q if do_search else default_query)
        
        web_results = st.session_state.get(f"web_res_{i}", [])
        if web_results:
            cols = st.columns(3)
            for idx, url in enumerate(web_results):
                with cols[idx % 3]:
                    try:
                        st.image(url, use_container_width=True)
                        if st.button("Select & Download", key=f"sel_web_{i}_{idx}"):
                            with st.spinner("Downloading image..."):
                                resp = requests.get(url, timeout=10)
                                if resp.status_code == 200:
                                    ext = url.split('.')[-1].split('?')[0].lower()
                                    if ext not in ['png', 'jpg', 'jpeg', 'webp']:
                                        ext = 'jpg'
                                    filename = f"web_{uuid.uuid4().hex[:8]}.{ext}"
                                    filepath = os.path.join(STOCK_DIR, filename)
                                    with open(filepath, 'wb') as f:
                                        f.write(resp.content)
                                    # Select the newly downloaded image natively
                                    st.session_state[f"stock_{i}"] = filename
                                    st.rerun()
                    except Exception:
                        st.error("Failed to preview.")


def process_metadata_chunk(client, model_name, chunk_topics):
    """Processes a small chunk of topics using MINIMAL thinking to ensure high speed & fidelity."""
    num_chunk = len(chunk_topics)
    topic_contexts =[]
    for topic_data in chunk_topics:
        topic_contexts.append({
            "topic": topic_data["topic"],
            "posts": {
                tone_key: topic_data["tones"].get(tone_key, "")
                for tone_key in ("tagalog_viral", "straight_news", "satirical")
            }
        })

    batch_prompt = f"""You are a graphic designer creating the single strongest visual headline for {num_chunk} news topics.
Each topic has three tone variants (Tagalog Viral, Straight News, Satirical). Use all three as context to craft the ONE best headline and subtext for a visual news card — it must work powerfully across all three design themes.

Topics and their posts:
{json.dumps([{"topic": t["topic"], "posts": t["posts"]} for t in topic_contexts], indent=2)}
ALL HEADLINES and SUBTEXT MUST STRICTLY BE IN TAGALOG. Use the English posts only as context to understand the topic and craft a compelling Tagalog headline and subtext.
For EACH topic provide ONE JSON object with these keys:
- "headline": ALL CAPS. Bold, punchy, urgent, 10 WORD LENGTH LIMIT, NEEDS TO BE COHERENT.
- "subtext": 12-16 words in English. Informative, never a repeat of the headline.
- "date": Exact date from the posts (format: "MONTH DD, YYYY").
- "scene": 20-word description of a photorealistic scene.
- "category": "Philippine News" or "International News".

Return ONLY a valid JSON array of {num_chunk} objects."""

    meta_resp = generate_with_retry(
        client=client,
        model=model_name,
        contents=batch_prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            system_instruction="You are a graphic designer assistant. Output ONLY a valid JSON array.",
            temperature=1.0
        )
    )
    parsed = extract_json_from_text(meta_resp.text)
    
    if not parsed or not isinstance(parsed, list):
        return [{} for _ in range(num_chunk)]
    
    while len(parsed) < num_chunk:
        parsed.append({})
        
    return parsed[:num_chunk]


def render_step3(client, model_name):
    st.header("Step 3: Nano Banana 2 — Multi‑Theme Visual Generator")
    st.info(
        "Extract metadata, edit text, choose image sources (Stock/Web/AI/Upload), and generate! "
        "You can **generate topics individually** or **batch process** them all at once."
    )

    # ── Image API Key Config (Sidebar) ────────────────────────────────────────
    with st.sidebar.expander("⚙️ Image Search API Keys", expanded=False):
        st.caption("Add free API keys for better image results. Priority: Pixabay → Pexels → Unsplash → DuckDuckGo (fallback).")
        pixabay_key  = "55388190-37b385c0161c87cda0016d883"
        pexels_key   = "Qxb53Dz12EqwaVQqF1El6TtwrTQSEH5zpyqavmGUxiEeSBz9Akh89VdT"
        unsplash_key = "unsplash_key"  # Replace with actual Unsplash key if available
        st.session_state["img_api_keys"] = {
            "pixabay":  pixabay_key.strip()  or None,
            "pexels":   pexels_key.strip()   or None,
            "unsplash": unsplash_key.strip() or None,
        }
        active = [k for k, v in st.session_state["img_api_keys"].items() if v]
        if active:
            st.success(f"Active: {', '.join(active)}")
        else:
            st.warning("No keys set — using DuckDuckGo fallback only.")

    if "fb_posts" not in st.session_state:
        st.warning("⚠️ Complete Step 2 first to generate Facebook Posts.")
        return

    num_topics = len(st.session_state["fb_posts"])

    if "step3_phase" not in st.session_state:
        st.session_state["step3_phase"] = 1

    # ── Page-label inputs (Always visible) ────────────────────────────────────
    st.subheader("🎨 Theme Page Labels")
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        dark_label = st.text_input("Dark Theme (Tagalog Viral)", value=TONE_CONFIG["tagalog_viral"]["default_label"])
    with col_l2:
        light_label = st.text_input("Light Theme (Straight News)", value=TONE_CONFIG["straight_news"]["default_label"])
    with col_l3:
        black_yellow_label = st.text_input("Black & Yellow (Satirical)", value=TONE_CONFIG["satirical"]["default_label"])

    theme_labels = {"dark": dark_label, "light": light_label, "black_yellow": black_yellow_label}

    # ==========================================
    # PHASE 1: Extract Metadata (Chunked by 2)
    # ==========================================
    if st.session_state["step3_phase"] == 1:
        st.subheader("Phase 1: Extract Base Metadata")
        if st.button(f"📥 Extract Initial Headlines & Subtext for {num_topics} Topics", type="primary"):
            with st.status(f"📋 Crafting metadata for {num_topics} topics (Batching 2 per call)…", expanded=True) as status:
                try:
                    all_topics = st.session_state["fb_posts"]
                    chunks = [all_topics[i:i + 2] for i in range(0, num_topics, 2)]
                    results_ordered = [None] * len(chunks)
                    
                    completed = 0
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        future_to_idx = {
                            executor.submit(process_metadata_chunk, client, model_name, chunks[i]): i 
                            for i in range(len(chunks))
                        }
                        
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            try:
                                res = future.result()
                                results_ordered[idx] = res
                            except Exception as e:
                                st.error(f"Chunk {idx+1} failed: {e}")
                                results_ordered[idx] = [{} for _ in range(len(chunks[idx]))]
                            
                            completed += 1
                            status.update(label=f"📋 Processing chunk {completed}/{len(chunks)}...", state="running")
                    
                    all_metadata = []
                    for r in results_ordered:
                        all_metadata.extend(r)
                    
                    all_metadata = all_metadata[:num_topics]
                    while len(all_metadata) < num_topics:
                        all_metadata.append({})

                    st.session_state["meta_edits"] = all_metadata
                    st.session_state["step3_phase"] = 2
                    st.session_state["generated_visuals"] = [None] * num_topics
                    
                    status.update(label="✅ Metadata Extracted!", state="complete")
                    st.rerun()

                except Exception as e:
                    status.update(label=f"❌ Extraction failed: {e}", state="error")

    # ==========================================
    # PHASE 2: Interactive Hub (Edit, Single Gen, Batch Gen)
    # ==========================================
    elif st.session_state["step3_phase"] == 2:
        st.subheader("Phase 2: Dashboard — Edit & Generate")

        col_top1, col_top2, col_top3 = st.columns([1, 2, 1])
        with col_top1:
            if st.button("⬅️ Restart Extraction"):
                st.session_state["step3_phase"] = 1
                if "generated_visuals" in st.session_state: del st.session_state["generated_visuals"]
                st.rerun()
                
        pending_count = sum(1 for v in st.session_state["generated_visuals"] if v is None)
        
        with col_top2:
            if pending_count > 0:
                if st.button(f"🚀 BATCH GENERATE ALL {pending_count} PENDING", type="primary", use_container_width=True):
                    st.info(f"🚀 Batch processing {pending_count} topics...")
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    task_args = []
                    for i in range(num_topics):
                        if st.session_state["generated_visuals"][i] is None:
                            cfg = build_config_from_state(i)
                            task_args.append((i, st.session_state["fb_posts"][i], cfg, API_KEY, theme_labels))
                            
                    completed = 0
                    max_workers = st.session_state.get("batch_workers", 2)
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_idx = {executor.submit(generate_single_image, args): args[0] for args in task_args}
                        for future in concurrent.futures.as_completed(future_to_idx):
                            orig_idx = future_to_idx[future]
                            try:
                                idx, result = future.result()
                                st.session_state["generated_visuals"][idx] = result
                                completed += 1
                                progress_bar.progress(completed / pending_count)
                                progress_text.text(f"✅ {completed}/{pending_count} processed — latest: {result['topic'][:40]}…")
                            except Exception as e:
                                st.error(f"❌ Topic {orig_idx + 1} failed: {e}")
                                completed += 1
                                progress_bar.progress(completed / pending_count)
                                
                    st.success("✅ Batch generation completed!")
                    st.rerun()
        with col_top3:
            st.slider("Parallel workers", 1, 6, 2, key="batch_workers")

        st.divider()

        # Render Each Topic Card
        for i, topic_data in enumerate(st.session_state["fb_posts"]):
            is_generated = st.session_state["generated_visuals"][i] is not None
            border_color = "✅" if is_generated else "⏳"
            
            with st.container(border=True):
                st.markdown(f"#### {border_color} Topic {i+1}: {topic_data['topic']}")
                meta = st.session_state["meta_edits"][i]
                
                c_txt, c_img = st.columns([1.5, 1])
                with c_txt:
                    st.text_input("Headline", value=meta.get("headline", ""), key=f"hl_{i}")
                    st.text_area("Subtext", value=meta.get("subtext", ""), key=f"st_{i}")
                    st.text_input("Date", value=meta.get("date", ""), key=f"dt_{i}")
                    st.text_area("Scene (for AI generation)", value=meta.get("scene", ""), key=f"sc_{i}")
                
                with c_img:
                    # 'Stock / Web' is now the default (Index 0)
                    src_type = st.radio("🖼️ Image Source", ["Stock / Web", "AI Generation", "Upload"], index=0, key=f"src_{i}")
                    
                    if src_type == "Upload":
                        uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"], key=f"up_{i}")
                        if uploaded: st.image(uploaded, width=250, caption="Preview")
                        
                    elif src_type == "Stock / Web":
                        selected_stock = st.session_state.get(f"stock_{i}")
                        if selected_stock and os.path.exists(os.path.join(STOCK_DIR, selected_stock)):
                            st.image(os.path.join(STOCK_DIR, selected_stock), width=250, caption="Selected Background")
                        else:
                            st.info("No image selected.")
                            
                        default_q = meta.get("headline", topic_data["topic"])[:60]
                        if st.button("🔍 Browse Stock & Web", key=f"browse_btn_{i}"):
                            image_selector_dialog(i, default_q)
                            
                    else:
                        st.caption("✨ AI will generate and automatically save the result to the Stock folder.")

                # Individual Action Buttons
                c_btn1, c_btn2 = st.columns([1, 4])
                with c_btn1:
                    btn_label = "🔄 Regenerate" if is_generated else "🎨 Generate Single"
                    if st.button(btn_label, key=f"btn_gen_{i}"):
                        current_cfg = build_config_from_state(i)
                        with st.spinner(f"Generating Visuals for Topic {i+1}..."):
                            args = (i, topic_data, current_cfg, API_KEY, theme_labels)
                            _, result = generate_single_image(args)
                            st.session_state["generated_visuals"][i] = result
                        st.rerun()

                # Display Results Inline if generated
                if is_generated:
                    st.divider()
                    _display_single_result(st.session_state["generated_visuals"][i], i, theme_labels)


def _display_single_result(result, index, theme_labels):
    with st.expander(f"🖼️ View Base / Stock Image Used (Source: {result.get('source_type', 'AI')})", expanded=False):
        st.image(result["images"]["base_background"], use_container_width=True)

    col1, col2, col3 = st.columns(3)
    theme_meta =[
        ("Dark — Cinematic Noir", "dark", col1),
        ("Light — Editorial Clean", "light", col2),
        ("Black & Yellow — Tabloid", "black_yellow", col3),
    ]

    for theme_name, theme_key, col_obj in theme_meta:
        img = result["images"][theme_key]
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        safe_name = re.sub(r"[^\w]", "_", result["topic"])[:30]

        with col_obj:
            st.caption(f"**{theme_name}**\n*{theme_labels[theme_key]}*")
            st.image(img, use_container_width=True)
            st.download_button(
                label=f"⬇️ {theme_name.split('—')[0].strip()}",
                data=buf,
                file_name=f"topic{index+1}_{theme_key}_{safe_name}.png",
                mime="image/png",
                key=f"dl_{index}_{theme_key}_{uuid.uuid4()}"
            )