import streamlit as st
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import concurrent.futures
from google import genai
from google.genai import types
import json
import re
import os
import requests
import base64
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import textwrap
import time
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- CONFIGURATION & FEEDS ---
st.set_page_config(page_title="Automated FB News Channel", layout="wide")

# API Key from Streamlit secrets (create .streamlit/secrets.toml)
API_KEY = "AIzaSyC2Ocy2OgfbgaHaX5xQzxLhDsb_ioAW6pA"

# Current Date
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")
CURRENT_DATE_SHORT = datetime.now().strftime("%b %d, %Y").upper()

# Expanded Reliable RSS Feeds
NEWS_FEEDS = {
    "Philippine News": {
        "Inquirer Headlines":  "https://www.inquirer.net/fullfeed",
        "Philstar News":       "https://www.philstar.com/rss/headlines",
        "GMA News":            "https://data.gmanetwork.com/gno/rss/news/feed.xml",
        "Manila Bulletin":     "https://mb.com.ph/rss",
        "Manila Times":        "https://www.manilatimes.net/news/feed",
        "Rappler":             "https://www.rappler.com/feed/",
        "ABS-CBN News":        "https://news.abs-cbn.com/rss/Headlines.xml",
        "BusinessWorld PH":    "https://www.bworldonline.com/feed/",
        "Inquirer Business":   "https://business.inquirer.net/feed",
        "SunStar Network":     "https://www.sunstar.com.ph/rss/latest",
    },
    "International News": {
        "BBC World":           "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Reuters World":       "https://feeds.reuters.com/reuters/worldNews",
        "Al Jazeera":          "https://www.aljazeera.com/xml/rss/all.xml",
        "The Guardian World":  "https://www.theguardian.com/world/rss",
        "NYT World":           "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "CNN Top Stories":     "http://rss.cnn.com/rss/edition.rss",
        "AP Top News":         "https://feeds.apnews.com/rss/apf-topnews",
        "DW World":            "https://rss.dw.com/rdf/rss-en-world",
        "Foreign Policy":      "https://foreignpolicy.com/feed/",
        "UN News":             "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
    },
}

# --- HELPER FUNCTIONS ---
def clean_html(raw_html):
    if not raw_html:
        return "No summary available."
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()[:400] + "..."

def fetch_single_feed(source_name, feed_url, category, limit=15):
    try:
        resp = requests.get(feed_url, timeout=10)
        parsed = feedparser.parse(resp.content)
        articles = []
        for entry in parsed.entries[:limit]:
            summary = entry.get("summary") or entry.get("description") or ""
            articles.append({
                "Category": category,
                "Source": source_name,
                "Title": entry.get("title", "No Title"),
                "Link": entry.get("link", ""),
                "Summary": clean_html(summary),
            })
        return articles
    except Exception as e:
        st.warning(f"Feed fetch error {source_name}: {e}")
        return []

@st.cache_data(ttl=900)
def fetch_all_news_simultaneously():
    all_articles = []
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for category, feeds in NEWS_FEEDS.items():
            for source_name, url in feeds.items():
                tasks.append(executor.submit(fetch_single_feed, source_name, url, category))
        for future in concurrent.futures.as_completed(tasks):
            all_articles.extend(future.result())
    df = pd.DataFrame(all_articles)
    df.reset_index(drop=True, inplace=True)
    return df

def clean_llm_output(text):
    """Aggressively strips <thought> blocks and markdown from the text."""
    if not text:
        return ""
    text = re.sub(r'<(thought|think|thinking)>.*?</\1>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)
    return text.strip()

def extract_json_from_text(text):
    """Bulletproof JSON extractor that cleans text first before finding brackets."""
    if not text:
        return None
    cleaned_text = clean_llm_output(text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    start_idx_array = cleaned_text.find('[')
    end_idx_array = cleaned_text.rfind(']')
    start_idx_obj = cleaned_text.find('{')
    end_idx_obj = cleaned_text.rfind('}')
    try:
        if start_idx_array != -1 and end_idx_array != -1 and end_idx_array > start_idx_array:
            return json.loads(cleaned_text[start_idx_array:end_idx_array + 1])
        elif start_idx_obj != -1 and end_idx_obj != -1 and end_idx_obj > start_idx_obj:
            return json.loads(cleaned_text[start_idx_obj:end_idx_obj + 1])
    except json.JSONDecodeError:
        pass
    return None

# Retry decorator for GenAI calls (exponential backoff up to ~1 min to respect 30 RPM)
@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=4, max=65),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def generate_with_retry(client, model, contents, config):
    """Wrapper for API call with retry logic."""
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )
    return response


# --- NANO BANANA 2: GOOGLE IMAGEN VISUAL ENGINE ---
class NanoBanana2:
    """
    Nano Banana 2 Visual Engine.
    Uses Imagen 4 Fast to generate a photorealistic news background image 
    (with a fallback to gemini-2.5-flash-image), then composites a professional 
    news banner over it using PIL.
    """

    @staticmethod
    @retry(
        stop=stop_after_attempt(15),  # robust retries for Imagen/Gemini API
        wait=wait_exponential(multiplier=2, min=10, max=70), # larger backoff to respect 10 RPM limit
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def generate_background_with_imagen(image_prompt: str, client) -> Image.Image:
        """
        Calls models/imagen-4.0-fast-generate-001 via google.genai 
        to generate a background image. Returns a PIL Image. Retries automatically on error.
        If Imagen 4 Fast fails, it falls back to the gemini-2.5-flash-image model.
        """
        try:
            result = client.models.generate_images(
                model="models/imagen-4.0-fast-generate-001",
                prompt=image_prompt,
                config=dict(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    person_generation="ALLOW_ADULT",
                    aspect_ratio="1:1",
                ),
            )

            if not result.generated_images:
                raise Exception("Imagen API returned no generated images")

            # Get the SDK's custom Image object
            sdk_image = result.generated_images[0].image
            
            # Convert the SDK's Image object into a standard PIL Image
            # The SDK stores the raw JPEG data in the `.image_bytes` attribute
            img_bytes = sdk_image.image_bytes
            
            # Load into PIL and convert to RGB
            return Image.open(BytesIO(img_bytes)).convert("RGB")

        except Exception as e:
            print(f"Imagen API encountered an error: {e}. Falling back to gemini-2.5-flash-image...")
            
            # Fallback to gemini-2.5-flash-image using generate_content API
            fallback_response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=image_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",
                    ),
                ),
            )
            
            # Extract the image from the parts payload returned by the Gemini Image model
            for part in fallback_response.parts:
                if part.inline_data:
                    img_bytes = part.inline_data.data
                    return Image.open(BytesIO(img_bytes)).convert("RGB")
                    
            raise Exception(f"Fallback gemini-2.5-flash-image API returned no image payload. Original error: {e}")


    @staticmethod
    def composite_news_banner(
        background: Image.Image,
        headline: str,
        subtext: str,
        date_text: str,
        source_label: str = "NEWS UPDATE"
    ) -> Image.Image:
        """
        Composites a professional dark gradient news banner
        with headline, subtext, and date onto the background image.
        """
        img = background.copy().resize((1080, 1080), Image.LANCZOS)
        draw = ImageDraw.Draw(img, "RGBA")
        width, height = img.size

        def load_font(path, size):
            try:
                return ImageFont.truetype(path, size)
            except:
                try:
                    return ImageFont.truetype("Arial.ttf", size)
                except:
                    return ImageFont.load_default()

        font_label    = load_font("arialbd.ttf", 28)
        font_headline = load_font("arialbd.ttf", 56)
        font_subtext  = load_font("arial.ttf", 32)
        font_date     = load_font("arial.ttf", 26)

        banner_start = int(height * 0.58)
        for y in range(banner_start, height):
            alpha = int(220 * ((y - banner_start) / (height - banner_start)) ** 0.6 + 30)
            draw.line([(0, y), (width, y)], fill=(10, 10, 20, min(alpha, 240)))

        accent_y = banner_start + 10
        draw.rectangle([(40, accent_y), (160, accent_y + 6)], fill=(220, 30, 30, 255))

        draw.text((40, accent_y + 18), source_label, font=font_label, fill=(220, 30, 30, 255))

        date_str = date_text.upper()
        bbox_d = draw.textbbox((0, 0), date_str, font=font_date)
        date_w = bbox_d[2] - bbox_d[0]
        draw.rectangle(
            [(width - date_w - 30, 28), (width - 18, 28 + 40)],
            fill=(220, 30, 30, 230)
        )
        draw.text((width - date_w - 22, 32), date_str, font=font_date, fill=(255, 255, 255, 255))

        headline_y = accent_y + 60
        lines = textwrap.wrap(headline.upper(), width=28)
        for line in lines:
            draw.text((40, headline_y), line, font=font_headline, fill=(255, 215, 0, 255))
            bbox = draw.textbbox((0, 0), line, font=font_headline)
            headline_y += (bbox[3] - bbox[1]) + 8

        subtext_y = headline_y + 18
        sub_lines = textwrap.wrap(subtext, width=46)
        for line in sub_lines[:3]:
            draw.text((40, subtext_y), line, font=font_subtext, fill=(230, 230, 230, 220))
            bbox = draw.textbbox((0, 0), line, font=font_subtext)
            subtext_y += (bbox[3] - bbox[1]) + 6

        return img

    @staticmethod
    def generate_imagen_prompt(headline: str, subtext: str, category: str = "news") -> str:
        """
        Crafts a detailed prompt for a photorealistic
        editorial news background based on the story topic.
        """
        # We intentionally omit the literal `headline` variable from the prompt
        # to prevent the model from attempting to render the text. 
        # The `subtext` argument (which receives your 'scene' variable) drives the visual.
        return (
            f"A purely photographic, wide cinematic establishing shot of {subtext}. "
            f"Category: {category}. "
            f"Photorealistic, high resolution, dramatic and moody cinematic lighting, 4K quality. "
            f"Clean composition focusing entirely on the environment and atmosphere. "
            f"All surfaces, walls, billboards, backgrounds, and objects are completely pristine, blank, featureless, and untouched. "
            f"Pure visual space only. "
            f"Any people present must be in the distant background, out of focus, with faces completely obscured."
        )


def generate_single_image(args):
    """
    Worker function for parallel image generation.
    Takes a tuple of (index, post, meta, client, source_label) and returns
    (index, result_dict_or_error).
    """
    i, post, meta, api_key, source_label = args
    # Each thread needs its own client instance
    thread_client = genai.Client(api_key=api_key)

    headline = meta.get('headline', post['topic'][:50].upper())
    subtext  = meta.get('subtext', 'Read the full post for complete details.')
    date_text = meta.get('date', CURRENT_DATE)
    scene    = meta.get('scene', 'dramatic cityscape at dusk')
    category = meta.get('category', post.get('category', 'news'))

    imagen_prompt = NanoBanana2.generate_imagen_prompt(headline, scene, category)
    background = NanoBanana2.generate_background_with_imagen(imagen_prompt, thread_client)

    final_image = NanoBanana2.composite_news_banner(
        background=background,
        headline=headline,
        subtext=subtext,
        date_text=date_text,
        source_label=source_label.upper()
    )

    return i, {
        "topic":    post['topic'],
        "headline": headline,
        "subtext":  subtext,
        "date":     date_text,
        "scene":    scene,
        "image":    final_image,
        "imagen_prompt": imagen_prompt,
    }


# --- UI & LLM SETUP ---
st.sidebar.title("⚙️ LLM Configuration")
st.sidebar.write("Google Generative Language API (google.genai)")
model_name = st.sidebar.text_input("Rewriter & Curator Model", value="gemma-4-26b-a4b-it")

client = genai.Client(api_key=API_KEY)

st.title("📰 Automated FB News Channel Pipeline")

st.sidebar.title("Pipeline Steps")
step = st.sidebar.radio("Navigate", [
    "1. Scrape & Curate Top 10",
    "2. Facebook Post Rewriter",
    "3. Visuals Generator (Nano Banana 2)",
    "4. Post Scheduler - Coming Soon"
])

log_container = st.empty()

# ============================================================
# STEP 1: SCRAPE & CURATE  (no thinking_config change — full reasoning)
# ============================================================
if step == "1. Scrape & Curate Top 10":
    st.header("Step 1: Simultaneous Scrape & AI Cross-Referencing")
    st.write("Scrapes all RSS feeds in parallel and uses AI to surface the Top 10 most important stories for a Philippine audience.")

    if st.button("🚀 Fetch, Cross-Reference & Curate Top 10"):
        with st.status("Fetching all local and international news...", expanded=True) as status:
            status.update(label="Scraping RSS feeds...")
            df = fetch_all_news_simultaneously()
            st.session_state['all_news'] = df
            status.update(label=f"✅ Scraped {len(df)} articles from {df['Source'].nunique()} sources!", state="complete")

        with st.status(f"Using {model_name} to curate Top 10...", expanded=True) as status:
            limited_df = df.groupby('Source').head(5)
            news_context = ""
            for idx, row in limited_df.iterrows():
                news_context += (
                    f"[ID:{idx}] [{row['Category']}] [{row['Source']}]\n"
                    f"TITLE: {row['Title']}\n"
                    f"SUMMARY: {row['Summary']}\n\n"
                )

            prompt = f"""You are the chief news editor of a major Philippine digital media outlet with 20 years of experience.
Today's date is {CURRENT_DATE}.

YOUR TASK:
Analyze the following scraped RSS articles and identify the TOP 10 most newsworthy stories for a Filipino audience right now.

SELECTION CRITERIA (ranked by priority):
1. Direct impact on Filipino lives (economy, safety, politics, disasters, public health)
2. Major Philippine government or political developments
3. International stories with significant Philippine implications (OFW issues, trade, diplomacy)
4. High-impact global events (wars, major elections, economic crises) that Filipinos care about
5. Avoid: celebrity gossip, sports unless major, duplicate stories (pick the best-sourced version)
6. Prefer: stories corroborated by multiple sources (cross-reference the IDs)

IMPORTANT: The article ID numbers in brackets (e.g., [ID:42]) are the exact row indices you must use in `related_article_ids`. Only include IDs that are directly relevant to the story. Do not invent IDs.
IMPORTANT: EVERY HEADLINE MUST BE IN TAGALOG, reflecting the Filipino voice of the original post.
OUTPUT FORMAT:
Return ONLY a valid JSON array of exactly 10 objects. No preamble, no markdown, no explanations.
Each object must follow this exact schema:

[
  {{
    "rank": 1,
    "topic": "<Clear, specific news headline — not vague>",
    "reason": "<2 sentences: why this matters specifically to Filipinos today, what the stakes are>",
    "category": "<Philippine News | International News>",
    "urgency": "<Breaking | Developing | Important>",
    "related_article_ids": [<id1>, <id2>, <id3>]
  }}
]

SCRAPED ARTICLES:
{news_context}"""

            try:
                # NOTE: Curation uses NO thinking_config override — full reasoning intentionally kept
                response = generate_with_retry(
                    client=client,
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are a senior Philippine news editor. You output ONLY valid JSON. "
                            "No markdown, no code fences, no commentary — raw JSON only."
                        ),
                        temperature=1.0
                    )
                )
                llm_output = response.text
                top_10_json = extract_json_from_text(llm_output)

                if top_10_json and len(top_10_json) > 0:
                    st.session_state['top_10_news'] = top_10_json
                    status.update(label=f"✅ Top {len(top_10_json)} stories curated successfully!", state="complete")
                else:
                    status.update(label="❌ LLM did not return valid JSON. See raw output below.", state="error")
                    st.code(llm_output)

            except Exception as e:
                status.update(label=f"❌ API error: {e}", state="error")
                st.error(f"Error communicating with GenAI API: {e}")

    if 'top_10_news' in st.session_state and 'all_news' in st.session_state:
        st.subheader(f"🌟 Today's Top {len(st.session_state['top_10_news'])} Topics — {CURRENT_DATE}")
        df = st.session_state['all_news']
        urgency_colors = {"Breaking": "🔴", "Developing": "🟠", "Important": "🟡"}

        for idx, item in enumerate(st.session_state['top_10_news']):
            urgency = item.get('urgency', 'Important')
            icon = urgency_colors.get(urgency, "🟡")
            rank = item.get('rank', idx + 1)
            with st.expander(
                f"{icon} #{rank} [{urgency}] {item.get('topic')}",
                expanded=(idx < 3)
            ):
                st.write(f"**Category:** {item.get('category', 'N/A')}")
                st.write(f"**Why it matters:** {item.get('reason')}")
                st.write("**Source articles:**")
                for a_id in item.get('related_article_ids', []):
                    try:
                        a_id = int(a_id)
                    except:
                        continue
                    if 0 <= a_id < len(df):
                        article = df.iloc[a_id]
                        st.markdown(f"- **{article['Source']}**: [{article['Title']}]({article['Link']})")
                    else:
                        st.caption(f"⚠️ Article ID {a_id} out of range (total articles: {len(df)}).")

# ============================================================
# STEP 2: FACEBOOK POST REWRITER  (MINIMAL thinking)
# ============================================================
elif step == "2. Facebook Post Rewriter":
    st.header("Step 2: AI Facebook Post Rewriter")
    st.write(f"Generates punchy, viral-ready Facebook posts for each curated story. **Date:** {CURRENT_DATE}")

    if 'top_10_news' not in st.session_state or 'all_news' not in st.session_state:
        st.warning("⚠️ Complete Step 1 first to generate the Top 10 topics.")
    else:
        df = st.session_state['all_news']
        num_topics = len(st.session_state['top_10_news'])

        tone_options = {
            "Tagalog Viral (Default)": "tagalog_viral",
            "Straight News Filipino": "straight_news",
            "Satirical / Sarcastic": "satirical"
        }
        selected_tone = st.selectbox("Post Tone Style:", list(tone_options.keys()), index=2)
        tone_key = tone_options[selected_tone]

        tone_instructions = {
    "tagalog_viral": (
        "Write in fluid, natural Tagalog "
        "Sound like the smartest, most opinionated friend in the group chat who says the thing everyone's thinking but won't post publicly. "
        "Be a little condescending — not cruel, but the kind of tone that implies 'sana gets niyo 'to.' "
        "Use rhetorical questions to corner the reader. Build momentum. Let the sarcasm land dry, not loud. "
        "You have a clear point of view and you're not hiding it. Never sound like you're trying to be neutral."
    ),
    "straight_news": (
        "Write with the weight of someone who's been watching this country long enough to be tired but not surprised. "
        "Factual and grounded, but let the gravity of the situation breathe — don't just report, make the reader feel what's at stake. "
        "Use the facts to do the provocating. No need for snark when the reality is already damning. "
        "Sound like a sharp columnist, not a wire report. Authoritative, but human. Never sterile."
    ),
    "satirical": (
        "Write with the energy of someone who has completely run out of patience and is now just narrating the absurdity with a straight face. "
        "Be cutting, condescending, and precise — mock the situation by taking it almost too seriously, or by being aggressively understanding of the nonsense. "
        "Filipino dark humor is the backbone: the kind that makes you laugh first, then feel the sting a second later. "
        "Pick a target — the policy, the statement, the hypocrisy — and do not let go. "
        "Deadpan is your best friend."
    ),
}

        if st.button(f"✍️ Generate {num_topics} Facebook Posts"):
            st.session_state['fb_posts'] = []
            progress_bar = st.progress(0)
            status_container = st.empty()

            for i, item in enumerate(st.session_state['top_10_news']):
                status_container.text(f"Writing Post {i + 1} of {num_topics}: {item.get('topic')}...")

                topic_context = ""
                related_links = []
                for a_id in item.get('related_article_ids', []):
                    try:
                        a_id = int(a_id)
                    except:
                        continue
                    if 0 <= a_id < len(df):
                        article = df.iloc[a_id]
                        topic_context += (
                            f"SOURCE: {article['Source']}\n"
                            f"TITLE: {article['Title']}\n"
                            f"SUMMARY: {article['Summary']}\n\n"
                        )
                        related_links.append(article['Link'])

                fb_prompt = f"""You are the voice behind a high-engagement Philippine Facebook news page — 500K+ followers, known for saying what the broadsheets won't.

DATE: {CURRENT_DATE}
NEWS TOPIC: {item.get('topic')}
URGENCY LEVEL: {item.get('urgency', 'Important')}
TONE: {tone_instructions[tone_key]}

SOURCE MATERIAL — stick to this, no fabricating:
{topic_context}

WRITE THE POST FOLLOWING THESE PRINCIPLES:

1. HOOK: First sentence is a gut punch. The most alarming, absurd, or consequential fact — stated plainly, no warm-up. If it doesn't make someone stop scrolling, rewrite it.

2. DATE: Work {CURRENT_DATE} naturally into the second line or opening paragraph.

3. THE STORY: 3-5 short paragraphs, max 3 sentences each. Cover what happened, who it hits, and what it means going forward. But don't just relay — react. Let the voice bleed through the facts. The reader should feel like someone is walking them through this, not reading them a report.

4. THE TAKE: 2-3 sentences of pointed, specific commentary that earns the share. This isn't "here are both sides" — this is the part where you say the quiet part out loud. Make it something people will screenshot.

5. CLOSER: End with one sharp, specific question that demands an answer. Not "what do you think?" — something that forces a real opinion, creates sides, or highlights the absurdity of the situation.

6. HASHTAGS: 3-5 relevant Filipino hashtags, last line.

7. EMOJIS: Use 1-3 max, placed deliberately — on the hook or to punch a key line. Not decorative. Not scattered.

HARD RULES:
— Output only the post. No headers, no labels, no meta-commentary.
— Every fact from source material only. Never invent quotes or statistics.
— 200-350 words.
— It should read like a person wrote it at 11pm after reading the news and having thoughts about it."""

                try:
                    response = generate_with_retry(
                        client=client,
                        model=model_name,
                        contents=fb_prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(
                                thinking_level="HIGH",
                            ),
                            system_instruction=(
                                "You are a seasoned Filipino social media journalist who knows how to write posts "
                                "that go viral for the right reasons: they're accurate, punchy, and shareable. "
                                "You never plagiarize — you synthesize and editorialize. "
                                "Output only the post text. No meta-commentary."
                            ),
                            temperature=1.0
                        )
                    )

                    fb_post_content = clean_llm_output(response.text)
                    links_text = "\n\nSources:\n" + "\n".join(list(dict.fromkeys(related_links)))
                    final_post = fb_post_content.strip() + links_text

                    st.session_state['fb_posts'].append({
                        "topic": item.get('topic'),
                        "urgency": item.get('urgency', 'Important'),
                        "category": item.get('category', ''),
                        "content": final_post
                    })

                except Exception as e:
                    st.error(f"Error generating post {i + 1}: {e}")

                progress_bar.progress((i + 1) / num_topics)

            status_container.text(f"✅ All {num_topics} posts written!")

        if 'fb_posts' in st.session_state:
            st.success(f"📋 {len(st.session_state['fb_posts'])} posts ready:")
            for idx, post_data in enumerate(st.session_state['fb_posts']):
                urgency_icons = {"Breaking": "🔴", "Developing": "🟠", "Important": "🟡"}
                icon = urgency_icons.get(post_data.get('urgency', 'Important'), "🟡")
                st.subheader(f"{icon} Post #{idx + 1} — {post_data['topic']}")
                st.text_area(
                    f"Copy Post #{idx + 1}:",
                    value=post_data['content'],
                    height=280,
                    key=f"post_{idx}"
                )
                char_count = len(post_data['content'])
                st.caption(f"Character count: {char_count} {'✅' if char_count <= 2000 else '⚠️ Consider trimming'}")

# ============================================================
# STEP 3: VISUALS GENERATOR — NANO BANANA 2 (PARALLEL + MINIMAL thinking)
# ============================================================
elif step == "3. Visuals Generator (Nano Banana 2)":
    st.header("Step 3: Nano Banana 2 — AI Image Generator")
    st.info(
        "**Nano Banana 2** uses gemini-3.1-flash-image-preview to generate a photorealistic editorial background "
        "for each post in **parallel**, then composites a professional news banner with headline, subtext, and date."
    )

    if 'fb_posts' not in st.session_state:
        st.warning("⚠️ Complete Step 2 first to generate Facebook Posts.")
    else:
        num_posts = len(st.session_state['fb_posts'])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            show_imagen_prompt = st.checkbox("Show Imagen prompts (debug)", value=False)
        with col_b:
            source_label = st.text_input("Page label on image", value="NOW, TODAY")
        with col_c:
            max_parallel_workers = st.slider("Parallel workers", min_value=1, max_value=10, value=2,
                                             help="How many images to generate simultaneously")

        generate_clicked = st.button(f"🎨 Generate {num_posts} AI Visuals in Parallel")

        if generate_clicked:
            st.session_state['generated_images'] = [None] * num_posts

            # ----- BATCH METADATA EXTRACTION (single LLM call, MINIMAL thinking) -----
            with st.status("📋 Extracting headlines/subtexts for all posts...", expanded=True) as status:
                batch_prompt = f"""You are a graphic designer extracting text elements for {num_posts} social media news cards.

For EACH of the following Facebook posts, extract exactly 5 elements and return a JSON array.

Facebook posts (in order):
{json.dumps([p['content'] for p in st.session_state['fb_posts']], indent=2)}

For each post, provide an object with these keys:
- "headline": The core news event in 6-8 ALL CAPS words. Punchy, urgent, no filler words.
- "subtext": A supporting detail or consequence in 12-16 words. Informative, not a repeat of the headline.
- "date": The exact date mentioned in the post (format: "MONTH DD, YYYY", e.g. "APRIL 08, 2026")
- "scene": A 20-word description of a photorealistic scene that visually represents this news story. No people's faces. No text in scene. Describe setting/environment/objects only.
- "category": The post's category (e.g., "Philippine News" or "International News")

Return ONLY a valid JSON array of {num_posts} objects. No markdown, no explanations.
IMPORTANT: EVERY HEADLINE MUST BE IN TAGALOG GRAMMATICALLY CORRECT AND COMPLETE, reflecting the Filipino voice of the original post.
Example output format:
[
  {{
    "headline": "PUNCHY HEADLINE IN CAPS",
    "subtext": "Supporting detail sentence here.",
    "date": "MONTH DD, YYYY",
    "scene": "Descriptive photorealistic scene without faces or text.",
    "category": "Philippine News"
  }},
  ...
]"""

                try:
                    meta_response = generate_with_retry(
                        client=client,
                        model=model_name,
                        contents=batch_prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(
                                thinking_level="HIGH",
                            ),
                            system_instruction=(
                                "You are a graphic designer assistant. Output only valid JSON array. No markdown, no code fences."
                            ),
                            temperature=1.0
                        )
                    )

                    all_metadata = extract_json_from_text(meta_response.text)
                    if not all_metadata or not isinstance(all_metadata, list):
                        raise ValueError("Invalid batch metadata response")
                    if len(all_metadata) != num_posts:
                        st.warning(f"Expected {num_posts} metadata objects, got {len(all_metadata)}. Using fallback where needed.")
                        while len(all_metadata) < num_posts:
                            all_metadata.append({})

                    status.update(label=f"✅ Metadata extracted for {len(all_metadata)} posts", state="complete")

                except Exception as e:
                    status.update(label=f"❌ Batch extraction failed: {e}. Using fallback metadata.", state="error")
                    all_metadata = [{} for _ in range(num_posts)]

            # ----- PARALLEL IMAGE GENERATION -----
            st.info(f"🚀 Generating {num_posts} images in parallel with {max_parallel_workers} workers...")
            progress_bar = st.progress(0)
            progress_text = st.empty()

            # Build task args — pass API_KEY so each thread makes its own client
            task_args = [
                (i, st.session_state['fb_posts'][i], all_metadata[i], API_KEY, source_label)
                for i in range(num_posts)
            ]

            results = [None] * num_posts
            completed = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
                future_to_idx = {executor.submit(generate_single_image, args): args[0] for args in task_args}

                for future in concurrent.futures.as_completed(future_to_idx):
                    try:
                        idx, result = future.result()
                        results[idx] = result
                        completed += 1
                        progress_bar.progress(completed / num_posts)
                        progress_text.text(f"✅ {completed}/{num_posts} images generated — latest: {result['headline'][:40]}...")
                    except Exception as e:
                        orig_idx = future_to_idx[future]
                        st.error(f"❌ Image {orig_idx + 1} failed: {e}")
                        completed += 1
                        progress_bar.progress(completed / num_posts)

            # Store ordered results
            st.session_state['generated_images'] = results

            progress_text.text(f"✅ All {num_posts} visuals generated!")
            st.success(f"✅ All {num_posts} visuals generated!")

            # ----- DISPLAY ALL RESULTS -----
            display_cols = st.columns(2)
            for i, result in enumerate(results):
                if result is None:
                    display_cols[i % 2].error(f"Image #{i+1} failed to generate.")
                    continue

                buf = BytesIO()
                result['image'].save(buf, format="PNG")
                buf.seek(0)
                safe_name = re.sub(r'[^\w]', '_', result['topic'])[:40]

                with display_cols[i % 2].container():
                    st.subheader(f"#{i+1} — {result['topic'][:40]}...")
                    st.image(result['image'], use_container_width=True)

                    if show_imagen_prompt:
                        st.caption(f"🖼️ Prompt: `{result.get('imagen_prompt', '')}`")

                    st.download_button(
                        label=f"⬇️ Download Image #{i+1}",
                        data=buf,
                        file_name=f"nb2_{safe_name}.png",
                        mime="image/png",
                        key=f"dl_{i}_{uuid.uuid4()}"
                    )
                    with st.expander("📋 Extracted Metadata"):
                        st.write(f"**Date:** {result['date']}")
                        st.write(f"**Headline:** {result['headline']}")
                        st.write(f"**Subtext:** {result['subtext']}")
                        st.write(f"**Scene prompt:** {result['scene']}")

        # Show existing images if already generated
        elif 'generated_images' in st.session_state:
            cols = st.columns(2)
            for idx, item in enumerate(st.session_state['generated_images']):
                if item is None:
                    continue
                with cols[idx % 2]:
                    st.subheader(f"#{idx+1} — {item['topic'][:40]}...")
                    st.image(item['image'], use_container_width=True)

                    buf = BytesIO()
                    item['image'].save(buf, format="PNG")
                    buf.seek(0)
                    safe_name = re.sub(r'[^\w]', '_', item['topic'])[:40]
                    st.download_button(
                        label=f"⬇️ Download Image #{idx+1}",
                        data=buf,
                        file_name=f"nb2_{safe_name}.png",
                        mime="image/png",
                        key=f"dl_{idx}_{uuid.uuid4()}"
                    )
                    with st.expander("📋 Extracted Metadata"):
                        st.write(f"**Date:** {item['date']}")
                        st.write(f"**Headline:** {item['headline']}")
                        st.write(f"**Subtext:** {item['subtext']}")
                        st.write(f"**Scene prompt:** {item['scene']}")

# ============================================================
# STEP 4: SCHEDULER (Coming Soon)
# ============================================================
elif step == "4. Post Scheduler - Coming Soon":
    st.header("Step 4: Automated Post Scheduler")
    st.info(
        "**Coming Soon:** Connect the Facebook Graph API to automatically schedule "
        "each Rewriter post + Nano Banana 2 image directly to your Facebook Page "
        "at optimal posting times."
    )
    st.markdown("""
    **Planned features:**
    - OAuth 2.0 Facebook Page authentication
    - Optimal time-slot scheduler based on your page's peak engagement hours  
    - Bulk queue: schedule all 10 posts for the day in one click
    - Preview mode before publishing
    - Post performance tracker (reach, engagement, shares)
    """)