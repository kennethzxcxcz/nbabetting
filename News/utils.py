# utils.py
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import concurrent.futures
import requests
import json
import re
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def clean_html(raw_html):
    """Strip HTML tags and truncate summary."""
    if not raw_html:
        return "No summary available."
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()[:400] + "..."

def fetch_single_feed(source_name, feed_url, category, limit=15):
    """Fetch a single RSS feed and return a list of article dicts."""
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
    """Fetch all news feeds in parallel and return a DataFrame."""
    from config import NEWS_FEEDS
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
    """Strip <thought> blocks and markdown fences from LLM response."""
    if not text:
        return ""
    text = re.sub(r'<(thought|think|thinking)>.*?</\1>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)
    return text.strip()

def extract_json_from_text(text):
    """Robust JSON extraction from LLM text."""
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

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=4, max=65),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def generate_with_retry(client, model, contents, config):
    """API call wrapper with exponential backoff."""
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )
    return response