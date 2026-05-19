# step1_curate.py
import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
from utils import fetch_all_news_simultaneously, generate_with_retry, extract_json_from_text
from config import CURRENT_DATE

def render_step1(client, model_name):
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
1. VIRALITY & POLARIZATION (Top Priority): Stories dominating social media algorithms (Facebook, TikTok, X), driving intense public debate, and splitting Filipino society along clear political, generational, or class lines. Prioritize topics with high engagement metrics (shares, quote-tweets, comment wars) and those fueling "us vs. them" narratives.
2. CONTROVERSIALITY: Developments involving public clashes between high-profile political figures, scandals with moral/ethical outrage, whistleblower accusations, or government actions sparking widespread protest/backlash. The more friction and public noise, the higher the rank.
3. HIGH-IMPACT VOLATILITY: Major political earthquakes (impeachment talk, coalition collapses, leadership challenges) that may not have immediate policy impact but generate significant uncertainty, speculation, and partisan reaction in the news cycle.
4. GLOBAL STORIES WITH LOCAL POLARIZATION: International events (West Philippine Sea confrontations, US politics, ICC developments) that are being weaponized in local Philippine political discourse to rile up specific voter bases or online communities.
5. The "HIGH IMPACT" Filter: The story must still be substantive enough to warrant national attention. The "impact" is measured by the *scale of the reaction* and its ability to dominate the national conversation, rather than just tangible economic/safety consequences.
6. AVOID: Celebrity gossip, non-controversial sports results, or routine weather updates UNLESS the story itself has become a viral, polarizing flashpoint (e.g., a celebrity's political statement causing a trending topic).
7. PREFER: Stories corroborated by multiple sources (cross-reference IDs) to confirm the factual core of the controversy exists, even if the interpretation of that fact is heavily contested.

IMPORTANT: The article ID numbers in brackets (e.g., [ID:42]) are the exact row indices you must use in `related_article_ids`. Only include IDs that are directly relevant to the story. Do not invent IDs.
IMPORTANT: LANGUAGE MUST BE IN TAGALOG. The post will be published on a Filipino Facebook page, so the tone, phrasing, and grammar should resonate with a local audience.
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
                response = generate_with_retry(
                    client=client,
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                        thinking_level="HIGH",
                        ),
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