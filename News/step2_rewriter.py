# step2_rewriter.py
import streamlit as st
from google import genai
from google.genai import types
from utils import generate_with_retry, clean_llm_output
from config import CURRENT_DATE

TONE_CONFIG = {
    "tagalog_viral": {
        "display_name": "Tagalog Viral",
        "instruction": (
            "Write in fluid, natural Tagalog. "
            "Sound like the smartest, most opinionated friend in the group chat who says the thing everyone's thinking but won't post publicly. "
            "Be a little condescending — not cruel, but the kind of tone that implies 'sana gets niyo 'to.' "
            "Use rhetorical questions to corner the reader. Build momentum. Let the sarcasm land dry, not loud. "
            "You have a clear point of view and you're not hiding it. Never sound like you're trying to be neutral."
        ),
        "theme": "dark",
        "default_label": "RAW TAKE PH"
    },
    "straight_news": {
        "display_name": "Straight News Filipino",
        "instruction": (
            "Write with the weight of someone who's been watching this country long enough to be tired but not surprised. "
            "Factual and grounded, but let the gravity of the situation breathe — don't just report, make the reader feel what's at stake. "
            "Use the facts to do the provocating. No need for snark when the reality is already damning. "
            "Sound like a sharp columnist, not a wire report. Authoritative, but human. Never sterile."
        ),
        "theme": "light",
        "default_label": "NOW, TODAY PH"
    },
    "satirical": {
        "display_name": "Satirical / Sarcastic",
        "instruction": (
            "Write with the energy of someone who has completely run out of patience and is now just narrating the absurdity with a straight face. "
            "Be cutting, condescending, and precise — mock the situation by taking it almost too seriously, or by being aggressively understanding of the nonsense. "
            "Filipino dark humor is the backbone: the kind that makes you laugh first, then feel the sting a second later. "
            "Pick a target — the policy, the statement, the hypocrisy — and do not let go. "
            "Deadpan is your best friend."
        ),
        "theme": "black_yellow",
        "default_label": "SIDE COMMENT PH"
    }
}

def render_step2(client, model_name):
    st.header("Step 2: AI Facebook Post Rewriter (All Three Tones)")
    st.write(f"Generates three tone variants per curated story: **Tagalog Viral**, **Straight News**, and **Satirical**.")

    if 'top_10_news' not in st.session_state or 'all_news' not in st.session_state:
        st.warning("⚠️ Complete Step 1 first to generate the Top 10 topics.")
        return

    df = st.session_state['all_news']
    num_topics = len(st.session_state['top_10_news'])

    if st.button(f"✍️ Generate {num_topics} Stories × 3 Tones"):
        st.session_state['fb_posts'] = []
        progress_bar = st.progress(0)
        status_container = st.empty()

        for topic_idx, item in enumerate(st.session_state['top_10_news']):
            status_container.text(f"Processing Topic {topic_idx + 1} of {num_topics}: {item.get('topic')}...")

            # Gather source material once per topic
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

            # Prepare links text (will be appended to each post)
            links_text = "\n\nSources:\n" + "\n".join(list(dict.fromkeys(related_links)))

            # Generate all three tones for this topic
            tone_posts = {}
            for tone_key, tone_info in TONE_CONFIG.items():
                fb_prompt = f"""You are the voice behind a high-engagement Philippine Facebook news page — 500K+ followers, known for saying what the broadsheets won't.

DATE: {CURRENT_DATE}
NEWS TOPIC: {item.get('topic')}
URGENCY LEVEL: {item.get('urgency', 'Important')}
TONE: {tone_info['instruction']}
LANGUAGE: TAGALOG INFORMAL

SOURCE MATERIAL — stick to this, no fabricating:
{topic_context}

WRITE THE POST FOLLOWING THESE PRINCIPLES:

1. HOOK: First sentence is a gut punch. The most alarming, absurd, or consequential fact — stated plainly, no warm-up. If it doesn't make someone stop scrolling, rewrite it.

2. DATE: Work {CURRENT_DATE} naturally into the second line or opening paragraph.

3. THE STORY: Cover what happened, who it hits, and what it means going forward. But don't just relay — react. Let the voice bleed through the facts. Keep it tight and fast-paced. The reader should feel like someone is quickly walking them through this, not reading them a report.

4. CLOSER: End with one sharp, specific question that demands an answer. Not "what do you think?" — something that forces a real opinion, creates sides, or highlights the absurdity of the situation.

5. THE TAKE: Specific commentary that earns the share. This isn't "here are both sides" — this is the part where you say the quiet part out loud. Make it something people will screenshot.

6. HASHTAGS: 3-5 relevant Filipino hashtags, last line.

7. EMOJIS: Use 1 max, placed deliberately — on the hook or to punch a key line. Not decorative. Not scattered.

STRICT TAGALOG GRAMMAR RULES:
1. Properly use Tagalog ligatures (-ng, -g, na).
2. If a word ends in a vowel and modifies the next word, attach '-ng' directly to the first word. 
3. NEVER use 'nang' as a linker between numbers or adjectives and nouns. 'Nang' is only for adverbs or conjunctions.
Examples to follow:
- Incorrect: Pito nang linggong | Correct: Pitong linggo
- Incorrect: Dalawa na aso | Correct: Dalawang aso
- Incorrect: Tatlo nang araw | Correct: Tatlong araw

HARD RULES:
— Do not mention the source material explicitly in the post. The information should be synthesized and woven into the narrative, not cited directly.
— Output only the post. No headers, no labels, no meta-commentary.
— Every fact from source material only. Never invent quotes or statistics.
— 125-175 words strictly enforced. 
— It should read like a person wrote it at 11pm after reading the news and having quick, sharp thoughts about it."""

                try:
                    response = generate_with_retry(
                        client=client,
                        model=model_name,
                        contents=fb_prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                            system_instruction = (
    "You are a seasoned Filipino social media journalist and an expert, native Tagalog speaker born and raised in the Philippines. "
    "You know how to write posts that go viral for the right reasons: they're accurate, punchy, and shareable. "
    "Your writing is highly natural, grammatically flawless, and uses proper conversational Tagalog without sounding like a machine translation. "
    "You never plagiarize — you synthesize and editorialize. "
    "Output only the post text. No meta-commentary."
),
                            temperature=1.0
                        )
                    )
                    post_content = clean_llm_output(response.text)
                    tone_posts[tone_key] = post_content.strip() + links_text
                except Exception as e:
                    st.error(f"Error generating {tone_info['display_name']} for topic {topic_idx+1}: {e}")
                    tone_posts[tone_key] = f"[ERROR: {e}]"

            st.session_state['fb_posts'].append({
                "topic": item.get('topic'),
                "urgency": item.get('urgency', 'Important'),
                "category": item.get('category', ''),
                "tones": tone_posts,
                "related_links": related_links
            })

            progress_bar.progress((topic_idx + 1) / num_topics)

        status_container.text(f"✅ All {num_topics} topics processed (3 tones each)!")

    # Display results
    if 'fb_posts' in st.session_state:
        st.success(f"📋 {len(st.session_state['fb_posts'])} topics ready, each with 3 tone variants:")
        for topic_idx, topic_data in enumerate(st.session_state['fb_posts']):
            urgency_icons = {"Breaking": "🔴", "Developing": "🟠", "Important": "🟡"}
            icon = urgency_icons.get(topic_data.get('urgency', 'Important'), "🟡")
            with st.expander(f"{icon} Topic #{topic_idx+1}: {topic_data['topic']}", expanded=(topic_idx < 2)):
                tabs = st.tabs(["🔥 Tagalog Viral", "📰 Straight News", "😏 Satirical"])
                for tab, (tone_key, tone_info) in zip(tabs, TONE_CONFIG.items()):
                    with tab:
                        post_content = topic_data['tones'].get(tone_key, "[Not generated]")
                        st.text_area(
                            f"Copy Post ({tone_info['display_name']})",
                            value=post_content,
                            height=280,
                            key=f"post_{topic_idx}_{tone_key}"
                        )
                        char_count = len(post_content)
                        st.caption(f"Character count: {char_count} {'✅' if char_count <= 2000 else '⚠️ Consider trimming'}")