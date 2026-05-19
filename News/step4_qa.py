# step4_qa.py
import streamlit as st
from google.genai import types
from utils import generate_with_retry, clean_llm_output
from step2_rewriter import TONE_CONFIG

def render_step4(client, model_name):
    st.header("Step 4: Copywriting & QA Polish")
    st.write("The 'Final Eye' phase. AI scans for grammatical errors, corrects phrasing, and ensures the text perfectly matches the generated visual mood.")

    if 'fb_posts' not in st.session_state or 'generated_visuals' not in st.session_state:
        st.warning("⚠️ Please complete Step 2 and Step 3 first.")
        return

    num_topics = len(st.session_state['fb_posts'])

    if st.button(f"✨ Polish & QA all {num_topics} Topics", type="primary"):
        progress_bar = st.progress(0)
        status_container = st.empty()

        for i in range(num_topics):
            topic_data = st.session_state['fb_posts'][i]
            visual_data = st.session_state['generated_visuals'][i]
            
            # Use the scene metadata from the visual step to inform the copywriter
            scene_context = "No specific scene provided"
            if visual_data:
                # Retrieve the scene used in Step 3 (stored in metadata)
                # We assume the user didn't change it or we use the one from session state
                scene_context = st.session_state.get("meta_edits", [{}])[i].get("scene", "General news scene")

            status_container.text(f"Polishing Topic {i+1}/{num_topics}: {topic_data['topic']}...")

            for tone_key, tone_info in TONE_CONFIG.items():
                original_text = topic_data['tones'].get(tone_key, "")
                
                qa_prompt = f"""You are a world-class Senior Copy Editor for a high-traffic digital news outlet. 
Your goal is to take a draft Facebook post and make it flawless, high-impact, and grammatically perfect.

TUNE/TONE: {tone_info['instruction']}
VISUAL CONTEXT: The image accompanying this post depicts: {scene_context}

DRAFT TEXT:
{original_text}

TASK:
1. GRAMMAR & SYNTAX: Fix all spelling, punctuation, and grammatical errors.
2. SOURCE PRESERVATION: Do not change the facts or the source links.
3. REMOVE ALL MENTION OF THE ORIGINAL SOURCE IN THE FINAL OUTPUT. The final text should be polished and ready to post, without any reference to the original draft or source.

OUTPUT:
Return ONLY the polished post text, almost 1:1. No commentary, no 'Here is the revised version', no markdown. Just the raw text.
"""
                try:
                    response = generate_with_retry(
                        client=client,
                        model=model_name,
                        contents=qa_prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                            temperature=1.0,
                        )
                    )
                    polished_text = clean_llm_output(response.text).strip()
                    
                    # Update the post in session state
                    st.session_state['fb_posts'][i]['tones'][tone_key] = polished_text
                except Exception as e:
                    st.error(f"Error polishing {tone_key} for topic {i+1}: {e}")

            progress_bar.progress((i + 1) / num_topics)
        
        status_container.text("✅ Quality Assurance Complete! All posts polished.")
        st.success("All posts have been scanned and corrected. You can now proceed to Step 5.")

    # Review Area
    st.divider()
    st.subheader("🔍 Final Review")
    for i, topic_data in enumerate(st.session_state['fb_posts']):
        with st.expander(f"Topic {i+1}: {topic_data['topic']}"):
            tabs = st.tabs(["🔥 Viral", "📰 News", "😏 Satire"])
            for tab, tone_key in zip(tabs, TONE_CONFIG.keys()):
                with tab:
                    # Allow manual final tweaks
                    current_val = topic_data['tones'].get(tone_key, "")
                    updated_val = st.text_area(
                        f"Final Polish ({tone_key})", 
                        value=current_val, 
                        height=250, 
                        key=f"qa_edit_{i}_{tone_key}"
                    )
                    st.session_state['fb_posts'][i]['tones'][tone_key] = updated_val