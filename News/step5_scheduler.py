# step5_scheduler.py
import streamlit as st
from io import BytesIO
import re
from step2_rewriter import TONE_CONFIG

def render_step5():
    st.header("Step 5: Final Post & Visual Compiler")
    st.write("Review, copy, and download your fully compiled Facebook posts paired perfectly with their respective themed visuals.")

    if 'fb_posts' not in st.session_state or 'generated_visuals' not in st.session_state:
        st.warning("⚠️ Please complete all previous steps first.")
        return

    if not any(st.session_state['generated_visuals']):
        st.warning("⚠️ No visuals have been generated yet. Please go back to Step 3.")
        return

    tabs = st.tabs([
        f"🔥 {TONE_CONFIG['tagalog_viral']['display_name']} ({TONE_CONFIG['tagalog_viral']['default_label']})",
        f"📰 {TONE_CONFIG['straight_news']['display_name']} ({TONE_CONFIG['straight_news']['default_label']})",
        f"😏 {TONE_CONFIG['satirical']['display_name']} ({TONE_CONFIG['satirical']['default_label']})"
    ])

    for tab, (tone_key, config) in zip(tabs, TONE_CONFIG.items()):
        theme_key = config["theme"]
        
        with tab:
            st.subheader(f"Compiled Channel: {config['display_name']}")
            st.caption(f"Ready to publish to your Facebook Page for the **{config['default_label']}** segment.")
            st.divider()
            
            count = 0
            for i, (post_data, visual_data) in enumerate(zip(st.session_state['fb_posts'], st.session_state['generated_visuals'])):
                if visual_data is None:
                    continue 
                
                count += 1
                with st.container(border=True):
                    st.markdown(f"#### Topic {i+1}: {post_data['topic']}")
                    
                    col_img, col_txt = st.columns([1, 1.5])
                    
                    with col_img:
                        img = visual_data["images"].get(theme_key)
                        if img:
                            st.image(img, use_container_width=True)
                            buf = BytesIO()
                            img.save(buf, format="PNG")
                            buf.seek(0)
                            safe_name = re.sub(r"[^\w]", "_", post_data["topic"])[:30]
                            file_name = f"topic{i+1}_{theme_key}_{safe_name}.png"
                            
                            st.download_button(
                                label=f"⬇️ Download {config['default_label']} Image",
                                data=buf,
                                file_name=file_name,
                                mime="image/png",
                                key=f"dl_step5_{i}_{theme_key}",
                                use_container_width=True
                            )
                        else:
                            st.warning("⚠️ Image missing for this theme.")
                            
                    with col_txt:
                        post_text = post_data["tones"].get(tone_key, "[Text generation missing]")
                        st.text_area(
                            "Final Polished Facebook Caption", 
                            value=post_text, 
                            height=320, 
                            key=f"txt_step5_{i}_{tone_key}"
                        )
                        st.caption(f"Word count: {len(post_text.split())} | Character count: {len(post_text)}")
                        
            if count == 0:
                st.info("No completed visuals found. Please go back to Step 3.")