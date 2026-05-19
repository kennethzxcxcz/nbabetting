# main_app.py
import streamlit as st
from google import genai
from config import API_KEY
from step1_curate import render_step1
from step2_rewriter import render_step2
from step3_visuals import render_step3
from step4_qa import render_step4  # New Import
from step5_scheduler import render_step5 # Updated Import

st.set_page_config(page_title="Automated FB News Channel", layout="wide")

# Initialize GenAI client
client = genai.Client(api_key=API_KEY)

# Sidebar
st.sidebar.title("⚙️ LLM Configuration")
st.sidebar.write("Google Generative Language API (google.genai)")
model_name = st.sidebar.text_input("Rewriter & Curator Model", value="gemma-4-31b-it")

st.title("📰 Automated FB News Channel Pipeline")

st.sidebar.title("Pipeline Steps")
step = st.sidebar.radio("Navigate", [
    "1. Scrape & Curate Top 10",
    "2. Facebook Post Rewriter",
    "3. Visuals Generator (Nano Banana 2)",
    "4. Copywriting & QA Polish",
    "5. Final Post & Visual Compiler"
])

# Route to appropriate step
if step == "1. Scrape & Curate Top 10":
    render_step1(client, model_name)
elif step == "2. Facebook Post Rewriter":
    render_step2(client, model_name)
elif step == "3. Visuals Generator (Nano Banana 2)":
    render_step3(client, model_name)
elif step == "4. Copywriting & QA Polish":
    render_step4(client, model_name)
elif step == "5. Final Post & Visual Compiler":
    render_step5()