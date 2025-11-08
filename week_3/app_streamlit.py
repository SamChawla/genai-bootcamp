# app_streamlit.py
import os
import traceback
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Load .env
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path, override=False)

# Modular UI imports
from modules.news import NewsUI
from modules.youtube import YouTubeUI
from modules.voice import VoiceUI

# LLM wrapper & config
try:
    from main import LLMApp, BOT_NAME
except Exception:
    LLMApp = None
    BOT_NAME = "Assistant"

try:
    from app_config import env_config
except Exception:
    env_config = None

st.set_page_config(page_title="Summarizer + Voice Assist Hub", layout="wide")
st.title("Summarizer + Voice Assist Hub")
st.markdown("Single UI for News, YouTube, and Voice-RAG workflows. Select a mode from the sidebar.")

# Sidebar
with st.sidebar:
    st.header("Global Configuration")
    mode = st.radio("Mode", ["News Summarizer", "YouTube Summarizer", "Voice Assistant (RAG)"])
    st.markdown("---")

    provider = st.selectbox("Provider / Model Family", ["openai", "ollama", "groq", "local"])
    
    try:
        from utils.ollama import list_local_models
    except Exception:
        def list_local_models():
            return []

    # model_suggest = "gpt-4o-mini" if provider == "openai" else "llama-3.3-70b-versatile"
    # model = st.text_input("Model name", value=model_suggest)
    
    if provider == "openai":
    # prefer the OPENAI_MODELS set from main if available, else fallback list
        try:
            from main import OPENAI_MODELS as _OPENAI_MODELS
            openai_models = sorted(list(_OPENAI_MODELS))
        except Exception:
            openai_models = ["gpt-5", "gpt-5-mini", "gpt-4o-mini", "gpt-4o", "gpt-4"]

        model = st.selectbox("Model name", openai_models, index=openai_models.index("gpt-4o-mini") if "gpt-4o-mini" in openai_models else 0)

    elif provider == "ollama":
        # list local ollama models if possible
        ollama_models = list_local_models() or []
        DEFAULT_OLLAMA = ["llama-3.3-70b-versatile", "llama3.2:latest"]
        model_options = ollama_models or DEFAULT_OLLAMA

        if ollama_models:
            st.caption(f"✅ Detected {len(ollama_models)} local Ollama model(s).")
            model = st.selectbox("Model name", model_options)
        else:
            st.caption("⚠️ No local Ollama models detected. Use a default or enter custom name.")
            model = st.selectbox("Model name (defaults)", model_options)
            # allow typing custom override if needed
            custom = st.text_input("Or type Ollama model name (optional)", value="")
            if custom and custom.strip():
                model = custom.strip()
    
    api_key_input = st.text_input("API Key (session only)", type="password", help="Optional: override env key for this session")
    st.markdown("---")

    st.subheader("Embeddings & Retriever")
    embedding_type = st.selectbox("Embedding backend", ["openai", "chroma", "nomic", "hf"], index=0)
    top_k = st.slider("Retriever top-k", 1, 20, 5)
    chunk_size = st.number_input("Chunk size", min_value=128, max_value=4000, value=1000, step=64)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=16)

    st.markdown("---")
    st.subheader("Other")
    elevenlabs_key = st.text_input("ElevenLabs API Key (for TTS)", type="password")

# Resolve API key
if api_key_input:
    api_key = api_key_input.strip()
else:
    api_key = None
    if env_config:
        api_key = env_config.openai_api_key if provider == "openai" else env_config.groq_api_key

# Initialize LLMApp (cached in session if available)
def init_llm_app_once(api_key_local, model_name):
    if "llm_app" not in st.session_state or st.session_state.get("llm_app_model") != model_name or st.session_state.get("llm_app_api_key") != api_key_local:
        try:
            if LLMApp is None:
                st.warning("LLMApp not available (main.py). Some flows will still use LangChain wrappers inside modules.")
                st.session_state.llm_app = None
            else:
                st.session_state.llm_app = LLMApp(api_key=api_key_local, model=model_name)
                st.session_state.llm_app_model = model_name
                st.session_state.llm_app_api_key = api_key_local
        except Exception as e:
            st.error(f"Failed to initialize LLMApp: {e}")
            if st.sidebar.checkbox("Show traceback"):
                st.code(traceback.format_exc())
            st.session_state.llm_app = None

init_llm_app_once(api_key, model)

# Route
if mode == "News Summarizer":
    NewsUI.render(api_key=api_key, provider=provider, model=model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
elif mode == "YouTube Summarizer":
    YouTubeUI.render(api_key=api_key, provider=provider, model=model, embedding_type=embedding_type)
else:
    VoiceUI.render(api_key=api_key, provider=provider, model=model, elevenlabs_key=elevenlabs_key)
