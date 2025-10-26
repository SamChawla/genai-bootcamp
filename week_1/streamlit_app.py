# streamlit._app.py
from dotenv import load_dotenv, find_dotenv
import os
import streamlit as st

# Load .env BEFORE importing appconfig/main so env vars are in os.environ for imported modules
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path, override=False)

from app_config import env_config
from main import LLMApp, BOT_NAME

# Page config
st.set_page_config(page_title="Simple LLM Chat Application", page_icon="🤖", layout="centered")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_app" not in st.session_state:
    st.session_state.llm_app = None
if "ui_config" not in st.session_state:
    st.session_state.ui_config = {"api_key": None, "model": None}

# UI
st.title("LLM Chat Application")
st.subheader(f"Assistant: {BOT_NAME}")
st.markdown("Enter your API key (Groq or OpenAI) in the sidebar or set the corresponding env var.")

with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("API Key (Groq or OpenAI)", type="password", help="Enter API key for selected model/provider")
    model = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "openai/gpt-oss-120b",
        ],
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.5, 0.1)
    max_tokens = st.slider("Max Tokens", 256, 2048, 1024, 256)
    system_prompt = st.text_area("System Prompt (Optional)", placeholder="You are a helpful assistant...")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.llm_app:
            st.session_state.llm_app.clear_history()
        st.rerun()

# Determine fallback key using env_config
is_openai_model = model in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}
fallback_key = env_config.openai_api_key if is_openai_model else env_config.groq_api_key
api_key = api_key_input.strip() if api_key_input else fallback_key

# Reinitialize LLM app if model/api_key changed
need_reinit = (
    st.session_state.llm_app is None
    or st.session_state.ui_config.get("api_key") != api_key
    or st.session_state.ui_config.get("model") != model
)

if need_reinit:
    try:
        if not api_key:
            st.warning("No API key provided for the selected provider. Provide it in the sidebar or set the corresponding env var.")
        st.session_state.llm_app = LLMApp(api_key=api_key, model=model)
        st.session_state.ui_config["api_key"] = api_key
        st.session_state.ui_config["model"] = model
    except Exception as e:
        st.error(f"Error initializing LLM App: {str(e)}")

# display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
if prompt := st.chat_input("Type your message here..."):
    if not api_key:
        st.warning("Please enter your API key in the sidebar")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.llm_app.chat(
                        user_message=prompt,
                        system_prompt=system_prompt if system_prompt else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")