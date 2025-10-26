from dotenv import load_dotenv, find_dotenv
import os
import re
import traceback
import time
import streamlit as st

# Load .env BEFORE importing appconfig/main so env vars are in os.environ for imported modules
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path, override=False)

from app_config import env_config
from main import LLMApp, BOT_NAME

# Debug toggle (set STREAMLIT_DEBUG=true in .env or environment)
DEBUG = os.getenv("STREAMLIT_DEBUG", "false").lower() in ("1", "true", "yes")

# Page config
st.set_page_config(page_title="Simple LLM Chat Application", page_icon="🤖", layout="centered")

# Session state defaults
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_app" not in st.session_state:
    st.session_state.llm_app = None
if "ui_config" not in st.session_state:
    st.session_state.ui_config = {"api_key": None, "model": None}

# Helpers
ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')
CODE_FENCE_RE = re.compile(r'```(?:[\w+-]*)\n(.*?)```', re.DOTALL)

def sanitize_response_for_ui(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    text = ANSI_ESCAPE_RE.sub('', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text

def render_text_with_code(container, text: str):
    if not text:
        container.markdown("")
        return
    pos = 0
    for m in CODE_FENCE_RE.finditer(text):
        start, end = m.span()
        if start > pos:
            before = text[pos:start]
            if before.strip():
                container.markdown(before)
        code_content = m.group(1).rstrip("\n")
        container.code(code_content)
        pos = end
    if pos < len(text):
        trailing = text[pos:]
        if trailing.strip():
            container.markdown(trailing)

# UI Header
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
        if DEBUG:
            st.exception(e)

# Display existing messages
for message in st.session_state.messages:
    role = message.get("role", "user")
    content = message.get("content", "") or ""
    content = sanitize_response_for_ui(content)
    with st.chat_message(role):
        if role == "assistant":
            render_text_with_code(st, content)
        else:
            st.markdown(content)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    if not api_key:
        st.warning("Please enter your API key in the sidebar")
    else:
        # Append and render user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Ensure we have an initialized LLM client before calling
        if not st.session_state.llm_app:
            try:
                st.session_state.llm_app = LLMApp(api_key=api_key, model=model)
                st.session_state.ui_config["api_key"] = api_key
                st.session_state.ui_config["model"] = model
            except Exception as e:
                st.error(f"Failed to (re)initialize LLM client: {e}")
                if DEBUG:
                    st.exception(e)
                st.session_state.messages.append({"role": "assistant", "content": "Error: LLM client not available."})
                with st.chat_message("assistant"):
                    st.markdown("Error: LLM client not available.")
                st.stop()

        # Show assistant spinner and call LLM
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Debug: show conversation history before call
                    if DEBUG:
                        try:
                            hist_before = st.session_state.llm_app.get_history()
                        except Exception:
                            hist_before = getattr(st.session_state.llm_app, "conversation_history", None)
                        st.write("LLMApp.conversation_history (before):", hist_before)

                    # First attempt
                    raw_response = st.session_state.llm_app.chat(
                        user_message=prompt,
                        system_prompt=system_prompt if system_prompt else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    if DEBUG:
                        st.write("RAW RESPONSE (first attempt repr):", repr(raw_response))

                    # If empty or whitespace, try one retry (helps transient API issues)
                    response_text = sanitize_response_for_ui(raw_response)
                    if not response_text.strip():
                        # brief pause before retry
                        time.sleep(0.3)
                        if DEBUG:
                            st.write("First response empty — retrying once...")
                        try:
                            raw_response_retry = st.session_state.llm_app.chat(
                                user_message=prompt,
                                system_prompt=system_prompt if system_prompt else None,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                        except Exception as e_retry:
                            raw_response_retry = None
                            if DEBUG:
                                st.write("Retry raised exception:", e_retry)
                        if DEBUG:
                            st.write("RAW RESPONSE (retry repr):", repr(raw_response_retry))
                        response_text = sanitize_response_for_ui(raw_response_retry)

                    # If still empty after retry, use fallback message
                    if not response_text.strip():
                        fallback_msg = "Sorry — I didn't get a reply. Please try again or adjust the prompt."
                        if DEBUG:
                            # include raw_repr for debugging (don't reveal keys in production)
                            st.write("Empty response details:", {
                                "first_raw": repr(raw_response),
                            })
                        response_text = fallback_msg

                    # Render and persist
                    render_text_with_code(st, response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                    if DEBUG:
                        try:
                            hist_after = st.session_state.llm_app.get_history()
                        except Exception:
                            hist_after = getattr(st.session_state.llm_app, "conversation_history", None)
                        st.write("LLMApp.conversation_history (after):", hist_after)

                except Exception as e:
                    # Show exception details in UI
                    st.error(f"Error generating response: {str(e)}")
                    if DEBUG:
                        st.exception(e)
                        st.write("Full traceback:")
                        st.write(traceback.format_exc())
                    # Persist an error assistant message so UI remains consistent
                    err_msg = f"Error generating response: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})