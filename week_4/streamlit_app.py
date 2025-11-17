# streamlit_app.py
"""
Streamlit UI for AI Blog Generator with real-time preview and optional auto-generate.

Features:
 - Uses placeholder containers so Preview updates in the same request (no rerun).
 - Optional "Auto-generate" — when the topic changes, the app generates automatically.
 - Offline demo mode fallback for UI testing when Groq/API key isn't available.
 - Diagnostic info and robust error handling.
"""

from __future__ import annotations
import os
import logging
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Local modules
from src.graphs.graph_builder import BlogGraphBuilder
from src.states.blog_state import BlogState
try:
    from src.llms.groq_client import GroqClientError
except Exception:
    class GroqClientError(Exception):  # fallback
        pass

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_blog_app")

# Constants
DEFAULT_MODEL = "llama-3.1-8b-instant"
ALT_MODEL = "llama-3.1-70b-versatile"
APP_TITLE = "AI Blog Generator"
APP_SUBTITLE = "Live Preview · Auto-generate option · Demo fallback"

# Simple CSS
CARD_CSS = """
<style>
[data-testid="stAppViewContainer"] > .main {{background: linear-gradient(180deg, #0f172a 0%, #07102a 100%); color: #e6eef8;}}
.card {{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius: 12px; padding: 18px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); border: 1px solid rgba(255,255,255,0.03); }}
h1,h2,h3 {{ color: #e6eef8; }}
.stTextInput>div>div>input, .stTextArea>div>div>textarea {{ background: rgba(255,255,255,0.03) !important; color: #e6eef8 !important; border-radius: 8px !important; padding: 10px !important; border: 1px solid rgba(255,255,255,0.04) !important; }}
.stButton>button {{ background: linear-gradient(90deg,#7c3aed,#06b6d4) !important; color: white !important; border-radius: 10px !important; padding: 8px 14px !important; font-weight: 600; box-shadow: 0 6px 18px rgba(92,57,158,0.16); border: none !important; }}
.small-muted {{ color: #9fb0c8; font-size: 13px; }}
</style>
"""

# ----------------------
# Offline demo generator
# ----------------------
def demo_generate(topic: str, approx_words: int = 700, tone: str = "Professional", audience: Optional[str] = None) -> BlogState:
    if not topic or not topic.strip():
        topic = "Demo: Why AI is transforming software development"
    title = f"Demo: {topic.strip().capitalize()}"
    intro = f"**Introduction**\n\nThis is a demo blog about *{topic.strip()}*. It demonstrates the layout and structure of a generated post.\n\n"
    body_sections = [
        intro,
        "## Background\n\nA short background section explaining core ideas.\n",
        "## Key Points\n\n- Point 1: Summary of the topic.\n- Point 2: Examples and context.\n- Point 3: Practical advice.\n",
        "## How to apply this\n\nSome actionable steps:\n1. Step one\n2. Step two\n3. Step three\n",
        "## Conclusion\n\nA short wrap-up with final takeaways.\n",
    ]
    body_text = "\n\n".join(body_sections)
    repeat = max(1, min((approx_words // 200), 6))
    body_text = ("\n\n".join([body_text] * repeat)).strip()
    note = f"\n\n*Tone: {tone}*"
    if audience:
        note += f"  *Audience: {audience}*"
    body_text += note
    return BlogState(topic=topic.strip(), title=title, body=body_text)

# ----------------------
# Helpers
# ----------------------
def get_builder(api_key: Optional[str], model: str) -> BlogGraphBuilder:
    return BlogGraphBuilder(api_key=api_key or None, model=model)

def reset_generation_state():
    for k in ("generated_blog", "last_topic", "last_error", "is_generating"):
        if k in st.session_state:
            del st.session_state[k]

def generate_and_show_preview(
    topic: str,
    approx_words: int,
    tone: str,
    audience: Optional[str],
    api_key_input: str,
    model: str,
    offline_demo: bool,
    preview_placeholder: st.delta_generator.DeltaGenerator,
):
    """
    Performs generation (demo or real) and updates preview_placeholder incrementally so preview updates
    in the same request (real-time feeling).
    """
    # Avoid parallel runs
    if st.session_state.get("is_generating"):
        return
    st.session_state["is_generating"] = True
    st.session_state.pop("last_error", None)
    try:
        preview_placeholder.markdown("### Generating... ⏳\n\nPlease wait — this may take a few seconds.")
        preview_placeholder.progress = None  # just ensure placeholder exists

        if offline_demo:
            blog_state = demo_generate(topic=topic, approx_words=approx_words, tone=tone, audience=audience)
            # show quickly
            preview_placeholder.markdown(f"### {blog_state.title}\n\n{blog_state.body}")
            st.session_state["generated_blog"] = blog_state
            st.session_state["last_topic"] = topic
            return

        # Real generation via Groq
        builder = get_builder(api_key=api_key_input or None, model=model)
        # show incremental progress states
        preview_placeholder.markdown("### Generating title...")
        # We cannot easily stream tokens without SDK streaming; we show logical steps and a progress bar
        progress_bar = preview_placeholder.empty()
        progress_bar.progress(10)
        blog_state = builder.build_and_run(topic=topic, approx_words=approx_words)
        progress_bar.progress(90)
        # final render
        preview_placeholder.markdown(f"### {blog_state.title}\n\n{blog_state.body}")
        st.session_state["generated_blog"] = blog_state
        st.session_state["last_topic"] = topic

    except GroqClientError as gce:
        st.session_state["last_error"] = str(gce)
        preview_placeholder.markdown(f"**Groq client error:** {gce}\n\nCheck Diagnostics in the sidebar.")
        logger.exception("GroqClientError during generation")
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        preview_placeholder.markdown(f"**Failed to generate blog:** {exc}\n\nCheck Diagnostics in the sidebar.")
        logger.exception("Unhandled exception during generation")
    finally:
        st.session_state["is_generating"] = False
        try:
            # remove temporary progress widget if exists
            progress_bar.empty()
        except Exception:
            pass

# ----------------------
# App main
# ----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    # Header
    h1, h2 = st.columns([6, 1])
    with h1:
        st.markdown(f"<h1 style='margin-bottom:6px'>{APP_TITLE}</h1>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
    with h2:
        st.image("AI_Blog_Post_Generator.jpg", width=220)
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Settings")
    api_key_from_env = os.getenv("GROQ_API_KEY", "")
    model = st.sidebar.selectbox("Model", [DEFAULT_MODEL, ALT_MODEL], index=0)
    api_key_input = st.sidebar.text_input("Groq API Key", value=api_key_from_env, type="password")
    approx_words = st.sidebar.slider("Approx. words", min_value=300, max_value=2500, value=700, step=100)
    show_markdown_preview = st.sidebar.checkbox("Show markdown preview", value=True)
    offline_demo_default = not bool(api_key_input.strip() or api_key_from_env.strip())
    offline_demo = st.sidebar.checkbox("Offline demo mode (no API key)", value=offline_demo_default)
    auto_generate = st.sidebar.checkbox("Auto-generate when topic changes", value=False)
    st.sidebar.markdown("---")
    with st.sidebar.expander("Diagnostics (useful for debugging)"):
        st.write("GROQ_API_KEY present in environment:", bool(api_key_from_env.strip()))
        st.write("API key entered in sidebar:", bool(api_key_input.strip()))
        st.write("Offline demo mode:", offline_demo)
        st.write("Auto-generate:", auto_generate)
        if "last_error" in st.session_state:
            st.markdown("**Last error:**")
            st.code(str(st.session_state["last_error"]))

    # Main layout
    left, right = st.columns([1, 1.4], gap="large")

    # Left: inputs
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Write a blog from a topic")
        topic = st.text_input("Topic", placeholder="E.g. The future of developer tooling in 2025", key="topic_input")
        tone = st.selectbox("Tone", ["Professional", "Conversational", "Friendly", "Technical"], index=0)
        audience = st.text_input("Target audience (optional)", placeholder="E.g. dev teams, CTOs")
        st.markdown("<div class='small-muted'>Refine tone or audience for better output.</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            generate_btn = st.button("Generate Blog Post")
        with col_b:
            reset_btn = st.button("Reset")
        st.markdown("</div>", unsafe_allow_html=True)

    # Right: preview placeholder (updates in the same request)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Preview")

        blog_state = st.session_state.get("generated_blog")

        if blog_state:
            # Show title + body
            st.markdown(f"### {blog_state.title}")
            st.markdown(blog_state.body)

            st.download_button(
                label="Download Markdown",
                data=blog_state.to_markdown().encode("utf-8"),
                file_name=f"{blog_state.title[:40].replace(' ', '_')}.md",
                mime="text/markdown"
            )
        else:
            st.markdown(
                "<em class='small-muted'>No blog generated yet — enter a topic and click Generate.</em>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)


    # Reset handler
    if reset_btn:
        reset_generation_state()
        st.success("Reset complete.")
        return

    # Manual generate handler (button)
    if generate_btn:
        if not topic.strip():
            st.error("Please enter a topic.")
        else:
            try:
                builder = get_builder(api_key_input or None, model)
                with st.spinner("Generating blog..."):
                    blog_state = builder.build_and_run(topic.strip(), approx_words)

                st.session_state["generated_blog"] = blog_state
                st.success("Blog generated successfully!")

            except Exception as e:
                st.error(f"Failed to generate blog: {e}")
                st.session_state["generated_blog"] = None

    # Auto-generate on topic change (if enabled)
    last_topic = st.session_state.get("last_topic_input")
    # If auto_generate enabled, topic is non-empty, and topic changed from last_topic, run generation
    if auto_generate and topic and topic.strip() and topic.strip() != (last_topic or "") and not st.session_state.get("is_generating"):
        # Save the last input so we don't loop continuously
        st.session_state["last_topic_input"] = topic.strip()
        generate_and_show_preview(
            topic=topic.strip(),
            approx_words=approx_words,
            tone=tone,
            audience=audience,
            api_key_input=api_key_input,
            model=model,
            offline_demo=bool(offline_demo),
            preview_placeholder=preview_placeholder,
        )
        return

    # Footer
    st.markdown("---")
    f1, f2 = st.columns([3, 1])
    with f1:
        st.markdown("Made with ❤️ — use `.env` for `GROQ_API_KEY` or enable Offline demo.")
    with f2:
        st.markdown("v1.2")

if __name__ == "__main__":
    main()
