# streamlit_app.py
import os
import streamlit as st

from src.llms.groqllm import GroqLLM

# Module-level cached factory that takes no unhashable args.
# Streamlit will cache this resource and reuse it between reruns.
@st.cache_resource
def get_cached_llm(api_key: str = None, model: str = "groq-1"):
    """
    Note: streamlit's cache_resource will hash the *arguments*.
    We are using only primitives (strings) here so hashing is safe.
    If you prefer to not include args, call get_cached_llm() without args
    and let it read env vars inside GroqLLM.
    """
    # prefer explicit param if provided, else environment variable
    api_key = api_key or os.getenv("GROQ_API_KEY")
    return GroqLLM(api_key=api_key, model=model)

# Example wrapper service (optional)
class LLMService:
    def __init__(self, api_key: str | None = None, model: str = "groq-1"):
        # store simple primitives only
        self.api_key = api_key
        self.model = model

    def get_llm(self):
        # This calls the cached module-level factory and passes only strings (hashable)
        return get_cached_llm(api_key=self.api_key, model=self.model)

# -------------------------
# Streamlit UI
# -------------------------
st.title("My Streamlit + Groq Demo")

# Create service (use env or explicit)
service = LLMService(api_key=None, model="groq-1")  # leave api_key None to use env var

try:
    llm = service.get_llm()
except Exception as ex:
    st.error(f"Failed to initialize LLM: {ex}")
    st.stop()

# simple chat UI
prompt = st.text_area("Prompt", value="Hello, tell me a one-line greeting.")
if st.button("Send"):
    with st.spinner("Calling LLM..."):
        resp = llm.chat(prompt)
    st.write(resp)
