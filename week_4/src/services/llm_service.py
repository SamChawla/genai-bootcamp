# src/services/llm_service.py
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from src.llms.groqllm import GroqLLM

class LLMService:
    def __init__(self, config: dict | None = None):
        # keep primitives only in config (strings) to avoid hash issues
        self.config = config or {}

    @st.cache_resource
    def get_llm(_self, api_key: str | None = None, model: str = "groq-1", **kwargs):
        """
        Note: first arg is `_self` (leading underscore) so Streamlit won't
        try to hash the instance. Only primitive args (api_key, model) are hashed.
        """
        final_key = api_key or _self.config.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        llm = GroqLLM(api_key=final_key, model=model, **kwargs)
        # ensure client initialized now (optional)
        llm.init_client()
        return llm
