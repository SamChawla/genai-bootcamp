# src/graphs/graph_builder.py
"""
Simple orchestration layer that composes the BlogGenerator and returns output.
In the original repository LangGraph was used; this module keeps the same
logical separation without requiring the LangGraph runtime.
"""

from __future__ import annotations
import logging
from typing import Optional

from src.llms.groq_client import GroqClient
from src.nodes.blog_node import BlogGenerator
from src.states.blog_state import BlogState

logger = logging.getLogger(__name__)


class BlogGraphBuilder:
    """
    Orchestrates the two-step flow: title -> body.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        self.client = GroqClient(api_key=api_key, default_model=model)
        self.generator = BlogGenerator(self.client)

    def build_and_run(self, topic: str, approx_words: int = 700) -> BlogState:
        logger.info("Running blog graph for topic: %s", topic)
        return self.generator.generate_blog(topic=topic, approx_words=approx_words)
