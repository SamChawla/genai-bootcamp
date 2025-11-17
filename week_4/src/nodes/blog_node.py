# src/nodes/blog_node.py
"""
Blog generation node(s): produce title and full blog text using the LLM client.
This module contains pure logic that is independent of Streamlit; it can be unit tested.
"""

from __future__ import annotations
from typing import Tuple
import logging

from src.states.blog_state import BlogState
from src.llms.groq_client import GroqClient, GroqClientError

logger = logging.getLogger(__name__)


class BlogGenerator:
    """
    Encapsulates logic for generating a blog title and body.
    """

    TITLE_PROMPT_TEMPLATE = "Write a catchy SEO-friendly blog title for the topic: {topic}"
    BODY_PROMPT_TEMPLATE = (
        "Write a long-form, well-structured blog post (approx {approx_words} words) "
        "about: {topic}\n\nMake it engaging, include headings, subheadings, "
        "a short intro, conclusion, and a list of actionable takeaways. Use markdown."
    )

    def __init__(self, llm_client: GroqClient):
        self.llm = llm_client

    def generate_title(self, topic: str) -> str:
        prompt = self.TITLE_PROMPT_TEMPLATE.format(topic=topic)
        logger.debug("Generating title with prompt: %s", prompt)
        try:
            raw = self.llm.generate(prompt=prompt, max_tokens=40, temperature=0.1)
        except GroqClientError:
            logger.exception("Groq client error while generating title.")
            raise
        title = self.llm.clean_text(raw)
        logger.debug("Generated title: %s", title)
        return title

    def generate_body(self, topic: str, approx_words: int = 700) -> str:
        prompt = self.BODY_PROMPT_TEMPLATE.format(topic=topic, approx_words=approx_words)
        logger.debug("Generating body with prompt length %d", len(prompt))
        try:
            raw = self.llm.generate(prompt=prompt, max_tokens=approx_words * 2, temperature=0.2)
        except GroqClientError:
            logger.exception("Groq client error while generating body.")
            raise
        body = self.llm.clean_text(raw)
        logger.debug("Generated body length: %d", len(body))
        return body

    def generate_blog(self, topic: str, approx_words: int = 700) -> BlogState:
        if not topic or not topic.strip():
            raise ValueError("Topic must be a non-empty string.")
        topic = topic.strip()
        title = self.generate_title(topic)
        body = self.generate_body(topic, approx_words=approx_words)
        return BlogState(topic=topic, title=title, body=body)
