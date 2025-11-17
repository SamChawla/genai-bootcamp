# src/states/blog_state.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BlogState:
    """
    Dataclass holding generated blog data.
    """
    topic: str
    title: str
    body: str

    def to_markdown(self) -> str:
        """Return a markdown representation suitable for download."""
        md = f"# {self.title}\n\n"
        md += f"**Topic:** {self.topic}\n\n"
        md += self.body
        return md
