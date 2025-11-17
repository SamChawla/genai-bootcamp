from typing import TypedDict, List

class BlogState(TypedDict):
    """
    This is the "memory" or "state" that all agents share and update.
    """
    topic: str              # The initial blog post topic from the user
    titles: List[str]       # A list of potential titles
    selected_title: str     # The single title chosen by the outline agent
    outline: str            # The blog post outline
    draft: str              # The first draft of the blog
    critique: str           # The editor's critique of the draft
    final_blog: str         # The final, polished blog post