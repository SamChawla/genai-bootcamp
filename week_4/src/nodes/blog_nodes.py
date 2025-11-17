from src.states.blog_state import BlogState

class BlogNodes:
    """
    A class to hold all the node/agent functions for the graph.
    """
    def __init__(self, llm):
        self.llm = llm

    def title_agent(self, state: BlogState):
        """
        Generates 5 potential blog post titles.
        """
        prompt = f"""
        You are an expert SEO and blog title writer. 
        Your goal is to generate 5 creative, click-worthy, and SEO-friendly titles 
        for a blog post on the topic: "{state['topic']}".
        
        Return them as a numbered list.

        Example:
        1. Title one
        2. Title two
        3. Title three
        4. Title four
        5. Title five
        """
        response = self.llm.invoke(prompt)
        
        # Parse the numbered list of titles
        titles = [
            t.strip().split('. ', 1)[-1] 
            for t in response.content.split('\n') 
            if t.strip() and '.' in t
        ]
        return {"titles": titles}

    def outline_agent(self, state: BlogState):
        """
        Selects the best title and generates a blog post outline.
        """
        prompt = f"""
        You are a blog architect. 
        Given the topic "{state['topic']}" and these 5 potential titles:
        {state['titles']}
        
        1. Select the BEST and most engaging title from the list.
        2. Generate a detailed, section-by-section outline for a blog post 
           based on that *single* selected title.
        
        Return your response *only* in this format:
        Title: [Your selected title]
        Outline:
        - Introduction: [Brief intro]
        - Section 1: [Name of section]
        - Section 2: [Name of section]
        - Section 3: [Name of section]
        - Conclusion: [Brief conclusion]
        """
        response = self.llm.invoke(prompt)
        
        try:
            parts = response.content.split("Outline:")
            title = parts[0].replace("Title:", "").strip()
            outline = parts[1].strip()
        except Exception:
            title = state['titles'][0]
            outline = "Introduction\n- Section 1\n- Conclusion"
            
        return {"selected_title": title, "outline": outline}

    def writer_agent(self, state: BlogState):
        """
        Writes the first draft of the blog post.
        """
        prompt = f"""
        You are a senior content writer.
        Write a full, high-quality blog post (in Markdown format) 
        based on this title and outline:
        
        Title: {state['selected_title']}
        Outline:
        {state['outline']}
        
        The post should be engaging, informative, and well-structured.
        Start the post directly with the content. Do not repeat the title.
        """
        response = self.llm.invoke(prompt)
        return {"draft": response.content}

    def editor_agent(self, state: BlogState):
        """
        Provides a critique of the draft.
        """
        prompt = f"""
        You are a critical editor. Review this blog post draft.
        
        Draft:
        {state['draft']}
        
        Provide a list of 3-5 high-level critique points for improvement.
        Focus on flow, clarity, missing information, and engagement.
        Be constructive.
        
        Return *only* the critique as a bulleted list.
        """
        response = self.llm.invoke(prompt)
        return {"critique": response.content}

    def final_blog_node(self, state: BlogState):
        """
        Rewrites the draft based on the critique.
        """
        prompt = f"""
        You are a master content writer.
        Your job is to rewrite the following draft based on the editor's critique.
        
        Original Draft:
        {state['draft']}
        
        Editor's Critique:
        {state['critique']}
        
        Produce the final, polished, and complete blog post (in Markdown).
        Make sure to incorporate all feedback from the critique.
        """
        response = self.llm.invoke(prompt)
        return {"final_blog": response.content}