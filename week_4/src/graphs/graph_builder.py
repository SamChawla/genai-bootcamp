# File: src/graphs/graph_builder.py
import streamlit as st
from langgraph.graph import StateGraph, START, END
from src.states.blog_state import BlogState
from src.nodes.blog_nodes import BlogNodes

class GraphBuilder:
    
    def __init__(self, llm):
        self.llm = llm
        self.blog_nodes_obj = BlogNodes(self.llm)

    @st.cache_resource
    def build_graph(_self): # Use _self for st.cache_resource
        """
        Assembles the nodes into a compiled Langgraph.
        Caches the compiled graph.
        """
        workflow = StateGraph(BlogState)
        
        # Add nodes
        workflow.add_node("title_agent", _self.blog_nodes_obj.title_agent)
        workflow.add_node("outline_agent", _self.blog_nodes_obj.outline_agent)
        workflow.add_node("writer_agent", _self.blog_nodes_obj.writer_agent)
        workflow.add_node("editor_agent", _self.blog_nodes_obj.editor_agent)
        workflow.add_node("final_blog_node", _self.blog_nodes_obj.final_blog_node)
        
        # Add edges
        workflow.add_edge(START, "title_agent")
        workflow.add_edge("title_agent", "outline_agent")
        workflow.add_edge("outline_agent", "writer_agent")
        workflow.add_edge("writer_agent", "editor_agent")
        workflow.add_edge("editor_agent", "final_blog_node")
        workflow.add_edge("final_blog_node", END)
        
        # Compile the graph
        return workflow.compile()