# src/llms/groqllm.py

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class GroqLLM:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing in environment variables")

        # Initialize model
        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-70b-versatile",
            temperature=0.2
        )

    def invoke(self, message: str):
        try:
            response = self.llm.invoke(message)
            return response.content
        except Exception as e:
            return f"Error in GroqLLM: {e}"
