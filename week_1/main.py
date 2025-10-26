"""
Simple LLM Application using Groq API
"""
from groq import Groq
from appconfig import env_config

class LLMApp:

    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        
        """
        Initialize the LLM application
        
        Args:
            api_key: Groq API key (if None, reads from GROQ_API_KEY env var)
            model: Model to use for completions
            conversation_history: List of converstaions in a session
        """

        self.api_key = api_key or env_config.groq_api_key
        if not self.api_key:
            raise ValueError("Groq API key must be provided or set in `GROQ_API_KEY` environment variable")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.conversation_history = []

    def chat(self, user_message, system_prompt=None, temperature=0.5, max_tokens=1024):
        """
        Send a message and get a response
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            
        Returns:
            The assistant's response text
        """

        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": f"{system_prompt}"
                }
            )

        # Add conversation history
        if self.conversation_history:
            messages.extend(self.conversation_history)
        
        # Add current user's message
        messages.append(
            {
                "role": "user",
                "content": f"{user_message}"
            }
        )

        # Make LLM call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract response text
        assistant_message = response.choices[0].message.content

        return assistant_message
    
    # def clear_history(self):
    #     """Clear the conversation history"""
    #     self.conversation_history = []
    
    # def get_history(self):
    #     """Get the current conversation history"""
    #     return self.conversation_history

if __name__=="__main__":

    # Initialize the app
    app = LLMApp()

    # while True:
    message = input(f"What do you want to ask: ")
    response = app.chat(message)
    print(f"\nAssistant Response: {response}\n")

