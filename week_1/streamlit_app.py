"""
Streamlit Frontend for Groq LLM Application
"""
import streamlit as st
from main import LLMApp

# page configuration
st.set_page_config(
    page_title="Simple LLM Chat Application",
    page_icon="🤖",
    layout="centered"
)

# initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_app" not in st.session_state:
    st.session_state.llm_app = None

# Title and description
st.title("Groq LLM Chat Application")
st.markdown("Chat with a powerful LLM from Groq. Please enter your Groq API key in the sidebar to get started. You can get one at https://console.groq.com/")

# Implement sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API key input
    api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key")
    if not api_key:
        api_key = LLMApp().api_key

    # Model selection
    model = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-120b",
        ],
        help="Select the model to use"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Select a value to control response randomness. Higher values make output more random."
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256,
        help="Set the response length"
    )

    system_prompt = st.text_area(
        "System Prompt (Optional)",
        placeholder="You are a helpful assistant...",
        help = "Set the context and behaviour of the assistant"
    )

    # Clear chat button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        # if st.session_state.llm_app:
        #     st.session_state.llm_app.clear_history()

        st.rerun()

if st.session_state.llm_app is None:
    try:
        st.session_state.llm_app = LLMApp(api_key=api_key, model=model)
    except Exception as e:
        st.error(f"Error initializing LLM App: {str(e)}")

# display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar")

    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": f"{prompt}"
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # get assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.llm_app.chat(
                        user_message=prompt,
                        system_prompt=system_prompt if system_prompt else None,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    st.markdown(response)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"{response}"
                        }
                    )
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


