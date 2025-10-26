# Simple LLM Q&A Application

In this week's expert session, we implemented a simple Large Language Model (LLM) application with a Streamlit frontend, allowing users to interact with various open-source models via the Groq API.

## 1\. Overview

The application is a basic question answering system that uses the Groq API to access and utilize open-source language models. It features a command-line interface (CLI) for core functionality and a Streamlit-based web interface for user interaction and configuration.

## 2\. Features

  * **Groq API Integration:** Connects to the Groq API to access a range of open-source LLMs.
  * **Configurable Models:** Users can select different LLMs (e.g., Llama 3.1 8B, Llama 3.3 70B, OpenAI OSS 20B) from the Groq API Client.
  * **API Key Input:** Option to input a Groq API key directly in the Streamlit frontend.
  * **Adjustable Temperature:** Controls the randomness of model responses using a temperature slider.
  * **Output Control:** Sets the maximum length of the model's generated responses.
  * **Prompt Customization:** Defines a system prompt to guide the LLM's behavior and context.
  * **Chat History Clearing:** A button to clear the current chat history in the Streamlit application.

## 3\. Tools & Frameworks Used

  * **Python:** The primary programming language.
  * **Groq API:** For accessing open-source LLMs.
  * **Streamlit:** For building the interactive web frontend.
  * **`python-dotenv`:** For loading environment variables (API keys).
  * **`uv` (or `pip`, `conda`, `poetry`):** For virtual environment and dependency management.

## 4\. Setup and Installation

### 4.1. Prerequisites

  * Python 3.x installed (In the workshop session, we specifically used Python 3.11)
  * A Groq API key (sign up at `https://console.groq.com/` to get one).

### 4.2. Clone the Repository

``` bash
git clone https://github.com/Andela-GenAI/genai-bootcamp
cd week_1

```

### 4.3. Virtual Environment Setup

In the workshop session, we primarily used `uv` for environment management. If you prefer `conda` or `poetry`, adjust the commands accordingly.

#### Using `uv` (Recommended)

1.  **Create a virtual environment:**
    
    ``` bash
    uv venv .venv
    
    ```

2.  **Activate the virtual environment:**
    
      * **On macOS/Linux:**
        
        ``` bash
        source .venv/bin/activate
        
        ```
    
      * **On Windows (PowerShell):**
        
        ``` bash
        .venv\Scripts\Activate.ps1
        
        ```
    
      * **On Windows (Command Prompt):**
        
        ``` bash
        .venv\Scripts\activate.bat
        
        ```

3.  **Install dependencies:**
    
    ``` bash
    uv pip install -r requirements.txt
    
    ```

#### Using `pip`

1.  **Create a virtual environment:**
    
    ``` bash
    python -m venv .venv
    
    ```

2.  **Activate the virtual environment:**
    
      * **On macOS/Linux:**
        
        ``` bash
        source .venv/bin/activate
        
        ```
    
      * **On Windows (PowerShell):**
        
        ``` bash
        .venv\Scripts\Activate.ps1
        
        ```
    
      * **On Windows (Command Prompt):**
        
        ``` bash
        .venv\Scripts\activate.bat
        
        ```

3.  **Install dependencies:**
    
    ``` bash
    pip install -r requirements.txt
    
    ```

### 4.4. API Key Configuration

1.  **Create a `.env` file** in the root directory of your project.

2.  **Add your Groq API key** to the `.env` file:
    
    ``` 
    GROQ_API_KEY="your_groq_api_key_here"
    
    ```

## 5\. Usage

### 5.1. Running the Main LLM App in Command Line (for testing)

To test the core LLM functionality without the Streamlit frontend:

``` bash
uv run main.py

```

The application will prompt you to enter questions in the console.

### 5.2. Running the Streamlit Web Application

To launch the interactive web interface:

``` bash
streamlit run app.py

```

This will open the application in your web browser. You can then:

  * **Enter your Groq API Key** in the sidebar (optional, if not set in `.env`).
  * **Select an LLM** from the dropdown menu.
  * **Adjust Temperature** and **Max Tokens** using the sliders.
  * **Customize the System Prompt** in the text area.
  * **Type your questions** in the chat input.
  * **Click "Clear Chat History"** to reset the conversation.

## 6\. Project Structure

``` 
.
├── .env                  # Environment variables (e.g., GROQ_API_KEY)
├── main.py               # Core LLM application logic
├── app_config.py         # Configuration for environment variables
├── requirements.txt      # Python dependencies
└── streamlit_app.py      # Streamlit web application

```

## 7\. Extendability

This application serves as a foundation. Here are some ideas for extending its functionality:

  * **Conversation History:** Implement a robust conversation history management system to provide context for follow-up questions.
  * **Multiple LLM Providers:** Integrate other LLM APIs (e.g., OpenAI, Gemini) and allow users to switch between them.
  * **Advanced Prompt Engineering:** Explore more sophisticated prompt engineering techniques and allow users to experiment with different prompt templates.
  * **Streaming Responses:** Implement streaming responses from the LLM for a more dynamic user experience.
  * **User Authentication:** Add user authentication to manage API key usage or personalize experiences.
  * **Enhanced Frontend:** Improve the Streamlit UI with more advanced components, custom styling and richer interactions.
  * **Error Handling:** Implement more comprehensive error handling and user feedback mechanisms.

## 8\. Mini Project (Week 1 Deliverables)

### Instructions

(a) Implement an improved version of the Q&A system with the following additional features:

  * **Chatbot name:** Give the chatbot its own identity.
  * **System prompt:** Implement a default system prompt in the script (in case the user doesn’t add theirs), and combine this with the chatbot name to give the assistant a unique persona.
  * **GPT-5 models:** Add OpenAI GPT-5, GPT-5-mini and GPT-5-nano to the model options in the frontend dropdown.
  * **OpenAI client:** Redesign the `LLMApp` class to include the OpenAI API client and automatically switch. between Groq and OpenAI depending on the selected model.

(b) Record a 1 to 2-minute video demoing your application on Streamlit, and upload the video to any platform of your choice (Google Drive, YouTube Unlisted, etc) to get the video URL.

(c) Submit the project via this [issue template](https://github.com/Andela-GenAI/genai-bootcamp/issues/new?assignees=&labels=&projects=&template=submission.yml&title=Project%3A+%3Cshort+description%3E)

**Deadline:** Sunday, Oct 26 2025 11:59 PM GMT-12


