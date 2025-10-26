# BotMan : A Simple LLM Q&A Application

## 1\. Overview

A minimal Q&A chat application with a Streamlit frontend. Supports Groq and OpenAI (GPT‑5 family) models, automatic provider switching, a named chatbot persona, and robust .env loading for Streamlit.

## 2\. Features

  * **Groq and OpenAI API Integration:** Connects to the Groq API to access a range of open-source LLMs.
  * **Configurable Models:** Users can select different LLMs (e.g., Llama 3.1 8B, Llama 3.3 70B, OpenAI OSS 20B) from the Groq API Client.
  * **API Key Input:** Option to input a Groq API key directly in the Streamlit frontend.
  * **Adjustable Temperature:** Controls the randomness of model responses using a temperature slider.
  * **Output Control:** Sets the maximum length of the model's generated responses.
  * **Prompt Customization:** Defines a system prompt to guide the LLM's behavior and context.
  * **Chat History Clearing:** A button to clear the current chat history in the Streamlit application.

## 3\. Tools & Frameworks Used

  * **Python:** The primary programming language.
  * **Groq and OpenAI API:** For accessing LLMs.
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


