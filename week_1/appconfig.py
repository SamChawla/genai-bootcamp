import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

class EnvConfig():

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")

# instantiate environment vars
env_config = EnvConfig()

