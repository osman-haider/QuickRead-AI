import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Fetch Hugging Face token from environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

if HUGGINGFACE_TOKEN is None:
    raise ValueError("HUGGINGFACE_TOKEN is not set in the .env file")
