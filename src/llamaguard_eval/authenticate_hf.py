from huggingface_hub import login
from dotenv import load_dotenv
import os

def authenticate():
    """
    Authenticate with Hugging Face using token from .env or environment variable HF_TOKEN.
    """
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    login(token=token)
    print("Successfully authenticated with Hugging Face")
