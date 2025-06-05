from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

def generate_text(prompt: str, model: str = "gemini-1.5-flash-8b") -> str:
    """
    Generate text using Google GenAI.

    Args:
        prompt (str): The input prompt for text generation.
        model (str): The model to use for generation. Default is "gemini-1.5-pro".

    Returns:
        str: The generated text.
    """
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text