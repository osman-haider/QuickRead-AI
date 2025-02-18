from transformers import pipeline
from app.core.config import GEMINI_API_KEY
from huggingface_hub import login
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)


# Initialize the Gemini model
model = genai.GenerativeModel('gemini-pro')

def summarize_text(text: str) -> str:
    # Generate a summary using Gemini
    prompt = f"Summarize the following text in a concise manner:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text

def answer_question(text: str, question: str) -> str:
    # Generate an answer using Gemini
    prompt = f"""
    Answer the following Question from the given context. Don't write any extra information make the Answer readable and clear accourding to the Question. Do not add unnecessary details or assumptions in Answer by your own.

    Context: {text}

    Question: {question}

    Answer:
    """
    response = model.generate_content(prompt)
    print(f"response.text: {response.text}")
    return response.text