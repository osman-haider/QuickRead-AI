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
    Act as Rag Assistant, Your task is to answer the question: {question} out of the context: {text}. Here's the instruction you need to keep in mind while generating answer.
    - Respond in clear, error free english.
    - Summarize your response into 1-2 lines, make sure you fulfill the question requirement
    - Make sure you respond only from context: {text}
    - Do not add this in your response `the provided context says this etc.`

"""
    response = model.generate_content(prompt)
    return response.text