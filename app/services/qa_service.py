from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from app.core.config import GEMINI_API_KEY
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-pro')

# --- Corpus Setup ---
# A single text block (10-15 lines). Adjust this text as needed.
corpus_text = """
A transformer is an essential electrical device that transfers energy between circuits using electromagnetic induction, playing a crucial role in power transmission and distribution. It consists of primary and secondary windings wrapped around a core, typically made of iron or ferrite, and operates based on Faraday’s Law of Electromagnetic Induction. When an alternating current flows through the primary coil, it generates a changing magnetic field, which induces voltage in the secondary coil. Transformers are classified into various types, including step-up transformers, which increase voltage for efficient long-distance power transmission, and step-down transformers, which reduce voltage for safe household and industrial use. Other types include distribution transformers for local power supply, power transformers for substations, and isolation transformers for safety in sensitive electronic circuits. The voltage transformation depends on the turns ratio of the coils, transformers takes 50 voltages to run
"""

# Split the text into segments (one per line).
segments = [line.strip() for line in corpus_text.strip().split('\n') if line.strip()]

# --- Embedding and FAISS Indexing ---
# Initialize the SentenceTransformer model.
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each segment.
embeddings = embedder.encode(segments, convert_to_numpy=True)
dimension = embeddings.shape[1]

# Create a FAISS index and add embeddings.
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Generate an embedding for the query and retrieve the top_k most similar segments.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_segments = [segments[idx] for idx in indices[0]]
    return "\n".join(retrieved_segments)

def answer_question(context: str, question: str) -> str:
    """
    Generate an answer using the Gemini model based solely on the retrieved context.
    """
    prompt = f"""
    Act as a Rag Assistant. Based on the following context, answer the question: "{question}".
    Context:
    {context}

    Instruction:
    - Respond in clear, error-free English.
    - Summarize your response in 1-2 lines.
    - Use only the provided context to answer the question.
    - Do not mention that you are using an external context.
    """
    response = model.generate_content(prompt)
    return response.text

def get_answer(question: str):
    """
    Retrieves context based on the question and generates an answer.
    """
    context = retrieve_context(question)
    answer = answer_question(context, question)
    return answer, context