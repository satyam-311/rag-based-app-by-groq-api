import os
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Initialize Local Embeddings (CPU-friendly)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def transcribe_and_index(audio_path, api_key):
    client = Groq(api_key=api_key)
    
    # 1. Groq Cloud Transcription (Fast)
    with open(audio_path, "rb") as file:
        transcript = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3"
        )
    
    # 2. Local Indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(transcript.text)
    vector_db = FAISS.from_texts(chunks, embeddings)
    vector_db.save_local("vector_store")
    return vector_db

def get_answer(query, vector_db, api_key):
    client = Groq(api_key=api_key)
    docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"Answer based only on this context: {context}"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content