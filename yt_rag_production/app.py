import streamlit as st
import os
from dotenv import load_dotenv
from utils import get_video_audio
from engine import transcribe_and_index, get_answer

# 1. ALWAYS FIRST: Load environment and set page config
load_dotenv()
st.set_page_config(page_title="YouTube RAG Pro", layout="centered")

# 2. Setup API Key (Pull from .env if available)
default_key = os.getenv("GROQ_API_KEY", "")

# Sidebar for configuration
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input(
    "Enter Groq API Key", 
    value=default_key, 
    type="password"
)

# Main UI
st.title("📺 YouTube Q&A Pro (Groq Powered)")
url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

# Processing Logic
if st.button("Process Video"):
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar!")
    elif not url:
        st.warning("Please paste a valid YouTube URL.")
    else:
        with st.spinner("Downloading & Transcribing..."):
            try:
                audio_path = get_video_audio(url)
                # Store the vector database in session state so it persists across refreshes
                st.session_state.vector_db = transcribe_and_index(audio_path, api_key)
                
                if os.path.exists(audio_path):
                    os.remove(audio_path) # Clean up audio file
                st.success("Video Processed Successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Q&A Logic
if "vector_db" in st.session_state:
    st.divider()
    query = st.text_input("Ask a question about the video")
    if query:
        with st.spinner("Thinking..."):
            answer = get_answer(query, st.session_state.vector_db, api_key)
            st.write("### Answer:")
            st.info(answer)