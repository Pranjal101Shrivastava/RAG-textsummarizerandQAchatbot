import streamlit as st
import os
import tempfile
from rag_logic import RAGSystem

st.set_page_config(page_title="RAG Summarizer & QA", layout="wide")

st.title("🤖 RAG-Based Text Summarizer & QA System")
st.markdown("""
Build a powerful Retrieval-Augmented Generation pipeline using Gemini & FAISS.
Upload a document, generate a summary, or ask questions!
""")

# Sidebar for API Key and Settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    model_choice = st.selectbox("LLM Model", [
        "models/gemini-flash-latest",
        "models/gemini-2.0-flash-lite",
        "models/gemini-pro-latest",
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash"
    ])
    embed_choice = st.selectbox("Embedding Model", [
        "models/gemini-embedding-001",
        "models/text-embedding-004",
        "models/embedding-001",
        "embedding-001",
        "text-embedding-004"
    ])
    st.info("Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
    
    if st.button("🔧 Diagnostics: List Models"):
        if api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            try:
                st.write("### Available Models")
                all_models = list(genai.list_models())
                
                embed_models = [m.name for m in all_models if 'embedContent' in m.supported_generation_methods]
                st.write("**Embedding Models:**")
                st.write(embed_models)
                
                gen_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
                st.write("**Generative Models:**")
                st.write(gen_models)
            except Exception as e:
                st.error(f"Error listing models: {e}")
        else:
            st.error("Please enter API Key first.")

# Initialize session state for RAG system and chunks
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file and api_key:
    if st.session_state.rag is None or st.session_state.rag.api_key != api_key:
        st.session_state.rag = RAGSystem(api_key=api_key, model_name=model_choice, embedding_model=embed_choice)

    # Process file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("Index Document"):
        with st.spinner("Processing and Indexing..."):
            chunks = st.session_state.rag.process_file(tmp_path)
            st.session_state.chunks = chunks
            latency = st.session_state.rag.build_index(chunks)
            st.success(f"Indexed {len(chunks)} chunks in {latency:.2f} seconds!")
    
    os.unlink(tmp_path)

# UI Layout: Two columns for Summary and QA
if st.session_state.chunks and st.session_state.rag.vector_store:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Summarization")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                try:
                    summary, latency = st.session_state.rag.summarize(st.session_state.chunks)
                    st.write(summary)
                    st.caption(f"Latency: {latency:.2f}s")
                except Exception as e:
                    if "429" in str(e) or "ResourceExhausted" in str(e):
                        st.error("📉 **Rate Limit Hit (429):** You've exceeded the free-tier quota. Please wait about 1 minute before trying again.")
                        st.info("Tip: Gemini Free tier has strict limits on how many requests you can make per minute.")
                    else:
                        st.error(f"Error: {e}")

    with col2:
        st.subheader("❓ Question Answering")
        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Searching..."):
                answer, sources, latency = st.session_state.rag.query(query)
                st.write("**Answer:**")
                st.write(answer)
                st.caption(f"Latency: {latency:.2f}s")
                
                with st.expander("Retrieved Context"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content[:500] + "...")
else:
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar.")
    elif not uploaded_file:
        st.info("Please upload a document to get started.")
