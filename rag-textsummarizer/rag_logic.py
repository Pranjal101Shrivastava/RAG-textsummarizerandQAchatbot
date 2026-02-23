import os
import time
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

class RAGSystem:
    def __init__(self, api_key: str, model_name: str = "models/gemini-flash-latest", embedding_model: str = "models/gemini-embedding-001"):
        self.api_key = api_key
        self.model_name = model_name
        os.environ["GOOGLE_API_KEY"] = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.vector_store = None

    def process_file(self, file_path: str) -> List[Document]:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def build_index(self, chunks: List[Document]):
        start_time = time.time()
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        end_time = time.time()
        return end_time - start_time

    def query(self, question: str):
        if not self.vector_store:
            return "No documents indexed yet.", [], 0
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        start_time = time.time()
        response = qa_chain.invoke({"query": question})
        end_time = time.time()
        
        latency = end_time - start_time
        return response["result"], response["source_documents"], latency

    def summarize(self, chunks: List[Document]):
        # Gemini has a huge context window (128k+). 
        # Using 'stuff' chain uses only ONE request, which is much better for free-tier rate limits.
        summary_chain = load_summarize_chain(self.llm, chain_type="stuff")
        
        start_time = time.time()
        summary = summary_chain.invoke(chunks)
        end_time = time.time()
        
        latency = end_time - start_time
        return summary["output_text"], latency

if __name__ == "__main__":
    # Internal test logic
    print("RAGSystem logic loaded.")
