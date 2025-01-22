import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import os
from typing import Dict, List
import faiss
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
import threading
from queue import Queue
from datetime import datetime, timedelta
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()

class RAGSystem:
    def __init__(self):
        self.setup_credentials()
        self.initialize_components()
        self.processing_queue = Queue()
        self._start_background_processor()

    def setup_credentials(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

    def initialize_components(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.index_name = "pdf-chat"
        self.llm = ChatOpenAI(temperature=0, openai_api_key=self.openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Initialize Pinecone vector store
        self.vectorstore = LangchainPinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace="pdf_documents"
        )

    def _start_background_processor(self):
        def process_queue():
            while True:
                chat_id, pdf_path = self.processing_queue.get()
                self._process_pdf(chat_id, pdf_path)
                self.processing_queue.task_done()
        
        thread = threading.Thread(target=process_queue, daemon=True)
        thread.start()

    def process_upload(self, uploaded_file, chat_id):
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        pdf_path = f"temp/{chat_id}_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        preview = convert_from_path(pdf_path, dpi=72, first_page=1, last_page=3)
        self.processing_queue.put((chat_id, pdf_path))
        return preview

    def _process_pdf(self, chat_id, pdf_path):
        try:
            # Load and split the PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            chunks = self.text_splitter.split_documents(pages)
            
            # Add document metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chat_id": chat_id,
                    "chunk_id": i,
                    "source": pdf_path
                })
            
            # Add documents to Pinecone
            self.vectorstore.add_documents(chunks)
            
            # Clean up
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

    def query(self, chat_id, query, chat_history):
        try:
            # Create a retriever with filtering for the specific chat_id
            search_kwargs = {"filter": {"chat_id": chat_id}}
            retriever = self.vectorstore.as_retriever(
                search_kwargs=search_kwargs,
                search_type="similarity",
                k=3
            )
            
            # Create QA chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            
            # Get response
            result = qa_chain({
                "question": query, 
                "chat_history": chat_history
            })
            
            return result["answer"]
            
        except Exception as e:
            print(f"Error during query: {e}")
            return f"An error occurred: {str(e)}"

def setup_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = "pdf-chat"
    
    # Check if index already exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    print("PineCone Setup completed...")
    return index_name

def main():
    st.title("PDF Chat Assistant")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = str(datetime.now().timestamp())

    # File upload section
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        try:
            preview = st.session_state.rag_system.process_upload(
                uploaded_file,
                st.session_state.chat_id
            )
            st.image(preview[0], caption="PDF Preview", use_column_width=True)
            st.success("PDF uploaded and processing started!")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

    # Query section
    query = st.text_input("Ask a question about your PDF:")
    if query:
        try:
            with st.spinner("Generating response..."):
                response = st.session_state.rag_system.query(
                    st.session_state.chat_id,
                    query,
                    st.session_state.chat_history
                )
                st.session_state.chat_history.append((query, response))
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

    # Chat history section
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for q, a in st.session_state.chat_history:
            st.write("Question:", q)
            st.write("Answer:", a)
            st.markdown("---")

if __name__ == "__main__":
    setup_pinecone()
    main()