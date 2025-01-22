












# main.py
import streamlit as st
from typing import Set
import spacy
import os
import time
from utils.pdf_processor import EnhancedPDFProcessor
from utils.vector_store import OptimizedVectorStore
from utils.qa_chain import QAChain
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processing system."""
        self.pdf_processor = EnhancedPDFProcessor()
        self.vector_store = OptimizedVectorStore()
        self.qa_chain = QAChain()
        
    def process_document(self, file, progress_bar=None):
        """Process a single document with progress tracking."""
        try:
            # Extract text with metadata
            pdf_content = self.pdf_processor.extract_text_from_pdf(file)
            if progress_bar:
                progress_bar.progress(0.3)
            
            # Split into chunks
            chunks = self.pdf_processor.split_text_into_chunks(
                pdf_content["text"],
                chunk_size=2000,
                overlap=100,
                respect_sentences=True
            )
            if progress_bar:
                progress_bar.progress(0.6)
            
            # Store in vector database
            self.vector_store.store_embeddings(chunks)
            if progress_bar:
                progress_bar.progress(1.0)
            print(chunks)
            print("_-----------------------")
            print(len(chunks))
            return len(chunks), pdf_content["metadata"]["page_count"]
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return 0, 0

def initialize_session_state():
    """Initialize session state variables."""
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {}

def display_file_stats():
    """Display statistics about processed files."""
    if st.session_state.processing_stats:
        st.subheader("📊 Processing Statistics")
        stats_cols = st.columns(3)
        
        total_files = len(st.session_state.processed_files)
        total_pages = sum(stat['pages'] for stat in st.session_state.processing_stats.values())
        total_chunks = sum(stat['chunks'] for stat in st.session_state.processing_stats.values())
        
        stats_cols[0].metric("Total Files", total_files)
        stats_cols[1].metric("Total Pages", total_pages)
        stats_cols[2].metric("Total Chunks", total_chunks)

def main():
    # Page configuration
    st.set_page_config(
        page_title="bot360",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        .metrics-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .stProgress > div > div > div {
            background-color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 Bot 360")

    
    initialize_session_state()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    st.info(f"Processing: {uploaded_file.name}")
                    progress_bar = st.progress(0)
                    
                    # Process document and track statistics
                    chunks, pages = st.session_state.processor.process_document(
                        uploaded_file,
                        progress_bar
                    )
                    
                    if chunks > 0:
                        st.session_state.processed_files.add(uploaded_file.name)
                        st.session_state.processing_stats[uploaded_file.name] = {
                            'chunks': chunks,
                            'pages': pages,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                    
                    # Clean up progress bar
                    time.sleep(0.5)
                    progress_bar.empty()
        
        # Display processing statistics
        display_file_stats()
        
        # Display processed files with details
        if st.session_state.processed_files:
            st.subheader("📋 Processed Documents")
            for file in st.session_state.processed_files:
                stats = st.session_state.processing_stats[file]
                st.markdown(f"""
                    - **{file}**
                      - Pages: {stats['pages']}
                      - Chunks: {stats['chunks']}
                      - Processed: {stats['timestamp']}
                """)
    
    with col2:
        st.header("Ask Questions")
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about the documents?"
        )
        
        # Search settings
        with st.expander("🔍 Search Settings"):
            top_k = 3 #3
            score_threshold = 0.3
        
        if question:
            with st.spinner('🔍 Searching for answer...'):
                # Perform hybrid search with custom settings
                search_results = st.session_state.processor.vector_store.hybrid_search(
                    question,
                    top_k=top_k,
                    score_threshold=score_threshold
                )
                
            # if search_results:
                # Generate answer
                answer = st.session_state.processor.qa_chain.generate_answer(
                    question,
                    search_results
                )
                
                # Display answer
                st.markdown("### 📝 Answer")
                st.markdown(answer)
                
                # Display sources
                with st.expander("📚 View Sources"):
                    for i, context in enumerate(search_results, 1):
                        st.markdown(f"""
                            **Source {i}** (Relevance: {context['score']:.2f})
                            ```
                            {context['text']}
                            ```
                            ---
                        """)
            # else:
            #     st.warning("No relevant information found in the documents. Try adjusting the search settings or reformulating your question.")

if __name__ == "__main__":
    main()

