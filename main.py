import os
import streamlit as st
from dotenv import load_dotenv
from typing import Tuple, List, Dict
import PyPDF2
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time
from tqdm import tqdm
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Constants for optimization
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
BATCH_SIZE = 50
MAX_WORKERS = 4

# CHUNK_SIZE = 20000       # Larger chunks for faster storage
# CHUNK_OVERLAP =2000     # Minimal overlap to reduce redundancy
# BATCH_SIZE = 2500        # Larger batch size to speed up embedding generation
# MAX_WORKERS = 16        # Fully utilize system resources



# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'processing_progress' not in st.session_state:
    st.session_state.processing_progress = 0

class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

class ProcessingManager:
    def __init__(self):
        self.progress = 0
        self.total = 0
        self.lock = threading.Lock()
        
    def update_progress(self, increment=1):
        with self.lock:
            self.progress += increment
            if st.session_state is not None:
                st.session_state.processing_progress = (self.progress / self.total) * 100

def initialize_components():
    """Initialize all necessary components for the RAG system"""
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    graph = Neo4jGraph(
        url=neo4j_url,
        username=neo4j_username,
        password=neo4j_password,
        database="neo4j"
    )
    
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        max_tokens=2000
    )
    
    embeddings = OpenAIEmbeddings(
        chunk_size=BATCH_SIZE
    )
    
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    
    return graph, llm, vector_index

def extract_text_from_pdf(pdf_path: str, filename: str) -> List[Document]:
    """Extract text from PDF and return as Document objects with metadata"""
    documents = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                # Create Document object with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,
                        "file_type": "pdf"
                    }
                )
                documents.append(doc)
    return documents

def process_chunk_batch(chunks: List[Document], graph: Neo4jGraph, vector_index: Neo4jVector, 
                       processing_manager: ProcessingManager):
    """Process a batch of chunks in parallel"""
    try:
        # Convert chunks to graph documents
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
        llm_transformer = LLMGraphTransformer(llm=llm)
        graph_documents = llm_transformer.convert_to_graph_documents(chunks)
        
        # Add to Neo4j graph
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        
        # Add to vector store
        vector_index.add_documents(chunks)
        
        # Update progress
        processing_manager.update_progress(len(chunks))
        
    except Exception as e:
        st.error(f"Error processing batch: {str(e)}")
        raise e

def process_pdf_file(uploaded_file, graph: Neo4jGraph, vector_index: Neo4jVector):
    """Process uploaded PDF file with parallel processing and progress tracking"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Extract text from PDF with metadata
        documents = extract_text_from_pdf(tmp_file_path, uploaded_file.name)
        
        # Create text splitter with optimized parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )

        # Split documents into chunks while preserving metadata
        chunks = text_splitter.split_documents(documents)

        # Update chunk IDs while preserving other metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        # Initialize processing manager
        processing_manager = ProcessingManager()
        processing_manager.total = len(chunks)

        # Process chunks in batches using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                future = executor.submit(
                    process_chunk_batch, 
                    batch, 
                    graph, 
                    vector_index,
                    processing_manager
                )
                futures.append(future)

            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()

        # Cleanup
        os.unlink(tmp_file_path)
        return len(chunks)

    except Exception as e:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise e

def retriever(question: str, graph: Neo4jGraph, vector_index: Neo4jVector, llm: ChatOpenAI):
    """Combined retriever for both structured and unstructured data"""
    entity_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are extracting organization and person entities from the text"),
        ("human", "Use the given format to extract information from the following input: {question}")
    ])
    entity_chain = entity_prompt | llm.with_structured_output(Entities)
    
    result = ""
    entities = entity_chain.invoke({"question": question})
    
    for entity in entities.names:
        response = graph.query(
            """
            MATCH (n)
            WHERE (n:Person OR n:Organization OR n:Document) 
            AND (n.name CONTAINS $entity OR n.text CONTAINS $entity OR n.id CONTAINS $entity)
            WITH n LIMIT 2
            CALL {
              WITH n
              MATCH (n)-[r]->(neighbor)
              RETURN n.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH n
              MATCH (n)<-[r]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + n.id AS output
            }
            RETURN DISTINCT output
            LIMIT 50
            """,
            {"entity": entity},
        )
        result += "\n".join([el['output'] for el in response])
    
    # Get relevant documents from vector store
    similar_docs = vector_index.similarity_search(question)
    unstructured_data = [doc.page_content for doc in similar_docs]
    
    final_data = f"""Structured data:
{result}
Unstructured data:
{"#Document ".join(unstructured_data)}
    """
    return final_data

def setup_qa_chain(llm: ChatOpenAI, graph: Neo4jGraph, vector_index: Neo4jVector):
    """Set up the question-answering chain"""
    answer_prompt = ChatPromptTemplate.from_template(
        """You are a professional assistant providing accurate, concise, and contextually relevant answers. 
        Respond to the user's question based on the given context. If the answer is not available in the provided context, 
        avoid wrong answers. 

        if the context is present give the good answer with proper headings or
        suppose the context not presenet
        Instead of wrong answer follow the process:
        1. Intimate the context not present, Give some relevent data with proper heading and Suggest 2-3 alternative or related questions that the user could ask for better assistance only if the context not available.
        2. Always maintain a professional and helpful tone.

        {context}
        Question: {question}
        Answer:"""
    )

    qa_chain = (
        RunnableParallel({
            "context": lambda x: retriever(x["question"], graph, vector_index, llm),
            "question": RunnablePassthrough(),
        })
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

def main():
    st.set_page_config(page_title="bot360", layout="wide")
    
    st.title("Bot360")
    
    try:
        # Initialize components
        graph, llm, vector_index = initialize_components()
        qa_chain = setup_qa_chain(llm, graph, vector_index)
        
        # Sidebar for document loading
        with st.sidebar:
            st.header("Upload PDF Documents")
            uploaded_files = st.file_uploader(
                "Choose PDF files", 
                type=['pdf'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Check if file has already been processed
                    if uploaded_file.name not in st.session_state.processed_files:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            try:
                                num_chunks = process_pdf_file(uploaded_file, graph, vector_index)
                                st.success(f"Successfully processed {uploaded_file.name} into {num_chunks} chunks")
                                st.session_state.processed_files.add(uploaded_file.name)
                            except Exception as e:
                                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Display processed files
            if st.session_state.processed_files:
                st.header("Processed Files")
                for file in st.session_state.processed_files:
                    st.text(f"✓ {file}")
        
        # Main chat interface
        st.header("Ask Questions")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about the uploaded documents"):
            # Display user question
            with st.chat_message("user"):
                st.write(question)
            
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = qa_chain.invoke({"question": question})
                        st.write(response)
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.error("Please make sure your environment variables are set correctly and Neo4j is running.")

if __name__ == "__main__":
    main()