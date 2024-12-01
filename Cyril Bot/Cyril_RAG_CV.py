## RAG SYSTEM USING LLamaIndex and Open AI Embedding for making CV BOT

import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.response.pprint_utils import pprint_response

# Set dark theme and page configuration
st.set_page_config(page_title="Cyril's CV Bot", page_icon="📄", layout="wide")

# Custom CSS for styling the input box
st.markdown(
    """
    <style>
        .css-18e3th9 {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        .stTextInput input {
            background-color: #333333;
            color: #e0e0e0;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #555555;
        }
        .stTextInput input::placeholder {
            color: #e0e0e0;  /* Placeholder text color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.markdown("<h1 style='text-align: center; color: #000000;'>🈺 Ask Cyril AI </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #e0e0e0;'>Ask anything about Cyril's experience, skills, or personal details!</p>", unsafe_allow_html=True)


# Load environment variables from .env file
load_dotenv()

# Load and index documents
documents1 = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents1, show_progress=True)

# Configure the query engine with retriever and postprocessor
retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)
query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])

# User input for the query with placeholder text
user_query = st.text_input("🔍 Type your question here...")

# Display the response from the query engine
if user_query:
    st.markdown("### ")
    
    # Run the query and get the response
    response = query_engine.query(user_query)
    
    # Display the response with source details
    with st.expander("📜 Detailed Response with Sources"):
        pprint_response(response, show_source=True)
        
    # Display the main response
    st.markdown("**Response:**")
    st.success(response.response)

# Footer
st.markdown("---")
# st.markdown("<p style='text-align: center; color: #e0e0e0;'>Made with ❤️ by Cyril's Bot - Powered by LLamaIndex & OpenAI</p>", unsafe_allow_html=True)
