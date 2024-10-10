import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
import pickle
import time
import faiss  # Direct FAISS usage
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import cohere  # Cohere API

# Cohere API key
COHERE_API_KEY = "7yIZExqyEWazvYQEKgwxIJsvZYXKzd9A7ZnZ28pK"  # Replace with your actual Cohere API key
co = cohere.Client(COHERE_API_KEY)  # Initialize Cohere client

# Streamlit app interface
st.title("Equity Research Tool 📈")
st.sidebar.title("News Article URLs")

# Load the SentenceTransformer model globally
model = SentenceTransformer('all-MiniLM-L6-v2')  # Hugging Face transformer model for embeddings

# Get URLs from user input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"  # File to save the FAISS index

main_placeholder = st.empty()

# Process URLs when the button is clicked
if process_url_clicked:
    try:
        valid_urls = [url for url in urls if url]  # Remove empty URLs
        if not valid_urls:
            st.error("No valid URLs provided.")
            raise ValueError("No valid URLs provided.")

        # Load data manually using requests and BeautifulSoup
        data = []
        for url in valid_urls:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text()
                doc = Document(page_content=text, metadata={"source": url})
                data.append(doc)
            else:
                st.warning(f"Failed to retrieve URL: {url}")

        if not data:
            st.error("No data loaded from the URLs.")
            raise ValueError("No data loaded from the URLs.")

        # Split the text data into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        if not docs:
            st.error("No documents generated after splitting the text.")
            raise ValueError("No documents generated after splitting the text.")

        # Create embeddings using Hugging Face model (Sentence-Transformers)
        embeddings = model.encode([doc.page_content for doc in docs])  # Generate embeddings

        # Initialize FAISS index
        dimension = embeddings.shape[1]  # Embedding dimension from model
        index = faiss.IndexFlatL2(dimension)  # L2 distance index
        index.add(embeddings)  # Add embeddings to FAISS index

        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index and docs to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump({"index": index, "docs": docs}, f)
        main_placeholder.text("FAISS Index saved successfully!")

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")

# Function to use Cohere for question-answering
def cohere_answer_question(question, context):
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=f"Question: {question}\n\nContext: {context}\n\nAnswer:",
        max_tokens=100
    )
    return response.generations[0].text.strip()

# Process user queries
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            saved_data = pickle.load(f)
            index = saved_data["index"]
            docs = saved_data["docs"]

            # Convert query into embedding using the same SentenceTransformer model
            query_embedding = model.encode([query])

            # Search FAISS index for the closest documents
            distances, indices = index.search(query_embedding, k=3)  # Retrieve top 3 results

            # Get the relevant documents based on the search
            relevant_docs = [docs[i] for i in indices[0]]

            if relevant_docs:
                context = ' '.join([doc.page_content for doc in relevant_docs])
            else:
                st.error("No relevant documents found.")
                raise ValueError("No relevant documents found.")

            # Use Cohere to answer the query
            result = cohere_answer_question(query, context)

            # Display the result
            st.header("Answer")
            st.write(result)

            # Display sources
            st.subheader("Sources:")
            for doc in relevant_docs:
                st.write(doc.metadata["source"])