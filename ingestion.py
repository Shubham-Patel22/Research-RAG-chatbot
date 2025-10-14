# Import necessary libraries
from PyPDF2 import PdfReader  # For reading PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable chunks
import os  # For interacting with the operating system
from dotenv import load_dotenv
import json

from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For generating embeddings using Google Generative AI
import google.generativeai as genai  # For using Google's generative AI capabilities
from langchain.vectorstores import FAISS  # For efficient similarity search with embeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # For chat-based interactions with Google AI
from langchain.chains.question_answering import load_qa_chain  # For loading QA chains
from langchain.prompts import PromptTemplate  # For creating templates for prompts
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

#import gdown

import kagglehub
import requests

load_dotenv()

pdf_list = []

def download_pdf(arxiv_id, save_dir="pdfs"):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    save_path = os.path.join(save_dir, f"{arxiv_id}.pdf")
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Downloaded: {arxiv_id}")
        pdf_list.append(arxiv_id)
    except Exception as e:
        print(f"❌ Failed {arxiv_id}: {e}")
    
def get_pdf_text(pdf_path):
    """
    Extracts text from a list of PDF files.

    Args:
        pdf_path (list): A list of directory paths to PDF files.

    Returns:
        str: A concatenated string containing the text extracted from all the specified PDF files.

    This function iterates over each PDF file provided in the 'pdf_paths' list,
    reads the content of each page, and appends the extracted text to a single string.
    """
    text = ""

    pdf_reader = PdfReader(f"pdfs/{pdf_path}.pdf") # Create a PdfReader object for the current PDF file
    for page in pdf_reader.pages: # Iterate through each page in the PDF
        text += page.extract_text() # Extract text from the page and append it to the 'text' variable
    return text # Return the concatenated text from all PDFs

def get_text_chunks(text):
    """
    Splits the input text into manageable chunks.

    Args:
        text (str): The input text to be split into chunks.

    Returns:
        list: A list of text chunks, each with a maximum size defined by 'chunk_size',
              and overlapping content defined by 'chunk_overlap'.

    This function utilizes the RecursiveCharacterTextSplitter to divide the provided text
    into smaller segments. Each chunk can be up to 'chunk_size' characters long,
    with a specified overlap of 'chunk_overlap' characters between consecutive chunks.
    This is useful for processing large texts without losing context.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    # Initialize the RecursiveCharacterTextSplitter with a chunk size of 10,000 characters
    # overlap of 1,000 characters between consecutive chunks to maintain context.
    chunks = text_splitter.split_text(text) # Split the text into chunks
    return chunks # Return the list of text chunks

    # I love apple -> s1
    # Wow, Even i love apple -> s2

# Document Ingestion and Embedding
index_path = "faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    

def create_vector_store(text_chunks):
    
    # Create new embeddings for the incoming text
    new_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    if not os.path.exists(index_path):
        new_store.save_local(index_path)
        print("✅ New FAISS index created.")
    else:
        print("The FAISS index already exists")
    return new_store

def update_vector_store(text_chunks):
    
    # If index exists, load and merge
    if os.path.exists(index_path):
        print("🔄 Existing FAISS index found. Updating...")
        # Create new embeddings for the incoming text
        new_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        existing_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        existing_store.merge_from(new_store)
        existing_store.save_local(index_path)
        print("✅ FAISS index updated.")
        return existing_store
    else:
        print("No FAISS index found, creating a new one.")
    
# Create folder for PDFs
os.makedirs("pdfs", exist_ok=True)

with open('arxivData.json') as f:
    # Load the JSON data from the file
    arxiv_data = json.load(f)

id_list = [data['id'] for data in arxiv_data]
#print(id_list)

text_chunks_list = []

# variable for tracking no of research papers to be downloaded
n = 10
for arxiv_id in id_list[0:n]:
    download_pdf(arxiv_id)
    raw_text = get_pdf_text(arxiv_id) # Extract text from the PDF files
    text_chunks = get_text_chunks(raw_text)  # Split the extracted text into chunks
    text_chunks_list.extend(text_chunks)

if not os.path.exists(index_path):
    print("Processing PDFs and creating index...")
    create_vector_store(text_chunks_list) # Create a vector store from the text chunks
    print("Index created.")
else:
    print("Processing PDFs and updating index...")
    create_vector_store(text_chunks_list)
    print("Index updated.")

