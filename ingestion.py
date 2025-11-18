# Import necessary libraries
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import json
import requests
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4

load_dotenv()

pdf_list = []
index_path = "faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
os.makedirs("pdfs", exist_ok=True)

with open('arxivData.json') as f:
    arxiv_data = json.load(f)

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
        return True
    except Exception as e:
        print(f"❌ Failed {arxiv_id}: {e}")
        return False
    
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(f"pdfs/{pdf_path}.pdf")
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def prepare_document(doc_metadata):
    if download_pdf(doc_metadata['id']):
        arxiv_id = doc_metadata['id']
        raw_text = get_pdf_text(arxiv_id)
        doc = Document(page_content=raw_text,
            metadata = {'author':[authr['name'] for authr in eval(doc_metadata['author'])],
                'id':doc_metadata['id'],
                'title':doc_metadata['title'],
                'day':doc_metadata['day'],
                'month':doc_metadata['month'],
                'year':doc_metadata['year']})
        return doc
    else:
        return None

# variables for tracking research papers to be downloaded
start_index = 10
end_index = 20

documents_list = []
for data in arxiv_data[start_index:end_index]:
    doc = prepare_document(data)
    if doc:
        documents_list.append(doc)
    else:
        continue

document_splits_list = splitter.split_documents(documents_list)

if len(document_splits_list):
    if not os.path.exists(index_path):
        print("Creating index...")
        vector_store = FAISS.from_documents(document_splits_list, embedding=embeddings)
        vector_store.save_local(index_path)
        print("Index created.")
    else:
        print("Updating index...")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        uuids = [str(uuid4()) for _ in range(len(document_splits_list))]
        vector_store.add_documents(documents=document_splits_list, ids=uuids)
        vector_store.save_local(index_path)
        print("Index updated.")
else:
    print('Failed to process docuemts for uploading.')