from langchain_text_splitters import CharacterTextSplitter
from embedding_model import get_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import os


def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

def create_vector_store(documents, embeddings):
    print("DOCUMENTS")
    print(documents)
    print("END")
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store


def save_vector_store(vector_store, file_path):
    vector_store.save_local(file_path)


