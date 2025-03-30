from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
import os
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

# def get_embeddings():
#     embeddings = AzureOpenAIEmbeddings(
#         model="text-embedding-ada-002-2",
#         openai_api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
#         retry_max_seconds=120,
#         retry_min_seconds = 70
#     )
#     return embeddings

def get_embeddings():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"  # Adjust this URL if Ollama is running on a different address
    )
    return embeddings