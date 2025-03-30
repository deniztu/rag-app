import openai
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def initialize_language_model():
     return AzureChatOpenAI(
         temperature=0,
         openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
         azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
         api_version=os.getenv("AZURE_OPENAI_VERSION")
     )