import os
import json
import streamlit as st
from dotenv import load_dotenv
from typing import List, Optional, Literal
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel
from streamlit_pdf_viewer import pdf_viewer

from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.pydantic_v1 import Field
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain import hub

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langgraph.graph import START, END, StateGraph

from llm import initialize_language_model
from db_handler import process_pdfs_to_db
from pdf_handling import (
    load_documents_from_directory,
    create_vector_store,
    save_vector_store,
)
from embedding_model import get_embeddings




# Load environment variables from a .env file
load_dotenv()

st.set_page_config(page_title="RAG", layout="wide")


# st.markdown(
#     "<h1 style='text-align: left;'>Chat to find your lightbulb</h1>", unsafe_allow_html=True
# )


@st.cache_resource
def load_or_create_db(data_path, db_path):
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' does not exist.")
        print(f"Creating database ...")
        process_pdfs_to_db(data_path, db_path)
        print(f"Database created.")
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    else:
        print(f"Database file '{db_path}' exists.")
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        print(f"Loading database ...")
    return db

data_path = "../data"
db_path = "lighting.db"
db = load_or_create_db(data_path, db_path)


# Path Index 
VECTOR_STORE_PATH = "./vector_store.faiss"

# Check if the file exists
if os.path.exists(VECTOR_STORE_PATH):
    print(f"The file '{VECTOR_STORE_PATH}' exists. Loading the vector store...")

    # Load the vector store from the file
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, get_embeddings(), allow_dangerous_deserialization=True)
    
else:
    print(f"The file '{VECTOR_STORE_PATH}' does not exist. Starting PDF ingestion...")

    documents = load_documents_from_directory(data_path)
    vector_store = create_vector_store(documents, get_embeddings())
    save_vector_store(vector_store, VECTOR_STORE_PATH)

# Setup Retriever
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

# Load the vector store from the file
vector_store = FAISS.load_local(VECTOR_STORE_PATH, get_embeddings(), allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

# Reranking
compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever)



# Initialize the LLM
llm = initialize_language_model()

#----------------- Build the graph -----------------

class State(TypedDict):
    question: str
    query: str
    context: str
    answer: str
    sources: List[str]


# edge
def route(State):

    # Data model
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""

        datasource: Literal["sql_db", "vector_db"] = Field(
            ...,
            description="Given a user question choose which datasource would be most relevant for answering their question",
                )

    # LLM with function call 
    structured_llm = llm.with_structured_output(RouteQuery)

    # Prompt 
    system = """You are an expert at routing a user question to the appropriate data source.

    Based on the user question, route it to the relevant data source."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # Define router 
    router = prompt | structured_llm

    result = router.invoke({"question": State['question']})

    return result.datasource.lower()

### SQL DB CHAIN
class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    """Generate SQL query to fetch information."""

    print("---WRITE QUERY---")

    examples = [

    {
        "input": "Welche Leuchten sind gut für die Ausstattung im Operationssaal geeignet?",
        "query": "SELECT a.application_area, p.product_name FROM application_areas a JOIN surgical_lighting p ON a.product_id = p.id WHERE a.application_area LIKE '%Surgical%' ORDER BY p.product_name LIMIT 10;",
    },
    {
        "input": "Welche Leuchten sind gut für die Ausstattung im OP-Bereich geeignet?",
        "query": "SELECT a.application_area, p.product_name FROM application_areas a JOIN surgical_lighting p ON a.product_id = p.id WHERE a.application_area LIKE '%Surgical%' ORDER BY p.product_name LIMIT 10;",
    },
    {
        "input": "Welche Leuchten sind gut für die Ausstattung im Chirurgiebereich geeignet?",
        "query": "SELECT a.application_area, p.product_name FROM application_areas a JOIN surgical_lighting p ON a.product_id = p.id WHERE a.application_area LIKE '%Surgical%' ORDER BY p.product_name LIMIT 10;",
    },
]


    example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. ALWAYS QUERY file_path IN YOUR SQL QUERIES. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
        suffix="User input: {input}\nSQL query: ",
        input_variables=["input", "top_k", "table_info"],
    )

    prompt = prompt_template.format(input=state["question"], top_k=20, table_info=db.get_table_info())
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""

    print("---EXECUTE QUERY---")

    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"context": execute_query_tool.invoke(state["query"])}

class FilePath(BaseModel):
    file_paths: List[str]

def extract_file_paths_from_sql_db_context(state: State):
    """Extract filepaths from SQL result."""
    prompt = (
        "Given the following SQL query, "
        "and SQL result, extract the file paths.\n\n"
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["context"]}'
    )

    print("---EXTRACT FILE PATHS FROM SQL---")

    structured_llm = llm.with_structured_output(FilePath)
    result = structured_llm.invoke(prompt)
    return {"sources": result.file_paths}

def generate_answer_from_sql_db_context(state: State):
    """Answer question using retrieved information as context."""

    print("---GENERATE ANSWER FROM SQL---")

    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["context"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

### Vector DB CHAIN

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve(state: State):

    print("---RETRIEVE FROM VECTOR DB---")
    #print(state['question'])
    
    result = compression_retriever.get_relevant_documents(state['question'])

    docs = format_docs(result)

    sources = [res.metadata['source'] for res in result]

    return {"context": docs, "sources": sources}

def generate_answer_from_vector_db_context(state: State):

    print("---GENERATE ANSWER FROM VECTOR DB---")

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Chain
    vector_db_chain = prompt | llm | StrOutputParser()

    response = vector_db_chain.invoke({"context": state['context'], "question": state['question']})
    
    return {"answer": response}

def build_langgraph(state: State):

    workflow = StateGraph(State)

    # Define the nodes
    workflow.add_node("route", route)  # route

    workflow.add_node("write_query", write_query)  # write query
    workflow.add_node("execute_query", execute_query)  # execute query
    workflow.add_node("extract_file_paths_from_sql_db_context", extract_file_paths_from_sql_db_context)  # extract file paths
    workflow.add_node("generate_answer_from_sql_db_context", generate_answer_from_sql_db_context)  # generate answer

    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("generate_answer_from_vector_db_context", generate_answer_from_vector_db_context)  # generate answer

    # Build graph
    #workflow.add_edge(START, "route")
    workflow.add_conditional_edges(
        START,
        route,
        {
            "sql_db": "write_query",
            "vector_db": "retrieve",
        },
    )
    workflow.add_edge("write_query", "execute_query")
    workflow.add_edge("execute_query", "generate_answer_from_sql_db_context")
    workflow.add_edge("generate_answer_from_sql_db_context", "extract_file_paths_from_sql_db_context")
    workflow.add_edge("extract_file_paths_from_sql_db_context", END)

    workflow.add_edge("retrieve", "generate_answer_from_vector_db_context")
    workflow.add_edge("generate_answer_from_vector_db_context", END)

    # Compile 
    app = workflow.compile()

    return app

graph = build_langgraph(State)

#----------------- Streamlit UI -----------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hallo, frage mich etwas über unsere Glühbirnen."}
    ]

if "current_page" not in st.session_state:
    st.session_state.current_page = 1


if "chat_occurred" not in st.session_state:
    default_pdf_path = "../data/ZMP_55977.pdf"
    st.session_state.doc = PyPDFLoader(default_pdf_path).load()
    st.session_state.total_pages = len(st.session_state.doc)

# Display the PDF viewer on the right side by default
#col1, col2 = st.columns([1, 2])

with st.sidebar:

    history = st.container(height=600)

    #st.markdown("### Chat Interface")
    if "chat_history" in st.session_state:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                history.chat_message(msg["role"], avatar="❓").write(msg["content"])  # User avatar
            elif msg["role"] == "assistant":
                history.chat_message(msg["role"], avatar="💡").write(msg["content"])  # Assistant avatar


    # Placeholder for chat input to keep it at the bottom
    chat_input_placeholder = st.empty()

    if user_input := chat_input_placeholder.chat_input("Frage stellen"):
        
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        history.chat_message("user", avatar="❓").write(user_input)

        with st.spinner("Generiere Antwort..."):
            try:
                result = graph.invoke({"question": user_input})

                answer = result['answer']

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
                history.chat_message("assistant", avatar="💡").write(answer)

                st.session_state.chat_occurred = True
                st.session_state.sources = result['sources'][0]

                if "chat_occurred" in st.session_state:
                    default_pdf_path = st.session_state.sources
                    print("default_pdf_path")
                    print(default_pdf_path)
                    st.session_state.doc = PyPDFLoader(default_pdf_path).load()
                    st.session_state.total_pages = len(st.session_state.doc)

            except json.JSONDecodeError:
                st.error(
                    "There was an error parsing the response. Please try again."
            )


with st.container(height = 800):

    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    with nav_col1:
        if st.button("Vorherige Seite", type ="primary"   ) and st.session_state.current_page > 1:
            st.session_state.current_page -= 1
    with nav_col3:
        if (
            st.button("Nächste Seite", type ="primary")
            and st.session_state.current_page
            < st.session_state.total_pages
        ):
            st.session_state.current_page += 1
    with nav_col2:
        st.write(
            f"Seite {st.session_state.current_page} von {st.session_state.total_pages}"
        )


    # Center-align the PDF viewer
    pdf_col1, pdf_col2, pdf_col3 = st.columns([1, 3, 1])

    with pdf_col2:
            
            with st.container(border= True):
                pdf_viewer(
                    default_pdf_path,
                    width=600,
                    height=800,
                    pages_to_render=[st.session_state.current_page],
                ) 