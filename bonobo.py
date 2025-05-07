import getpass
import os
import bs4
from urllib.parse import urlparse
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from IPython.display import Image, display
from langchain_core.vectorstores import InMemoryVectorStore
from typing import Literal
from typing_extensions import Annotated


# Load environment variables from .env file
load_dotenv()

# Retrieve API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
user_agent = os.getenv("USER_AGENT")

print(f"USER_AGENT: {user_agent}")

llm = init_chat_model("gpt-4o", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# Loading documents
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)

for path in web_paths:
    parsed = urlparse(path)
    if not parsed.scheme:
        raise ValueError(f"Invalid URL: {path}")

loader = WebBaseLoader(
    web_paths=web_paths,
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# Splitting documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

# Storing documents
document_ids = vector_store.add_documents(documents=all_splits)

# Retrieval and Generation

prompt = hub.pull("rlm/rag-prompt")
example_messages = prompt.invoke({"context": "(context goes here)", "question": "(question goes here)"}).to_messages()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

# Query Analysis

total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["category"] = "beginning"
    elif i < 2 * third:
        document.metadata["category"] = "middle"
    else:
        document.metadata["category"] = "end"

all_splits[0].metadata

vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)

class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[Literal["beginning", "middle", "end"], ..., "Section to query.",]

class State(TypedDict):
    question: str
    query: Search
    context: List
    answer: str

def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}
    
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# result = graph.invoke({"question": "What is task decomposition?"})

""" for step in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="updates"
):
    print(f"{step}\n\n----------------\n") 

for message, metadata in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="messages"
):
    print(message.content, end="|")
"""



# Output

""" 
assert len(example_messages) == 1
print(f"Total characters: {len(docs[0].page_content)}")
print(docs[0].page_content[:500]) 
print(f"Split blog post into {len(all_splits)} sub-documents.")
print(document_ids[:3])
print(example_messages[0].content)
print(f"Context: {result['context']}\n\n")
print(f"Answer: {result['answer']}")"""

display(Image(graph.get_graph().draw_mermaid_png()))

for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")