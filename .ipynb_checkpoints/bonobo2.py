import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv()

# Environment variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Step 1: Extract text from all PDFs in the parent directory
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Locate all PDF files in the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs"))
pdf_files = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith(".pdf")]

if not pdf_files:
    print("No PDF files found in the directory.")
    exit(1)

# Extract text from all PDFs
all_text = ""
for pdf_file in pdf_files:
    print(f"Processing: {pdf_file}")
    all_text += extract_text_from_pdf(pdf_file) + "\n"

# Step 2: Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(all_text)
print(f"Number of chunks created: {len(chunks)}")

# Step 3: Embed the text
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(chunks, embeddings)

# Step 4: Create a retriever
retriever = vectorstore.as_retriever()

# Step 5: Integrate retriever into the agent
search = TavilySearchResults(max_results=2)

# Wrap the retriever in a callable function
def retrieve_tool(query: str):
    """Tool to retrieve relevant chunks from the vector store."""
    response = retriever.get_relevant_documents(query)
    print(f"Retrieved {len(response)} documents for query: {query}")
    return response

# Add the wrapped retriever as a tool
tools = [search, retrieve_tool]  # Use the callable function instead of the raw retriever

# Bind tools to the model
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
memory = MemorySaver()

model_with_tools = model.bind_tools(tools)
agent_executor = create_react_agent(model_with_tools, tools, checkpointer=memory)

# Interactive console
print("Console interattiva per parlare con il LLM. Scrivi 'exit' per uscire.")
while True:
    user_input = input("Inserisci il tuo messaggio: ")
    if user_input.lower() == "exit":
        print("Arrivederci!")
        break

    # Process user input
    config = {"configurable": {"thread_id": "abc123"}}
    print("Elaborazione del tuo input...\n")
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=user_input)]}, config
    ):
        try:
            if (response := chunk.get("agent")) is None:
                response = chunk["messages"][0].content
                print(response)
                continue
            else:
                # Extract and print only the agent's response
                response = chunk["agent"]["messages"][0].content
                print(response)  # Print the agent's message
        except (KeyError, IndexError, TypeError):
            print("Unexpected response structure:", chunk)  # Fallback for debugging 
        print("-----------------------")

