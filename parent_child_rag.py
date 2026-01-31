import os
import gradio as gr
from dotenv import load_dotenv

# LangChain Imports
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore, LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Load Env
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env")

# config
VECTOR_DB_DIR = "vector_stores/parent_child_db"
DOC_STORE_DIR = "doc_stores/parent_child_docs"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# models
# Using Gemini 3.0 lfash
MODEL_NAME = "gemini-3.0-flash-preview" 

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# storage setup 

# 1. The Vector Store (Indices SMALL Child Chunks)
vectorstore = Chroma(
    collection_name="parent_child_split",
    embedding_function=embeddings,
    persist_directory=VECTOR_DB_DIR
)

# 2. The Document Store (Stores LARGE Parent Chunks)
# store = InMemoryByteStore() <-- for in memory storre very fast
store = LocalFileStore(DOC_STORE_DIR) # <-- for persistence

# splitters
# Parent: Big chunks (context)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Child: Small chunks (index)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# retriever
# Searches child chunks, returns parent chunks
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 3} # Retrieve top 3 Parents
)

# logic

def process_file(files):
    if not files:
        return "No files."
    
    docs = []
    for file in files:
        # Simple loading strategy
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file.name)
            docs.extend(loader.load())
        elif file.name.endswith(".txt"):
            loader = TextLoader(file.name)
            docs.extend(loader.load())
            
    if not docs:
        return "Could not load documents."
        
    # Add documents to the retriever
    # This automatically splits into Parents, then Children, indexes Children, stores Parents
    retriever.add_documents(docs, ids=None)
    
    return f"Processed {len(files)} files. Documents added to Parent-Child Index."

def query_rag(message, history):
    # 1. Retrieve
    # This returns the PARENT documents
    retrieved_docs = retriever.invoke(message)
    
    if not retrieved_docs:
        return "No relevant context found."
        
    # clarity for me 
    context_text = "\n\n".join([f"--- Context (Parent Chunk) ---\n{d.page_content}" for d in retrieved_docs])
    
    # 2. Generate
    prompt = f"""You are an expert AI tutor.
    Answer the question based strictly on the following context.
    The context consists of large book sections (Parent Chunks) retrieved by matching specific details (Child Chunks).
    
    Context:
    {context_text}
    
    Question: {message}
    """
    
    response = llm.invoke(prompt)
    
    return f"**[Retrieved {len(retrieved_docs)} Parent Chunks of ~2000 chars]**\n\n{response.content}"

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Parent-Child RAG (Small-to-Big)")
    gr.Markdown("""
    **Concept**: Index small chunks (400 chars) for high-precision search, but retrieve large parent chunks (2000 chars) for high-context generation.\n
    **Why**: Solves the "Context Fragmentation" problem where the answer key is in one sentence but the explanation is in the surrounding paragraph.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Documents", file_count="multiple")
            upload_btn = gr.Button("Index Documents")
            status = gr.Textbox(label="Status")
            
            upload_btn.click(process_file, inputs=file_input, outputs=status)
            
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(fn=query_rag)

if __name__ == "__main__":
    demo.launch()
