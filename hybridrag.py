import os
import shutil
import gradio as gr
from dotenv import load_dotenv

# LangChain Imports
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

# Import our new processor
import rag_processor

# Load Env
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env. Please check your .env file.")

# --- Models ---
# using Gemini 2 flash or older models for saving credits for semantic routing
MODEL_NAME_ROUTER = "gemini-2.0-flash-exp"
# Using Gemini 3 flahs as its fast and good, can change to other models if needed
MODEL_NAME = "gemini-3.0-flash-preview" 

# 1. Router Model
router_llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME_ROUTER,
    temperature=0.0,
)

# 2. Main Reasoning Model
main_llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.3,
)

# 3. Online Grounding Tool
search = GoogleSearchAPIWrapper()
search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

# --- 1. Semantic Router ---
router_prompt = PromptTemplate.from_template(
    """You are an intelligent router for an educational AI assistant.
    Analyze the user's query and classify it into one of three scopes:
    
    1. "LOW_SCOPE": The user is asking about a specific concept, definition, formula, or fact.
    2. "HIGH_SCOPE": The user is asking for a module summary, a full exam, a study plan, or synthesis of multiple topics.
    3. "TEACHING_MATERIAL": The user is providing new documents, instructions, or asking to update the knowledge base.
    
    Return ONLY a JSON object with a single key "scope".
    
    Query: {question}
    Output JSON:"""
)

router_chain = router_prompt | router_llm | JsonOutputParser()

# Retrival and generation

def get_full_context():
    # Retrieve full context from the Object Store (file system).
    # we can add an indexer function here to automatically call the correct file instead of using full_corpus.txt
    # this will be usefull for multiple documents or lots of subjects
    corpus_path = os.path.join(rag_processor.DATA_DIR, "full_corpus.txt")
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            return f.read()
    return "No documents uploaded yet."

def ground_with_search(query):
    # Perform a google search to verify or augment info.
    try:
        results = search_tool.run(query)
        return f"\n\nOnline Verification Results: \n{results}\n"
    except Exception as e:
        return f"\n\n(Online search unavailable: {str(e)})"

def hybrid_rag_chat(message, history):
    # 1. Route
    try:
        routing_decision = router_chain.invoke({"question": message})
        scope = routing_decision.get("scope", "LOW_SCOPE")
    except Exception as e:
        print(f"Routing Error: {e}, defaulting to LOW_SCOPE")
        scope = "LOW_SCOPE"
        
    print(f"DEBUG: Routing Decision: {scope}")
    
    # 2. Handle Scopes
    if scope == "TEACHING_MATERIAL":
        return "Please use the 'Upload Course Materials' button on the left to add teaching documents."

    context = ""
    source = ""
    
    if scope == "LOW_SCOPE":
        # Vector Search via Chroma
        # We access the vector store defined in rag_processor (or re-instantiate if needed, but imports work)
        retriever = rag_processor.vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(message)
        context = "\n\n".join([d.page_content for d in docs])
        source = "Vector Database (Specific Chunks)"
        
    else: # HIGH_SCOPE
        # Long Context Retrieval
        context = get_full_context()
        source = "Full Corpus (Long Context Window)"

    # 3. Online Grounding Check
    # Heuristic: If confidence is low or user asks to "verify", we search. 
    # For this demo, we'll append search if the query contains "verify" or "current" or "latest".
    if any(keyword in message.lower() for keyword in ["verify", "check", "current", "latest", "real-world"]):
        search_context = ground_with_search(message)
        context += search_context
        source += " + Online Grounding"

    # 4. Final Generation
    system_prompt = f"""You are an expert AI educational assistant using Model: {MODEL_NAME}.
    
    Source of Context: {source}
    
    Context:
    {context}
    
    Answer the user's question based strictly on the provided context. 
    If you used Online Grounding, explicitly mention the verified facts.
    If the context contains visual descriptions (from OCR), use them to explain diagrams or charts."""
    
    response = main_llm.invoke([
        ("system", system_prompt),
        ("human", message)
    ])
    
    return f"**[Mode: {scope} | Source: {source}]**\n\n{response.content}"

# --- UI Handler for Uploads ---
def handle_upload(files):
    if not files:
        return "No files selected."
    
    status = []
    for file in files:
        # Save to temp path for processor
        dest_path = os.path.join(rag_processor.DATA_DIR, os.path.basename(file.name))
        shutil.copy(file.name, dest_path)
        
        # Trigger Processor
        result = rag_processor.process_document(dest_path)
        status.append(result)
        
    return "\n".join(status)

Gradio 

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Hybrid Multimodal RAG (Gemini 3.0 Flash)")
    gr.Markdown(f"**Architecture**: Routes between **Vector Search** (Specific Facts) and **Long Context Corpus** (Module Summaries).\n**Online Grounding**: Google Search enabled for verification.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Course Materials (PDF)", file_count="multiple")
            upload_btn = gr.Button("Process Materials (RAG-ise)")
            upload_status = gr.Textbox(label="Processing Status")
            
            upload_btn.click(handle_upload, inputs=file_input, outputs=upload_status)
            
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(fn=hybrid_rag_chat)

if __name__ == "__main__":
    demo.launch()
