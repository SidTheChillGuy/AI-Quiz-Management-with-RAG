# import dependencies
import os, shutil, torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from pdf2image import convert_from_path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

load_dotenv()

os.makedirs("vector_stores/chroma_langchain_db", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# check for API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "Please enter the GOOGLE_API_KEY in .env file at the root of the folder containing this code file."
    )

# setup requirements
prompt = hub.pull("rlm/rag-prompt")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(
    collection_name="simplerag",
    embedding_function=embeddings,
    persist_directory="vector_stores/chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# create helper functions
def split_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )  # change as required
    return text_splitter.split_text(text)

def append_vectorstore_data(chunks):
    global vector_store
    vector_store.add_texts(chunks)
    # put texts in vector store
    return vector_store

# Format retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question, hist):
    global vector_store
    qa_chain = (
        {
            "context": vector_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain.invoke(question)

# OCR 
model_path = "PaddlePaddle/PaddleOCR-VL"
task = "ocr" # ← change to "table" | "chart" | "formula"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
}

def OCRfunc(img):
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": PROMPTS[task]}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False,
            use_cache=True
        )
    
    outputs = processor.batch_decode(out, skip_special_tokens=True)[0]
    return outputs

def upload_files(files):
    yield "Files are uploaded, performing Vectorization"
    saved_paths = []
    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join("data", filename)
        shutil.copy(file.name, dest_path)
        saved_paths.append(dest_path)
    
    for file in saved_paths:
        images = convert_from_path(file)
        for img in images:
            outputs = OCRfunc(img)
            append_vectorstore_data(split_text(outputs))
        
        yield f"{file} processed,"


with gr.Blocks(theme=gr.themes.Glass()) as demo:
    with gr.Column():
        gr.Markdown("<h1><center>Welcome to Simple RAG</h1>")
        gr.Markdown("<h3><center>A demo implementation for simple RAG application as discussed in the paper.")
        gr.Markdown("""Working Details: \n
            * The app loads documents, converts them to searchable text. \n
            * The text is converted into embeddings and stored in a vector database. \n
            * Every query retrieves the text from Db, passes to the Answering Model along with query.""")
        gr.Markdown("""Current implementatino specific details: \n
            - Document supported type: PDF only
            - Extraction Pipeline: PDf to Image to OCR (transformers + PaddleOCR)
            - Vector Store: ChromaDB, local, persistent
            - Embeddings: Gemini 001 (API key required)
            - Similarity search operation: Inbuilt search function
            - Answering LLM: Gemini 2.5 Flash (API Key required)""")
    
    with gr.Row():
        with gr.Column():
            file_upload = gr.File(label="Upload multiple files", file_types=[".pdf"], interactive=True, file_count="multiple")
            output = gr.Textbox(label="Saved File Paths", lines=11)
            file_upload.upload(upload_files, inputs=file_upload, outputs=output)


        with gr.Column():
            gr.ChatInterface(fn=rag_chain, type="messages",chatbot=gr.Chatbot(height=450, show_copy_button=True, type="messages"),
                textbox=gr.Textbox(placeholder="Start chatting", container=True, max_lines=5))
        
        

demo.launch()
