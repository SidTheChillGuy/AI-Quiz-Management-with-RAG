import os
import shutil
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoProcessor
from pdf2image import convert_from_path
from PIL import Image

# Load Env
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Configure our vector DB for our documents on Hybrid Rag
VECTOR_DB_DIR = "vector_stores/hybrid_db"
DATA_DIR = "data"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# setup model embeddings and other stuff
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma(
    collection_name="hybrid_rag",
    embedding_function=embeddings,
    persist_directory=VECTOR_DB_DIR
)

# OCR Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OCR_MODEL_PATH = "PaddlePaddle/PaddleOCR-VL"
_ocr_model = None
_ocr_processor = None

def get_ocr_model():
    """Lazy load the OCR model."""
    global _ocr_model, _ocr_processor
    if _ocr_model is None:
        print(f"Loading OCR Model on {DEVICE}...")
        _ocr_model = AutoModelForCausalLM.from_pretrained(
            OCR_MODEL_PATH, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            device_map=DEVICE
        ).eval()
        _ocr_processor = AutoProcessor.from_pretrained(OCR_MODEL_PATH, trust_remote_code=True)
        print("OCR Model Loaded.")
    return _ocr_model, _ocr_processor

def perform_ocr(image):
    """Run OCR on a single image."""
    model, processor = get_ocr_model()
    prompt_text = "OCR:"
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(DEVICE)
    
    with torch.inference_mode():
        # Limiting tokens for speed in demo
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    return processor.batch_decode(out, skip_special_tokens=True)[0]

def add_to_vector_store(texts):
    """Add text chunks to the vector store."""
    if texts:
        vector_store.add_texts(texts)

def process_document(file_path):
    """
    Full pipeline for a single document:
    1. Copy to data dir
    2. Convert PDF to Image (if needed)
    3. Perform OCR
    4. Save Full Text (Object Store pattern)
    5. Chunk & Vectorize
    """
    filename = os.path.basename(file_path)
    print(f"Processing {filename}...")
    
    # 1. OCR / Text Extraction
    file_text = ""
    if file_path.lower().endswith('.pdf'):
        images = convert_from_path(file_path)
        for i, img in enumerate(images):
            print(f"  - OCR Page {i+1}...")
            text = perform_ocr(img)
            file_text += f"\n--- Page {i+1} ---\n{text}\n"
    else:
        # Assume text file for simplicity or extend logic
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_text = f.read()
        except Exception:
            file_text = "Error reading text file."

    # 2. Store in "Object Store" (File System)
    # We append to a global corpus file for the 'Module' scope, 
    # but ideally we'd keep separate files. For this demo, appending is fine.
    corpus_path = os.path.join(DATA_DIR, "full_corpus.txt")
    with open(corpus_path, "a", encoding="utf-8") as f:
        f.write(f"\n\nDOCUMENT: {filename}\n{file_text}")

    # 3. Chunk & Vectorize
    print("\n\nChunking and Vectorizing...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(file_text)
    add_to_vector_store(chunks)
    
    return f"Successfully processed {filename} (Pages: {len(chunks)} chunks generated)"

if __name__ == "__main__":
    # Test run
    # process_document("path/to/test.pdf")
    print("rag_processor module loaded.")
