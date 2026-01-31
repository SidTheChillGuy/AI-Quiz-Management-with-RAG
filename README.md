# AI Quiz Management with RAG

A comprehensive exploration of Retrieval-Augmented Generation (RAG) architectures designed for educational content and quiz generation. This project implements and compares three distinct RAG strategies to overcome common limitations in AI-driven education tools.

## Implementations

### 1. Naïve RAG (`simplerag.py`)
The baseline implementation representing the standard RAG approach.
- **Mechanism**: Splits documents into fixed-size chunks (1000 chars), embeds them, and performs simple semantic search.
- **Multimodal**: Uses PaddleOCR to ingest PDFs but flattens visual data into text.
- **Pros**: Simple, fast for specific fact retrieval.
- **Cons**: Suffers from context fragmentation and "multimodal blindness" (loses visual reasoning context).

### 2. Hybrid Multimodal RAG (`hybridrag.py`)
An advanced, dynamically routed system designed for complex intent.
- **Semantic Router**: Uses **Gemini 2.0 Flash** to classify user queries into:
    - `LOW_SCOPE`: Specific facts (routes to Vector DB).
    - `HIGH_SCOPE`: Broad summaries (routes to full document processing with Long Context).
    - `TEACHING_MATERIAL`: New content injection.
- **Architecture**: Separates ingestion (`rag_processor.py`) from retrieval.
- **Online Grounding**: Integrates **Google Search** to verify answers with real-time data.
- **Model**: Powers reasoning with **Gemini 3.0 Flash Preview**.

### 3. Parent-Child RAG (`parent_child_rag.py`)
Implements the "Small-to-Big" retrieval strategy to solve the precision-context trade-off.
- **Mechanism**: 
    - Indexes **Small Chunks** (400 chars) for high-precision vector search.
    - Retrieves **Parent Chunks** (2000 chars) that contain the match + surrounding context.
- **Benefit**: Ensures the LLM receives the full explanation of a concept, not just the isolated sentence that matched the query.

## Installation & Setup

1. **Clone the repository**

2. **Install Dependencies**:
   ```bash
   pip install langchain-chroma langchain-google-genai langchain-community gradio pdf2image paddlepaddle paddleocr python-dotenv
   ```
   *Note: Windows users must install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) and add it to the system PATH for PDF processing.*

3. **Environment Setup**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
   *(Get your key from [Google AI Studio](https://aistudio.google.com/app/api-keys))*

## Usage

Run the desired architecture using Python. Each script launches a local Gradio web interface.

**Run Naïve RAG:**
```bash
python simplerag.py
```

**Run Hybrid RAG:**
```bash
python hybridrag.py
```

**Run Parent-Child RAG:**
```bash
python parent_child_rag.py
```

## Project Structure

- `simplerag.py`: Baseline logic.
- `hybridrag.py`: Main hybrid app with routing and search.
- `rag_processor.py`: Helper module for Hybrid RAG (OCR, Ingestion).
- `parent_child_rag.py`: Parent-Child implementation.
- `data/`: Stores uploaded/processed documents.
- `vector_stores/`: Persisted Vector Databases (Chroma).
