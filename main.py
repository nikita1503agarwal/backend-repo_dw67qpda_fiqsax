import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict

from database import db, create_document, get_documents
from schemas import Document, Chunk, ChatMessage

from io import BytesIO
from pypdf import PdfReader
from collections import Counter
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "PDF Chat API is up"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunk_count: int


# --------- Lightweight text processing (no heavy deps) ---------

def tokenize(text: str) -> List[str]:
    # Very simple tokenizer for demo purposes
    return [t for t in ''.join([c.lower() if c.isalnum() or c.isspace() else ' ' for c in text]).split() if t]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunks.append(" ".join(chunk_words))
        i += max(1, chunk_size - overlap)
    return chunks or [text]


def build_idf(chunks: List[str]) -> Dict[str, float]:
    # Document frequency across chunks
    N = len(chunks)
    df: Dict[str, int] = {}
    for ch in chunks:
        terms = set(tokenize(ch))
        for t in terms:
            df[t] = df.get(t, 0) + 1
    # Smooth IDF
    idf: Dict[str, float] = {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}
    return idf


def vectorize(text: str, idf: Dict[str, float]) -> Dict[str, float]:
    toks = tokenize(text)
    tf = Counter(toks)
    vec: Dict[str, float] = {}
    for term, freq in tf.items():
        if term in idf:
            vec[term] = (freq / len(toks)) * idf[term]
    return vec


def cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    # Dot product over intersection
    dot = 0.0
    for k in v1.keys() & v2.keys():
        dot += v1[k] * v2[k]
    n1 = math.sqrt(sum(x*x for x in v1.values())) or 1e-12
    n2 = math.sqrt(sum(x*x for x in v2.values())) or 1e-12
    return dot / (n1 * n2)


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    content = await file.read()

    try:
        reader = PdfReader(BytesIO(content))
        pages = len(reader.pages)
        full_text = []
        for i in range(pages):
            full_text.append(reader.pages[i].extract_text() or "")
        full_text = "\n".join(full_text)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF")

    # Chunking
    chunks = chunk_text(full_text)

    # Build IDF across chunks and store it
    idf = build_idf(chunks)

    # Persist document metadata
    doc_meta = Document(filename=file.filename, pages=pages, chunk_count=len(chunks), status="ready")
    doc_id = create_document("document", doc_meta)

    # Store chunks (text only)
    for idx, text_chunk in enumerate(chunks):
        create_document("chunk", Chunk(doc_id=doc_id, text=text_chunk, index=idx))

    # Store idf object
    create_document("chunk", {"doc_id": doc_id, "text": "__idf__", "index": -1, "idf": idf})

    return UploadResponse(doc_id=doc_id, filename=file.filename, pages=pages, chunk_count=len(chunks))


class ChatRequest(BaseModel):
    doc_id: str
    message: str
    history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[int]


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_doc(payload: ChatRequest):
    # Load chunks + idf
    records = get_documents("chunk", {"doc_id": payload.doc_id})
    if not records:
        raise HTTPException(status_code=404, detail="Document not found or not processed yet")

    chunks: List[str] = []
    idf: Optional[Dict[str, float]] = None
    index_map: List[int] = []
    for r in records:
        if r.get("index") == -1 and r.get("text") == "__idf__":
            idf = r.get("idf")
        elif r.get("index") is not None and r.get("index") >= 0:
            index_map.append(int(r["index"]))
            chunks.append(r["text"])

    if idf is None:
        raise HTTPException(status_code=500, detail="Index data missing")

    # Vectorize chunks and query
    chunk_vecs = [vectorize(text, idf) for text in chunks]
    q_vec = vectorize(payload.message, idf)

    # Compute similarities
    sims = [(i, cosine(q_vec, cv)) for i, cv in enumerate(chunk_vecs)]
    sims.sort(key=lambda x: x[1], reverse=True)
    top = [i for i, _ in sims[:3]]

    selected = [chunks[i] for i in top]
    context = "\n\n".join(selected)
    answer = ". ".join(context.split(". ")[:3]).strip() or "I couldn't find relevant information in the document."

    # Map back to original chunk indices for sources
    sources = [index_map[i] for i in top]
    return ChatResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
