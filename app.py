import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding
from starlette.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

BASE_DIR = Path(__file__).resolve().parent
KB_FILE = Path(os.getenv("KB_FILE", str(BASE_DIR / "knowledge_base.json")))
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "10"))
MIN_PASSAGE_SCORE = float(os.getenv("MIN_PASSAGE_SCORE", "0.34"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))
MAX_ANSWER_SENTENCES = int(os.getenv("MAX_ANSWER_SENTENCES", "5"))

# Use ONNX Runtime through FastEmbed instead of PyTorch.
embedding_model: Optional[TextEmbedding] = None
passages: list[str] = []
passage_vectors: Optional[np.ndarray] = None
sentence_records: list[tuple[int, str]] = []
sentence_vectors: Optional[np.ndarray] = None
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    r"\bdisregard\s+(all\s+)?(rules|instructions)\b",
    r"\breveal\s+(the\s+)?(system|developer|hidden)\s+prompt\b",
    r"\bact\s+as\s+(the\s+)?(system|developer)\b",
    r"\bprint\s+your\s+(system|developer)\s+prompt\b",
]

def looks_like_instruction_injection(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in INJECTION_PATTERNS)

def split_into_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]

def normalize_vectors(vectors) -> np.ndarray:
    arr = np.asarray(list(vectors), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def load_knowledge_base() -> None:
    global passages, sentence_records
    if not KB_FILE.exists():
        raise RuntimeError(f"Knowledge base file not found: {KB_FILE}")
    loaded = json.loads(KB_FILE.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise RuntimeError("knowledge_base.json must contain a JSON array.")
    if len(loaded) != 50:
        raise RuntimeError(f"Expected exactly 50 passages, but found {len(loaded)}.")
    if not all(isinstance(item, str) and item.strip() for item in loaded):
        raise RuntimeError("Every knowledge-base passage must be a non-empty string.")
    passages = loaded
    sentence_records = []
    for pid, passage in enumerate(passages):
        for sentence in split_into_sentences(passage):
            if not looks_like_instruction_injection(sentence):
                sentence_records.append((pid, sentence))
    logger.info("Loaded %d passages and %d evidence sentences.", len(passages), len(sentence_records))

def encode_texts(texts: list[str]) -> np.ndarray:
    # FastEmbed returns an iterator of vectors. Normalize explicitly for cosine/dot scores.
    return normalize_vectors(embedding_model.embed(texts, batch_size=16))

def build_embedding_index() -> None:
    global embedding_model, passage_vectors, sentence_vectors
    logger.info("Loading ONNX embedding model %s", MODEL_NAME)
    embedding_model = TextEmbedding(model_name=MODEL_NAME, threads=1)
    passage_vectors = encode_texts(passages)
    evidence_sentences = [sentence for _, sentence in sentence_records]
    sentence_vectors = encode_texts(evidence_sentences)
    logger.info("Embedding index ready: passages=%s, sentences=%s", passage_vectors.shape, sentence_vectors.shape)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_knowledge_base()
    await run_in_threadpool(build_embedding_index)
    yield

app = FastAPI(
    title="50-Passage Technical RAG API",
    description="Semantic retrieval and extractive evidence answering.",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)

class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float

def retrieve(question: str, top_k: int) -> QueryResponse:
    if embedding_model is None or passage_vectors is None or sentence_vectors is None:
        raise RuntimeError("Embedding index is not ready.")
    query_vector = encode_texts([question])[0]
    passage_similarities = passage_vectors @ query_vector
    sentence_similarities = sentence_vectors @ query_vector

    best_sentence_by_passage: dict[int, float] = {}
    for idx, (pid, _) in enumerate(sentence_records):
        score = float(sentence_similarities[idx])
        if score > best_sentence_by_passage.get(pid, -1.0):
            best_sentence_by_passage[pid] = score

    combined_scores = []
    for pid in range(len(passages)):
        pscore = float(passage_similarities[pid])
        sscore = best_sentence_by_passage.get(pid, pscore)
        combined = 0.70 * pscore + 0.30 * sscore
        combined_scores.append(float(np.clip(combined, 0.0, 1.0)))

    ranked_ids = sorted(range(len(passages)), key=lambda pid: combined_scores[pid], reverse=True)
    best_score = combined_scores[ranked_ids[0]]
    if best_score < MIN_PASSAGE_SCORE:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.10)

    selected_ids = ranked_ids[:top_k]
    selected_set = set(selected_ids)
    evidence = []
    for idx, (pid, sentence) in enumerate(sentence_records):
        if pid not in selected_set or looks_like_instruction_injection(sentence):
            continue
        evidence.append((float(sentence_similarities[idx]), pid, sentence))
    evidence.sort(key=lambda x: x[0], reverse=True)

    if not evidence or evidence[0][0] < 0.20:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.10)

    answer_sentences = []
    used_passages = set()
    best_evidence_score = evidence[0][0]
    relative_floor = max(0.20, best_evidence_score - 0.16)
    for score, pid, sentence in evidence:
        if score < relative_floor or pid in used_passages:
            continue
        answer_sentences.append(sentence)
        used_passages.add(pid)
        if len(answer_sentences) >= min(MAX_ANSWER_SENTENCES, top_k + 1):
            break

    if not answer_sentences:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.10)

    contributing_ids = [pid for pid in selected_ids if pid in used_passages]
    source_scores = [round(combined_scores[pid], 4) for pid in contributing_ids]
    answer = " ".join(answer_sentences)
    second_score = combined_scores[ranked_ids[1]] if len(ranked_ids) > 1 else best_score
    confidence = float(np.clip(0.25 + 0.55 * best_score + 0.20 * second_score, 0.0, 0.99))
    return QueryResponse(answer=answer, sources=contributing_ids, scores=source_scores, confidence=round(confidence, 4))

@app.get("/")
async def root():
    return {"status": "ok", "service": "technical-rag", "query_endpoint": "POST /query"}

@app.get("/health")
async def health():
    ready = embedding_model is not None and passage_vectors is not None and sentence_vectors is not None
    return {"status": "ready" if ready else "loading", "passages": len(passages), "model": MODEL_NAME}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be blank.")
    async with request_semaphore:
        try:
            return await run_in_threadpool(retrieve, question, request.top_k)
        except Exception:
            logger.exception("Failed to process query.")
            raise HTTPException(status_code=500, detail="Internal query processing error.")
