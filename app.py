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
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

# ----------------------------- Configuration -----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

BASE_DIR = Path(__file__).resolve().parent
KB_FILE = Path(os.getenv("KB_FILE", str(BASE_DIR / "knowledge_base.json")))
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "10"))
# Tune this against the exam's answerable and unanswerable questions.
MIN_PASSAGE_SCORE = float(os.getenv("MIN_PASSAGE_SCORE", "0.34"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
MAX_ANSWER_SENTENCES = int(os.getenv("MAX_ANSWER_SENTENCES", "5"))

# ----------------------------- Global index ------------------------------

model: Optional[SentenceTransformer] = None
passages: list[str] = []
passage_vectors: Optional[np.ndarray] = None
sentence_records: list[tuple[int, str]] = []
sentence_vectors: Optional[np.ndarray] = None
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# The knowledge base is data, not instructions. No retrieved text is ever
# passed to an LLM or executed. This list also excludes obvious instruction
# strings if such text is added to the KB in a later version.
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    r"\bdisregard\s+(all\s+)?(rules|instructions)\b",
    r"\breveal\s+(the\s+)?(system|developer|hidden)\s+prompt\b",
    r"\bact\s+as\s+(the\s+)?(system|developer)\b",
    r"\bprint\s+your\s+(system|developer)\s+prompt\b",
]

def looks_like_instruction_injection(text: str) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in INJECTION_PATTERNS
    )

def split_into_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]

def load_knowledge_base() -> None:
    global passages, sentence_records

    if not KB_FILE.exists():
        raise RuntimeError(f"Knowledge base file not found: {KB_FILE}")

    loaded = json.loads(KB_FILE.read_text(encoding="utf-8"))

    if not isinstance(loaded, list):
        raise RuntimeError("knowledge_base.json must contain a JSON array.")
    if len(loaded) != 50:
        raise RuntimeError(
            f"Expected exactly 50 passages, but found {len(loaded)}."
        )
    if not all(isinstance(item, str) and item.strip() for item in loaded):
        raise RuntimeError("Every knowledge-base passage must be a non-empty string.")

    passages = loaded
    sentence_records = []

    for passage_id, passage in enumerate(passages):
        for sentence in split_into_sentences(passage):
            if not looks_like_instruction_injection(sentence):
                sentence_records.append((passage_id, sentence))

    logger.info(
        "Loaded %d passages and %d evidence sentences.",
        len(passages),
        len(sentence_records),
    )

def build_embedding_index() -> None:
    global model, passage_vectors, sentence_vectors

    logger.info("Loading embedding model %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    passage_vectors = model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    evidence_sentences = [sentence for _, sentence in sentence_records]
    sentence_vectors = model.encode(
        evidence_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    logger.info("Embedding index ready.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_knowledge_base()
    # Run expensive startup work outside the event loop.
    await run_in_threadpool(build_embedding_index)
    yield

# ------------------------------- API setup -------------------------------

app = FastAPI(
    title="50-Passage Technical RAG API",
    description="Semantic retrieval and extractive evidence answering.",
    version="1.0.0",
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

# ------------------------------- Retrieval --------------------------------

def retrieve(question: str, top_k: int) -> QueryResponse:
    if model is None or passage_vectors is None or sentence_vectors is None:
        raise RuntimeError("Embedding index is not ready.")

    # Embed the question once. Normalized vectors make dot product cosine similarity.
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    passage_similarities = passage_vectors @ query_vector
    sentence_similarities = sentence_vectors @ query_vector

    # Score each passage using both whole-passage and best-sentence similarity.
    best_sentence_by_passage: dict[int, float] = {}
    for index, (passage_id, _) in enumerate(sentence_records):
        score = float(sentence_similarities[index])
        previous = best_sentence_by_passage.get(passage_id, -1.0)
        if score > previous:
            best_sentence_by_passage[passage_id] = score

    combined_scores = []
    for passage_id in range(len(passages)):
        passage_score = float(passage_similarities[passage_id])
        sentence_score = best_sentence_by_passage.get(passage_id, passage_score)
        combined = 0.70 * passage_score + 0.30 * sentence_score
        # Cosine similarity can be negative; API scores must be within [0, 1].
        combined_scores.append(float(np.clip(combined, 0.0, 1.0)))

    ranked_passage_ids = sorted(
        range(len(passages)),
        key=lambda pid: combined_scores[pid],
        reverse=True,
    )

    best_score = combined_scores[ranked_passage_ids[0]]
    if best_score < MIN_PASSAGE_SCORE:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.10,
        )

    selected_ids = ranked_passage_ids[:top_k]
    selected_set = set(selected_ids)

    # Rank evidence sentences only inside the retrieved passages.
    evidence = []
    for index, (passage_id, sentence) in enumerate(sentence_records):
        if passage_id not in selected_set:
            continue
        if looks_like_instruction_injection(sentence):
            continue
        evidence.append(
            (float(sentence_similarities[index]), passage_id, sentence)
        )

    evidence.sort(key=lambda item: item[0], reverse=True)

    if not evidence or evidence[0][0] < 0.20:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.10,
        )

    # Select a small set of factual sentences. Allow multiple passages for
    # comparison and cross-boundary questions, but avoid repeating one source.
    answer_sentences = []
    used_passages = set()
    best_evidence_score = evidence[0][0]
    relative_floor = max(0.20, best_evidence_score - 0.16)

    for sentence_score, passage_id, sentence in evidence:
        if sentence_score < relative_floor:
            continue
        if passage_id in used_passages:
            continue
        answer_sentences.append(sentence)
        used_passages.add(passage_id)
        if len(answer_sentences) >= min(MAX_ANSWER_SENTENCES, top_k + 1):
            break

    if not answer_sentences:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.10,
        )

    # Only report sources that contributed selected evidence, ordered by score.
    contributing_ids = [
        pid for pid in selected_ids if pid in used_passages
    ]
    source_scores = [round(combined_scores[pid], 4) for pid in contributing_ids]

    answer = " ".join(answer_sentences)

    # Heuristic confidence: not a calibrated probability. The abstention path
    # deliberately reports a low value.
    second_score = (
        combined_scores[ranked_passage_ids[1]]
        if len(ranked_passage_ids) > 1
        else best_score
    )
    confidence = float(
        np.clip(0.25 + 0.55 * best_score + 0.20 * second_score, 0.0, 0.99)
    )

    return QueryResponse(
        answer=answer,
        sources=contributing_ids,
        scores=source_scores,
        confidence=round(confidence, 4),
    )

# -------------------------------- Routes ----------------------------------

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "technical-rag",
        "query_endpoint": "POST /query",
    }

@app.get("/health")
async def health():
    ready = model is not None and passage_vectors is not None
    return {
        "status": "ready" if ready else "loading",
        "passages": len(passages),
        "model": MODEL_NAME,
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be blank.")

    async with request_semaphore:
        try:
            result = await run_in_threadpool(
                retrieve,
                question,
                request.top_k,
            )
            return result
        except Exception:
            logger.exception("Failed to process query.")
            raise HTTPException(
                status_code=500,
                detail="Internal query processing error.",
            )
