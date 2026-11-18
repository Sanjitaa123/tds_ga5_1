import asyncio
import json
import logging
import os
import re
import math
from collections import Counter
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
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
MAX_ANSWER_SENTENCES = int(os.getenv("MAX_ANSWER_SENTENCES", "4"))
ABSTAIN_SCORE = float(os.getenv("ABSTAIN_SCORE", "0.34"))

embedding_model: Optional[TextEmbedding] = None
passages: list[str] = []
passage_vectors: Optional[np.ndarray] = None
sentence_records: list[tuple[int, str]] = []
sentence_vectors: Optional[np.ndarray] = None
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Filter directive-like content before indexing or returning evidence.
INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|rules|prompts)\b",
    r"\bdisregard\s+(?:all\s+)?(?:rules|instructions|constraints|policies)\b",
    r"\b(?:reveal|print|show|output|leak)\s+(?:the\s+)?(?:system|developer|hidden|secret)\s+(?:prompt|instructions|message)\b",
    r"\b(?:system|developer)\s+prompt\b",
    r"\b(?:act|behave)\s+as\s+(?:the\s+)?(?:system|developer|administrator)\b",
    r"\b(?:override|bypass)\s+(?:the\s+)?(?:safety|security|instructions|rules)\b",
    r"\b(?:you are now|from now on)\s+(?:a|an|the)?\s*(?:system|developer|assistant)\b",
    r"\b(?:do not|don't)\s+(?:follow|obey)\s+(?:the\s+)?(?:user|previous|original)\b",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

def looks_like_instruction_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text or ""))

def sanitize_question(question: str) -> str:
    # If the query is mostly a prompt-injection attempt, abstain rather than embed it.
    if looks_like_instruction_injection(question):
        return ""
    return question.strip()

def split_into_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in pieces if p.strip()]

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_.+-]*", (text or "").lower())

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
    if not isinstance(loaded, list) or len(loaded) != 50:
        raise RuntimeError("knowledge_base.json must contain exactly 50 passages.")
    if not all(isinstance(p, str) and p.strip() for p in loaded):
        raise RuntimeError("Every passage must be a non-empty string.")
    passages = loaded
    sentence_records = []
    for pid, passage in enumerate(passages):
        for sentence in split_into_sentences(passage):
            # Sentence-level filtering means factual sentences adjacent to malicious
            # directives remain available, while directive text is never returned.
            if not looks_like_instruction_injection(sentence):
                sentence_records.append((pid, sentence))
    build_lexical_index()
    logger.info("Loaded %d passages; %d safe evidence sentences.", len(passages), len(sentence_records))

def encode_texts(texts: list[str]) -> np.ndarray:
    return normalize_vectors(embedding_model.embed(texts, batch_size=16))

def build_embedding_index() -> None:
    global embedding_model, passage_vectors, sentence_vectors
    logger.info("Loading ONNX embedding model %s", MODEL_NAME)
    embedding_model = TextEmbedding(model_name=MODEL_NAME, threads=1)
    passage_vectors = encode_texts(passages)
    sentence_vectors = encode_texts([s for _, s in sentence_records])
    logger.info("Index ready: %d passages, %d evidence sentences", len(passages), len(sentence_records))

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_knowledge_base()
    await run_in_threadpool(build_embedding_index)
    yield

app = FastAPI(title="50-Passage Technical RAG API", version="2.0.0", lifespan=lifespan)
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

# Small dependency-free BM25 to complement semantic similarity. This is particularly
# useful for exact technical terms, quantities, acronyms, and confounder pairs.
passage_tokens: list[list[str]] = []
passage_tf: list[Counter] = []
passage_lengths: list[int] = []
idf: dict[str, float] = {}
avgdl: float = 0.0

def build_lexical_index():
    global passage_tokens, passage_tf, passage_lengths, idf, avgdl
    passage_tokens = [tokenize(p) for p in passages]
    passage_tf = [Counter(toks) for toks in passage_tokens]
    passage_lengths = [len(toks) for toks in passage_tokens]
    avgdl = max(1.0, sum(passage_lengths) / max(1, len(passage_lengths)))
    df = Counter()
    for toks in passage_tokens:
        df.update(set(toks))
    n = max(1, len(passages))
    idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

def bm25_scores(question: str) -> np.ndarray:
    terms = tokenize(question)
    scores = np.zeros(len(passages), dtype=np.float32)
    if not terms:
        return scores
    k1, b = 1.5, 0.75
    for pid, tf in enumerate(passage_tf):
        dl = passage_lengths[pid]
        norm = k1 * (1 - b + b * dl / avgdl)
        for term in terms:
            f = tf.get(term, 0)
            if f:
                scores[pid] += idf.get(term, 0.0) * (f * (k1 + 1)) / (f + norm)
    maximum = float(scores.max()) if len(scores) else 0.0
    if maximum > 0:
        scores /= maximum
    return scores

def lexical_overlap(question: str, text: str) -> float:
    q = set(tokenize(question))
    t = set(tokenize(text))
    # Ignore generic stop words so overlap is driven by topic terms.
    stop = {"what","which","who","when","where","why","how","does","do","is","are",
             "the","a","an","to","of","in","on","for","with","and","or","by","from",
             "tell","explain","describe","compare","give","define","used","use"}
    q -= stop
    if not q:
        return 0.0
    return len(q & t) / len(q)

def rank_passages(question: str):
    qvec = encode_texts([question])[0]
    sem = passage_vectors @ qvec
    bm = bm25_scores(question)
    # Semantic search handles paraphrases; BM25 helps exact tokens and technical names.
    # Lexical overlap is a modest tie-breaker rather than a dominant signal.
    overlap = np.array([lexical_overlap(question, p) for p in passages], dtype=np.float32)
    sem01 = np.clip((sem + 1.0) / 2.0, 0.0, 1.0)
    combined = 0.68 * sem01 + 0.24 * bm + 0.08 * overlap
    return sem, bm, overlap, combined

def retrieve(question: str, top_k: int) -> QueryResponse:
    if embedding_model is None or passage_vectors is None or sentence_vectors is None:
        raise RuntimeError("Embedding index is not ready.")
    clean_question = sanitize_question(question)
    if not clean_question or len(tokenize(clean_question)) < 2:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.05)

    sem, bm, overlap, combined = rank_passages(clean_question)
    ranked = sorted(range(len(passages)), key=lambda i: float(combined[i]), reverse=True)
    best_pid = ranked[0]
    best_score = float(combined[best_pid])
    second_score = float(combined[ranked[1]]) if len(ranked) > 1 else 0.0
    best_sem = float(sem[best_pid])
    best_bm = float(bm[best_pid])
    best_overlap = float(overlap[best_pid])

    # Require at least some semantic relevance and a second signal for answerability.
    # Exact-term queries may pass via BM25; paraphrased questions via semantic score.
    if best_sem < 0.20 or (best_score < ABSTAIN_SCORE and best_bm < 0.18 and best_overlap < 0.20):
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.12)

    # Score evidence sentences only in the top-ranked candidate passages.
    top_ids = ranked[:max(top_k, 4)]
    candidate_set = set(top_ids)
    qvec = encode_texts([clean_question])[0]
    sentence_sim = sentence_vectors @ qvec
    evidence = []
    for i, (pid, sentence) in enumerate(sentence_records):
        if pid not in candidate_set:
            continue
        ssem = float(sentence_sim[i])
        ov = lexical_overlap(clean_question, sentence)
        # Relevance blend: semantic sentence score + question term overlap.
        score = 0.82 * ((ssem + 1.0) / 2.0) + 0.18 * ov
        evidence.append((score, ssem, ov, pid, sentence))
    evidence.sort(key=lambda e: e[0], reverse=True)

    if not evidence:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.10)

    best_evidence = evidence[0]
    # For near-domain questions, require that at least one sentence itself supports
    # the query, not merely that the containing passage is broadly related.
    if best_evidence[1] < 0.16 and best_evidence[2] < 0.20:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.12)

    # Collect concise evidence sentences, avoid duplicate passage contribution,
    # and include a second source only when it has close relevance to the best.
    answer_sentences = []
    used = set()
    evidence_floor = max(0.52, best_evidence[0] - 0.13)
    for score, ssem, ov, pid, sentence in evidence:
        if score < evidence_floor or pid in used:
            continue
        answer_sentences.append(sentence)
        used.add(pid)
        if len(answer_sentences) >= min(MAX_ANSWER_SENTENCES, max(2, top_k)):
            break

    if not answer_sentences:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.10)

    # Returned sources are evidence-bearing passages, ordered by the combined score.
    source_ids = [pid for pid in ranked[:top_k] if pid in used]
    if not source_ids:
        source_ids = [best_pid]
    source_scores = [round(float(np.clip(combined[pid], 0.0, 1.0)), 4) for pid in source_ids]
    # Ensure strictly descending presentation by source ranking (not sentence rank).
    answer = " ".join(answer_sentences)
    margin = max(0.0, best_score - second_score)
    confidence = float(np.clip(0.20 + 0.62 * best_score + 0.18 * min(1.0, margin * 3), 0.0, 0.99))
    if confidence < 0.61:
        confidence = max(0.61, confidence)  # confidence reflects the decision to answer
    return QueryResponse(answer=answer, sources=source_ids, scores=source_scores, confidence=round(confidence, 4))

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
            logger.exception("Query processing failed")
            raise HTTPException(status_code=500, detail="Internal query processing error.")
