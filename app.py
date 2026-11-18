import asyncio
import json
import logging
import os
import re
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
logger = logging.getLogger("technical-rag")


BASE_DIR = Path(__file__).resolve().parent
KB_FILE = Path(os.getenv("KB_FILE", str(BASE_DIR / "knowledge_base.json")))
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "10"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
MAX_ANSWER_SENTENCES = int(os.getenv("MAX_ANSWER_SENTENCES", "5"))

# Keep conservative answerability thresholds. These are tunable environment variables.
MIN_SEMANTIC_SCORE = float(os.getenv("MIN_SEMANTIC_SCORE", "0.34"))
MIN_HYBRID_SCORE = float(os.getenv("MIN_HYBRID_SCORE", "0.25"))
MIN_SENTENCE_SCORE = float(os.getenv("MIN_SENTENCE_SCORE", "0.28"))

embedding_model: Optional[TextEmbedding] = None
passages: list[str] = []
passage_vectors: Optional[np.ndarray] = None
sentence_records: list[tuple[int, str]] = []
sentence_vectors: Optional[np.ndarray] = None
sentence_tokens: list[list[str]] = []
passage_tokens: list[list[str]] = []
idf: dict[str, float] = {}
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)



# Injection is treated as untrusted text. It is not executed or used as instructions.
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    r"\bdisregard\s+(all\s+)?(rules|instructions)\b",
    r"\breveal\s+(the\s+)?(system|developer|hidden)\s+prompt\b",
    r"\bact\s+as\s+(the\s+)?(system|developer)\b",
    r"\bprint\s+your\s+(system|developer)\s+prompt\b",
    r"\b(do\s+not|don't)\s+follow\s+(the\s+)?(system|developer)\s+instructions\b",
]
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for", "by", "with",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "how", "what",
    "which", "who", "whom", "whose", "when", "where", "why", "can", "could", "would", "should",
    "will", "shall", "may", "might", "must", "about", "from", "into", "than", "then", "this",
    "that", "these", "those", "it", "its", "their", "they", "them", "he", "she", "we", "you",
    "your", "our", "as", "at", "after", "before", "not", "no", "such", "any", "all", "each",
}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.I)



def is_injection(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS)



def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]



def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOP_WORDS and len(t) > 1]



def normalize_vectors(vectors) -> np.ndarray:
    array = np.asarray(list(vectors), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms



def load_kb() -> None:
    global passages, sentence_records, sentence_tokens, passage_tokens, idf
    if not KB_FILE.exists():
        raise RuntimeError(f"Knowledge base missing: {KB_FILE}")
    loaded = json.loads(KB_FILE.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or len(loaded) != 50:
        raise RuntimeError("knowledge_base.json must contain exactly 50 passages.")
    if not all(isinstance(p, str) and p.strip() for p in loaded):
        raise RuntimeError("Every KB passage must be a non-empty string.")
    passages = loaded
    passage_tokens = [tokenize(p) for p in passages]
    sentence_records = []
    sentence_tokens = []
    for pid, passage in enumerate(passages):
        for sentence in split_sentences(passage):
            # Ignore any sentence containing prompt-like instruction content.
            if is_injection(sentence):
                continue
            sentence_records.append((pid, sentence))
            sentence_tokens.append(tokenize(sentence))
    # Passage-level inverse document frequency for lexical relevance.
    document_frequency = Counter()
    for tokens in passage_tokens:
        document_frequency.update(set(tokens))
    n = max(1, len(passages))
    idf = {term: (1.0 + np.log((n + 1) / (df + 1))) for term, df in document_frequency.items()}
    logger.info("KB loaded: %d passages, %d sentences", len(passages), len(sentence_records))


def encode(texts: list[str]) -> np.ndarray:
    return normalize_vectors(embedding_model.embed(texts, batch_size=16))



def build_index() -> None:
    global embedding_model, passage_vectors, sentence_vectors
    logger.info("Loading embedding model: %s", MODEL_NAME)
    embedding_model = TextEmbedding(model_name=MODEL_NAME, threads=2)
    passage_vectors = encode(passages)
    sentence_vectors = encode([s for _, s in sentence_records])
    logger.info("Embeddings ready: %d passage vectors, %d sentence vectors", len(passages), len(sentence_records))



def lexical_score(query_tokens: list[str], document_tokens: list[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    qset = set(query_tokens)
    dset = set(document_tokens)
    matched = qset & dset
    if not matched:
        return 0.0
    total_weight = sum(idf.get(t, 1.0) for t in qset)
    matched_weight = sum(idf.get(t, 1.0) for t in matched)
    coverage = matched_weight / max(total_weight, 1e-9)
    # Small precision term helps penalize broad passages that match only a few query terms.
    precision = len(matched) / max(1, len(dset))
    return float(np.clip(0.85 * coverage + 0.15 * min(1.0, precision * 8.0), 0.0, 1.0))


def score_evidence(question: str):
    qtokens = tokenize(question)
    qvec = encode([question])[0]
    pcos = passage_vectors @ qvec
    scos = sentence_vectors @ qvec
    sentence_lex_scores = [lexical_score(qtokens, tokens) for tokens in sentence_tokens]
    best_sentence_semantic: dict[int, float] = {}
    best_sentence_lexical: dict[int, float] = {}
    for idx, (pid, _) in enumerate(sentence_records):
        best_sentence_semantic[pid] = max(best_sentence_semantic.get(pid, -1.0), float(scos[idx]))
        best_sentence_lexical[pid] = max(best_sentence_lexical.get(pid, 0.0), sentence_lex_scores[idx])

    p_lex = [lexical_score(qtokens, tokens) for tokens in passage_tokens]
    # Dense embeddings lead, lexical relevance helps disambiguate closely related concepts.
    passage_scores = []
    for pid in range(len(passages)):
        semantic = float(pcos[pid])
        sentence_semantic = best_sentence_semantic.get(pid, semantic)
        lexical = p_lex[pid]
        sentence_lexical = best_sentence_lexical.get(pid, lexical)
        hybrid = 0.58 * semantic + 0.18 * sentence_semantic + 0.16 * lexical + 0.08 * sentence_lexical
        passage_scores.append(float(np.clip(hybrid, 0.0, 1.0)))
    return qtokens, qvec, pcos, scos, sentence_lex_scores, passage_scores


def retrieve(question: str, top_k: int) -> dict:
    if embedding_model is None or passage_vectors is None or sentence_vectors is None:
        raise RuntimeError("Embedding index is still initializing.")
    qtokens, qvec, pcos, scos, sentence_lex_scores, passage_scores = score_evidence(question)
    ranked = sorted(range(len(passages)), key=lambda pid: passage_scores[pid], reverse=True)
    best_pid = ranked[0]
    best_semantic = float(pcos[best_pid])
    best_hybrid = passage_scores[best_pid]
    best_lexical = lexical_score(qtokens, passage_tokens[best_pid])

    # Abstain if evidence is weak both semantically and lexically. This avoids
    # answering from merely related passages for off-topic questions.
    if best_semantic < MIN_SEMANTIC_SCORE and best_hybrid < MIN_HYBRID_SCORE:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.10}
    if best_hybrid < MIN_HYBRID_SCORE and best_lexical < 0.12:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.10}

    candidates = ranked[:max(top_k, min(5, top_k + 2))]
    candidate_set = set(candidates)
    evidence = []
    for idx, (pid, sentence) in enumerate(sentence_records):
        if pid not in candidate_set:
            continue
        sem = float(scos[idx])
        lex = float(sentence_lex_scores[idx])
        # Sentence ranking is more precise than passage-only ranking.
        score = 0.70 * sem + 0.30 * lex
        evidence.append((score, sem, lex, pid, sentence))
    evidence.sort(key=lambda row: row[0], reverse=True)
    if not evidence:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.10}

    best_evidence = evidence[0][0]
    # A low absolute floor and a relative floor prevent irrelevant sentences from
    # being appended just because they belong to a selected passage.
    floor = max(MIN_SENTENCE_SCORE, best_evidence - 0.14)
    chosen = []
    used_pids = set()
    for score, sem, lex, pid, sentence in evidence:
        if score < floor or pid in used_pids:
            continue
        chosen.append((score, pid, sentence))
        used_pids.add(pid)
        if len(chosen) >= min(MAX_ANSWER_SENTENCES, max(1, top_k + 1)):
            break

    # When query is highly specific, keep only sentences with meaningful lexical
    # or semantic support. If semantic similarity is good for a synonym query,
    # lexical matching is not required.
    if not chosen or (best_evidence < MIN_SENTENCE_SCORE and best_semantic < MIN_SEMANTIC_SCORE):
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.10}


    sources = []
    scores = []
    for _, pid, _ in chosen:
        if pid not in sources:
            sources.append(pid)
            scores.append(round(passage_scores[pid], 4))
    # Ensure output score order matches source order and is descending.
    ordered = sorted(zip(sources, scores), key=lambda x: x[1], reverse=True)
    sources = [pid for pid, _ in ordered]
    scores = [score for _, score in ordered]
    answer = " ".join(sentence for _, _, sentence in chosen)
    # Confidence is higher for strong semantic evidence with a margin over runner-up.
    runner_up = passage_scores[ranked[1]] if len(ranked) > 1 else 0.0
    margin = max(0.0, best_hybrid - runner_up)
    confidence = float(np.clip(0.28 + 0.48 * best_hybrid + 0.20 * min(1.0, margin * 3), 0.0, 0.98))
    if len(sources) > 1 and best_hybrid > 0.50:
        confidence = min(0.98, confidence + 0.03)
    return {"answer": answer, "sources": sources, "scores": scores, "confidence": round(confidence, 4)}



@asynccontextmanager
async def lifespan(app: FastAPI):
    load_kb()
    await run_in_threadpool(build_index)
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



@app.get("/")
async def root():
    return {"status": "ok", "service": "technical-rag", "query_endpoint": "POST /query"}



@app.get("/health")
async def health():
    ready = embedding_model is not None and passage_vectors is not None and sentence_vectors is not None
    return {"status": "ready" if ready else "loading", "passages": len(passages), "model": MODEL_NAME}



@app.post("/query")
async def query_endpoint(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be blank.")
    async with request_semaphore:
        try:
            result = await run_in_threadpool(retrieve, question, request.top_k)
            # Contract guardrails: source IDs and scores are valid, ordered, and aligned.
            result["sources"] = [int(pid) for pid in result["sources"][:request.top_k]]
            result["scores"] = [float(np.clip(score, 0.0, 1.0)) for score in result["scores"][:request.top_k]]
            if result["answer"] is None:
                result["sources"], result["scores"] = [], []
                result["confidence"] = min(float(result["confidence"]), 0.39)
            else:
                result["confidence"] = max(float(result["confidence"]), 0.61)
            return result
        except Exception:
            logger.exception("Query failed")
            raise HTTPException(status_code=500, detail="Internal query processing error.")
