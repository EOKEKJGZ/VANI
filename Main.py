"""
Main.py — Enterprise Government Schemes RAG System

================================================
This is the core RAG system for Government Schemes & Welfare Initiatives.
It ingests updated_data.csv, builds a hybrid dense + sparse vector index,
leverages a LangGraph StateGraph for decision making, and provides
conversational retrieval with SQLite chat memory.
"""

import os
import io 
import sys 
import re
import ssl
import json
import math
import time
import pickle
import hashlib
import sqlite3
import textwrap
import warnings
import threading
from typing import Optional, TypedDict, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np
import pandas as pd

if hasattr(torch, "set_num_threads"):
    try:
        torch.set_num_threads(os.cpu_count() or 8)
    except Exception:
        pass

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


device = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# SSL / CORPORATE PROXY FIX
# MUST be the very first thing — before importing httpx, requests,
# huggingface_hub, or sentence_transformers, which all cache their
# SSL context at import time.
# ══════════════════════════════════════════════════════════════════════════════

# 1. Disable Python's built-in SSL verification globally
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# 2. Tell HuggingFace Hub to skip SSL (covers hf_hub_download & snapshot_download)
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "warning"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 3. Tell tokenizers / transformers to skip SSL
os.environ["CURL_CA_BUNDLE"] = ""               # disables curl cert checking
os.environ["REQUESTS_CA_BUNDLE"] = ""           # disables requests cert checking

# 4. Monkey-patch `requests` to never verify certs
try:
    import requests
    requests.packages.urllib3.disable_warnings()  # type: ignore
    _orig_request = requests.Session.request
    def _no_verify_request(self, method, url, **kwargs):
        kwargs["verify"] = False
        return _orig_request(self, method, url, **kwargs)
    requests.Session.request = _no_verify_request
except Exception:
    pass

# 5. Monkey-patch httpx to never verify certs
try:
    import httpx
    _orig_httpx_init = httpx.Client.__init__
    def _httpx_no_verify_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_httpx_init(self, *args, **kwargs)
    httpx.Client.__init__ = _httpx_no_verify_init

    _orig_async_init = httpx.AsyncClient.__init__
    def _httpx_async_no_verify_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_async_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = _httpx_async_no_verify_init
except Exception:
    pass

# 6. Suppress all SSL / InsecureRequest warnings
warnings.filterwarnings("ignore")
import urllib3
urllib3.disable_warnings()

# ── Safe imports ───────────────────────────────────────────────────────────
import httpx
import certifi


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH        = "updated_data.csv"
CHUNKS_CACHE    = "chunks_cache_schemes_v1.json"
QDRANT_DB_DIR   = "qdrant_db_schemes_v1"
CHAT_DB_PATH    = "chat_history.db"
BM25_CACHE      = "bm25_schemes_index.pkl"

COLLECTION_NAME = "government_schemes_v1"

# Chunking & Retrieval Parameters
CHUNK_SIZE      = 450    # target words per chunk
CHUNK_OVERLAP   = 80     # overlap words for continuous text

DENSE_TOP_K     = 10     # dense retrieval candidates from Qdrant
SPARSE_TOP_K    = 10     # BM25 candidates
RERANK_TOP_N    = 6      # after cross-encoder reranking
CONTEXT_CHARS   = 12000  # max characters fed to LLM context
DENSE_WEIGHT    = 0.65   # Weightage for dense vector search
SPARSE_WEIGHT   = 0.35   # Weightage for BM25 keyword search

# LLM & Embedding Models
LLM_MODEL       = "openai/gpt-oss-20b"
EMBED_MODEL     = "BAAI/bge-base-en-v1.5"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ------------------------------------------------------------------
# Terminal Output Capture
# ------------------------------------------------------------------

TERMINAL_LOG = io.StringIO()

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class Tee:
    def __init__(self, file):
        self.file = file

    def write(self, message):
        try:
            self.file.write(message)
        except (UnicodeEncodeError, Exception):
            try:
                enc = getattr(self.file, "encoding", "utf-8") or "utf-8"
                self.file.write(message.encode(enc, errors="replace").decode(enc))
            except Exception:
                pass
        try:
            self.file.flush()
        except Exception:
            pass
        try:
            TERMINAL_LOG.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.file.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.file, "isatty", lambda: False)()

    def fileno(self):
        return getattr(self.file, "fileno", lambda: 1)()


sys.stdout = Tee(sys.__stdout__)
sys.stderr = Tee(sys.__stderr__)


def get_terminal_output() -> str:
    """Return the recent terminal logs captured in-memory."""
    return TERMINAL_LOG.getvalue()[-15000:]


# ══════════════════════════════════════════════════════════════════════════════
# 2. GROQ CLIENT & LLM WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

from groq import Groq

_api_key = os.environ.get("GROQ_API_KEY", "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")


def _make_groq() -> Groq:
    """Returns a fresh Groq client with custom SSL verification disabled."""
    return Groq(
        api_key=_api_key,
        http_client=httpx.Client(verify=False),
    )


def _llm(messages: list, temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """Wrapper around Groq chat completions with exponential retry."""
    last_err = None
    for attempt in range(3):
        try:
            client = _make_groq()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after 3 attempts: {last_err}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CSV PARSING & STRUCTURE-AWARE SCHEME CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def _clean_text(val: Any) -> str:
    """Normalize string and remove extra whitespace."""
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()


def _load_and_chunk_schemes(csv_path: str) -> list[dict]:
    """
    Parse updated_data.csv and convert each government scheme record into
    optimized, semantic structured chunks with rich metadata.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    print(f"[Chunking] Reading schemes dataset from {csv_path} ...")
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df.fillna("")

    chunks: list[dict] = []
    chunk_counter = 0

    for idx, row in df.iterrows():
        scheme_name = _clean_text(row.get("scheme_name", ""))
        slug        = _clean_text(row.get("slug", ""))
        level       = _clean_text(row.get("level", "Central/State"))
        category    = _clean_text(row.get("schemeCategory", "General"))
        tags        = _clean_text(row.get("tags", ""))
        details     = _clean_text(row.get("details", ""))
        benefits    = _clean_text(row.get("benefits", ""))
        eligibility = _clean_text(row.get("eligibility", ""))
        application = _clean_text(row.get("application", ""))
        documents   = _clean_text(row.get("documents", ""))

        if not scheme_name:
            continue

        header = f"[Scheme: {scheme_name}]\n[Level: {level} | Category: {category} | Tags: {tags}]"

        # Construct sections
        sec_overview = f"{header}\n\nOverview & Details:\n{details}\n\nBenefits:\n{benefits}".strip()
        sec_application = f"{header}\n\nEligibility Criteria:\n{eligibility}\n\nHow to Apply:\n{application}\n\nRequired Documents:\n{documents}".strip()

        combined_full = f"{header}\n\nOverview & Details:\n{details}\n\nBenefits:\n{benefits}\n\nEligibility:\n{eligibility}\n\nHow to Apply:\n{application}\n\nRequired Documents:\n{documents}".strip()

        # If total words <= 450, keep as a single unified chunk
        if len(combined_full.split()) <= CHUNK_SIZE:
            chunks.append({
                "chunk_id": chunk_counter,
                "scheme_name": scheme_name,
                "slug": slug,
                "level": level,
                "category": category,
                "tags": tags,
                "section": "Complete Scheme Information",
                "text": combined_full,
            })
            chunk_counter += 1
        else:
            # Chunk 1: Overview & Benefits
            if details or benefits:
                chunks.append({
                    "chunk_id": chunk_counter,
                    "scheme_name": scheme_name,
                    "slug": slug,
                    "level": level,
                    "category": category,
                    "tags": tags,
                    "section": "Overview & Benefits",
                    "text": sec_overview,
                })
                chunk_counter += 1

            # Chunk 2: Eligibility, Application & Documents
            if eligibility or application or documents:
                chunks.append({
                    "chunk_id": chunk_counter,
                    "scheme_name": scheme_name,
                    "slug": slug,
                    "level": level,
                    "category": category,
                    "tags": tags,
                    "section": "Eligibility & Application Process",
                    "text": sec_application,
                })
                chunk_counter += 1

    print(f"[Chunking] Generated {len(chunks)} chunks from {len(df)} schemes in CSV.")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 4. EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════════════════

from sentence_transformers import SentenceTransformer

_embed_model: Optional[SentenceTransformer] = None
_query_embedding_cache: dict[str, np.ndarray] = {}
MAX_QUERY_CACHE = 1000


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = _load_sentence_transformer(EMBED_MODEL)
    return _embed_model


def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Robust SentenceTransformer loader with offline cache and proxy fallback."""
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as e1:
        if "SSL" not in str(e1) and "certificate" not in str(e1).lower() and "closed" not in str(e1).lower():
            raise

    print(f"[SSL] Normal model load failed ({type(e1).__name__}). Trying snapshot download ...")
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=model_name,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        return SentenceTransformer(local_dir, device=device)
    except Exception as e2:
        print(f"[SSL] snapshot_download failed: {e2}")

    raise RuntimeError(f"Could not load embedding model '{model_name}'.")


def embed(texts: list[str] | str, batch_size: int = 64) -> np.ndarray:
    """Encode texts using BGE embedding model with LRU cache and L2 normalization."""
    model = _get_embed_model()

    if isinstance(texts, str):
        key = texts.strip().lower()
        cached = _query_embedding_cache.get(key)
        if cached is not None:
            return cached

        vec = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if len(_query_embedding_cache) >= MAX_QUERY_CACHE:
            _query_embedding_cache.pop(next(iter(_query_embedding_cache)))
        _query_embedding_cache[key] = vec
        return vec

    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
        batch_size=batch_size,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. BM25 SPARSE RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

class BM25:
    """Lightweight and fast BM25 implementation for lexical keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus_size  = 0
        self.avgdl        = 0.0
        self.doc_freqs: list[dict] = []
        self.idf: dict[str, float] = {}
        self.doc_lens: list[int] = []
        self._corpus: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z0-9]{2,}\b", text.lower())

    def fit(self, corpus: list[str]):
        tokenized = [self._tokenize(d) for d in corpus]
        self._corpus = tokenized
        self.corpus_size = len(tokenized)
        self.doc_lens = [len(d) for d in tokenized]
        self.avgdl = sum(self.doc_lens) / max(self.corpus_size, 1)

        df: dict[str, int] = {}
        for doc in tokenized:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1
            )

        self.doc_freqs = []
        for doc in tokenized:
            tf: dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self.doc_freqs.append(tf)

    def get_scores(self, query: str) -> np.ndarray:
        q_terms = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for term in q_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf_doc in enumerate(self.doc_freqs):
                tf = tf_doc.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * num / den
        return scores

    def top_k(self, query: str, k: int) -> list[int]:
        scores = self.get_scores(query)
        return np.argsort(scores)[::-1][:k].tolist()

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "doc_freqs": self.doc_freqs,
            "idf": self.idf,
            "doc_lens": self.doc_lens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BM25":
        inst = cls(k1=d.get("k1", 1.5), b=d.get("b", 0.75))
        inst.corpus_size = d.get("corpus_size", 0)
        inst.avgdl = d.get("avgdl", 0.0)
        inst.doc_freqs = d.get("doc_freqs", [])
        inst.idf = d.get("idf", {})
        inst.doc_lens = d.get("doc_lens", [])
        return inst


class _CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to resolve BM25 across module namespace differences."""
    def find_class(self, module, name):
        if name == "BM25":
            return BM25
        return super().find_class(module, name)


# ══════════════════════════════════════════════════════════════════════════════
# 6. VECTOR STORE (QDRANT)
# ══════════════════════════════════════════════════════════════════════════════

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
)

_qdrant: Optional[QdrantClient] = None
bm25_index: Optional[BM25] = None
chunks: list[dict] = []
chunk_lookup: dict[int, dict] = {}


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(path=QDRANT_DB_DIR, force_disable_check_same_thread=True)
    return _qdrant


def qdrant_search(query_vector: list[float], limit: int):
    """Compatibility wrapper for Qdrant client."""
    client = _get_qdrant()
    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
        )
        return result.points
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. INDEX BUILD & CACHE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _csv_fingerprint(path: str) -> str:
    if not os.path.exists(path):
        return ""
    stat = os.stat(path)
    return f"{stat.st_size}_{stat.st_mtime}"


def _cache_valid() -> bool:
    if not os.path.exists(CHUNKS_CACHE) or not os.path.exists(QDRANT_DB_DIR) or not os.path.exists(BM25_CACHE):
        return False
    try:
        with open(CHUNKS_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("fingerprint") == _csv_fingerprint(CSV_PATH)
    except Exception:
        return False


def _build_index():
    global chunks, chunk_lookup, bm25_index

    print(f"[Index] Building index from {CSV_PATH} ...")
    chunks = _load_and_chunk_schemes(CSV_PATH)
    texts = [c["text"] for c in chunks]
    embed_texts = [c["text"][:280] for c in chunks]

    print(f"[Index] Embedding {len(chunks)} chunks ...")
    vectors = embed(embed_texts, batch_size=128)

    # Setup Qdrant
    qdrant = _get_qdrant()
    try:
        qdrant.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vectors.shape[1],
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
    )

    # Upsert in batches
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]
        points = [
            PointStruct(
                id=c["chunk_id"],
                vector=batch_vectors[j].tolist(),
                payload=c,
            )
            for j, c in enumerate(batch_chunks)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

    print(f"[Index] Qdrant upsert done ({len(chunks)} points)")

    # Build BM25
    bm25_index = BM25()
    bm25_index.fit(texts)
    print("[Index] BM25 fitted")

    # Persist caches
    fingerprint = _csv_fingerprint(CSV_PATH)
    with open(CHUNKS_CACHE, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "chunks": chunks}, f, ensure_ascii=False)
    with open(BM25_CACHE, "wb") as f:
        pickle.dump(bm25_index, f)

    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    print("[Index] Build complete ✓")


def _load_index():
    global chunks, chunk_lookup, bm25_index
    print("[Index] Loading cached index ...")
    with open(CHUNKS_CACHE, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    with open(BM25_CACHE, "rb") as f:
        loaded = _CustomUnpickler(f).load()
        if isinstance(loaded, dict):
            bm25_index = BM25.from_dict(loaded)
        else:
            bm25_index = loaded

    _get_qdrant()
    print(f"[Index] Loaded {len(chunks)} chunks and BM25 index.")


def initialise():
    if _cache_valid():
        print("Loading cached index from disk...")
        _load_index()
    else:
        print("CSV modified or cache missing. Rebuilding index...")
        _build_index()


# ══════════════════════════════════════════════════════════════════════════════
# 8. CROSS-ENCODER RERANKER
# ══════════════════════════════════════════════════════════════════════════════

initialise()

from sentence_transformers import CrossEncoder

print("Loading reranker...")
_reranker = CrossEncoder(RERANK_MODEL, trust_remote_code=True, device=device)
print("Reranker loaded.")


def _get_reranker() -> CrossEncoder:
    return _reranker


_get_embed_model()
_get_reranker()


# ══════════════════════════════════════════════════════════════════════════════
# 9. HYBRID RETRIEVAL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _dense_retrieve(query: str, top_k: int) -> list:
    """Qdrant dense vector search."""
    qvec = embed(query)
    hits = qdrant_search(qvec.tolist(), top_k)
    return hits


def _sparse_retrieve(query: str, top_k: int) -> list[int]:
    """BM25 sparse search."""
    if bm25_index is None:
        return []
    return bm25_index.top_k(query, k=top_k)


def _rrf_merge(dense_hits: list, sparse_ids: list[int], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over dense and sparse ranked lists."""
    scores: dict[int, float] = {}

    for rank, hit in enumerate(dense_hits):
        cid = hit.payload["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + DENSE_WEIGHT * (1.0 / (k + rank + 1))

    for rank, cid in enumerate(sparse_ids):
        scores[cid] = scores.get(cid, 0.0) + SPARSE_WEIGHT * (1.0 / (k + rank + 1))

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [chunk_lookup[cid] for cid in sorted_ids if cid in chunk_lookup]


def _rerank(query: str, chunks_in: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """Cross-encoder reranking."""
    if not chunks_in:
        return []
    pairs = [(query, c["text"]) for c in chunks_in[:12]]
    scores = _get_reranker().predict(pairs, batch_size=8)
    ranked = sorted(zip(chunks_in[:12], scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]


def _expand_scheme_context(top_chunks: list[dict]) -> list[dict]:
    """
    Scheme-aware expansion: for top matched schemes, pull companion chunks
    belonging to the same scheme to ensure full context (eligibility + benefits + application).
    """
    seen_chunk_ids = set()
    expanded = []
    top_schemes = {c["scheme_name"] for c in top_chunks[:3]}

    for c in top_chunks:
        if c["chunk_id"] not in seen_chunk_ids:
            expanded.append(c)
            seen_chunk_ids.add(c["chunk_id"])

    # Pull related chunks for the top schemes
    for other_c in chunks:
        if other_c["scheme_name"] in top_schemes and other_c["chunk_id"] not in seen_chunk_ids:
            expanded.append(other_c)
            seen_chunk_ids.add(other_c["chunk_id"])

    return expanded


def retrieve(question: str, top_k: int = DENSE_TOP_K) -> list[dict]:
    """Parallel Hybrid Retrieval: Dense + Sparse -> RRF -> Cross-Encoder -> Scheme Expansion."""
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(_dense_retrieve, question, top_k)
        sparse_future = executor.submit(_sparse_retrieve, question, SPARSE_TOP_K)
        dense_hits = dense_future.result()
        sparse_ids = sparse_future.result()

    merged = _rrf_merge(dense_hits, sparse_ids)
    reranked = _rerank(question, merged, top_n=RERANK_TOP_N)
    expanded = _expand_scheme_context(reranked)

    print(f"[Retrieval] Retrived {len(expanded)} scheme context chunks in {time.time()-total_start:.2f}s")
    return expanded


# ══════════════════════════════════════════════════════════════════════════════
# 10. CORRECTIVE RAG (CRAG) EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _llm_relevance_score(question: str, chunks_in: list[dict]) -> float:
    """Evaluate whether retrieved schemes are relevant to user question."""
    if not chunks_in:
        return 0.0
    sample_text = "\n\n".join(c["text"][:300] for c in chunks_in[:4])
    prompt = f"""You are evaluating whether retrieved Government Scheme records are relevant to a citizen's question.

Question: {question}

Retrieved Excerpts:
{sample_text}

On a scale of 0 to 10, how well do these scheme records match or answer the question?
Reply with ONLY a single integer from 0 to 10."""
    try:
        resp = _llm([{"role": "user", "content": prompt}], temperature=0, max_tokens=5)
        score = int(re.search(r"\d+", resp).group()) / 10.0
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 0.6


def corrective_retrieve(question: str):
    results = retrieve(question, top_k=DENSE_TOP_K)
    score = _llm_relevance_score(question, results)
    found = score >= 0.4
    print(f"[CRAG] Relevance score = {score:.2f} (found={found})")
    return results, found


# ══════════════════════════════════════════════════════════════════════════════
# 11. CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_context(chunks_in: list[dict], max_chars: int = CONTEXT_CHARS) -> str:
    """Format retrieved scheme chunks cleanly for LLM input."""
    parts = []
    used = 0

    for c in chunks_in:
        block = f"--- SCHEME RECORD ---\n{c['text']}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# 12. QUESTION PROCESSING (REWRITE & DECOMPOSE)
# ══════════════════════════════════════════════════════════════════════════════

def rewrite_question(question: str, chat_history: list[dict]) -> str:
    """Resolve pronouns and conversational follow-ups into standalone scheme queries."""
    recent = chat_history[-6:]
    if not recent:
        return question

    history_str = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    )

    pronouns = {"it", "they", "them", "their", "this", "that", "those", "these", "its", "scheme", "yojana"}
    question_lower = question.lower()
    needs_rewrite = any(
        p in question_lower.split() or f"{p}?" in question_lower or f"{p}." in question_lower
        for p in pronouns
    )

    if not needs_rewrite:
        return question

    prompt = f"""You are a query rewriting assistant for a Government Schemes RAG system.
Convert the user's latest follow-up question into a standalone search query by replacing pronouns with the specific Scheme Name or topic discussed.

Rules:
1. Replace pronouns (it, they, this, that, etc.) with the scheme name from conversation history.
2. Never answer the question.
3. Preserve the exact user intent.
4. If already standalone, return unchanged.

Conversation History:
{history_str}

Latest Question:
{question}

Return ONLY the rewritten standalone question."""

    try:
        rewritten = _llm([{"role": "user", "content": prompt}], temperature=0, max_tokens=100).strip()
        if len(rewritten) > 4:
            print(f"[Rewrite] {question} -> {rewritten}")
            return rewritten
    except Exception:
        pass
    return question


def decompose_question(question: str) -> list[str]:
    """Break complex, situational, or comparative questions into focused search queries."""
    COMPLEX_KEYWORDS = {
        "compare", "difference", "versus", "vs", "both", "all", "list all",
        "which schemes", "eligibility and benefits", "multiple schemes",
        "destroyed", "rainfall", "rain", "loss", "damage", "calamity", "disaster",
        "what should i do", "how to claim", "pacs", "dispute", "grievance",
        "compensation", "relief", "insurance", "flood", "drought"
    }
    words = question.lower().split()
    if len(words) < 6 and not any(kw in question.lower() for kw in COMPLEX_KEYWORDS):
        return []

    prompt = f"""You are an expert search query planner for a Cooperative Governance & Government Schemes database.
Convert the user's question, situation, or problem statement into 2-4 focused, specific search queries that will retrieve the most relevant government schemes, disaster relief policies, crop insurance, cooperative services, or legal assistance.

Question / Problem:
{question}

Rules:
• If the user describes a problem (e.g., crop destroyed by heavy rain, loan issue, cooperative dispute, flood damage), generate search queries for the specific government schemes, relief funds, insurance claims, or legal procedures to solve it.
• Each query must be on a separate line.
• Do not include numbers or bullet points.

Return one search query per line."""

    try:
        resp = _llm([{"role": "user", "content": prompt}], temperature=0, max_tokens=150)
        sub_queries = [
            line.strip().lstrip("•-–*0123456789. ")
            for line in resp.splitlines()
            if line.strip() and len(line.strip()) > 4
        ]
        return sub_queries[:4]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 13. SQLITE CHAT MEMORY (SINGLE SESSION)
# ══════════════════════════════════════════════════════════════════════════════

_db_lock = threading.Lock()
DEFAULT_CHAT_ID = 1


def _get_db():
    """
    Establish thread-safe connection to SQLite database and initialize tables.
    Includes:
    - chats table: stores chat session metadata
    - messages table: stores individual user/assistant conversational messages
    """
    conn = sqlite3.connect(CHAT_DB_PATH, check_same_thread=False)

    # SQL Query: Create chats table if it doesn't exist
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chats(
        chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # SQL Query: Create messages table with foreign key reference
    conn.execute("""
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
    )
    """)

    # Ensure default single session exists
    cur = conn.cursor()
    cur.execute("SELECT chat_id FROM chats WHERE chat_id=?", (DEFAULT_CHAT_ID,))
    if not cur.fetchone():
        # SQL Query: Insert default chat session
        cur.execute("INSERT INTO chats(chat_id, title) VALUES(?, ?)", (DEFAULT_CHAT_ID, "Cooperative & Schemes Session"))

    conn.commit()
    return conn


def save_message(role: str, content: str, chat_id: int = DEFAULT_CHAT_ID):
    """
    SQL Query: Insert a new message (user or assistant) into the database.
    """
    with _db_lock:
        conn = _get_db()
        conn.execute(
            """
            INSERT INTO messages(chat_id, role, content)
            VALUES(?, ?, ?)
            """,
            (chat_id, role, content)
        )
        conn.commit()
        conn.close()


def load_history(limit: int = 20, chat_id: int = DEFAULT_CHAT_ID) -> list[dict]:
    """
    SQL Query: Load recent conversation messages ordered by message ID.
    """
    with _db_lock:
        conn = _get_db()
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE chat_id=?
            ORDER BY id ASC
            """,
            (chat_id,)
        ).fetchall()
        conn.close()
        history = [{"role": r, "content": c} for r, c in rows]
        return history[-limit:]


def clear_history(chat_id: int = DEFAULT_CHAT_ID):
    """
    SQL Query: Delete all messages for the session to reset conversation memory.
    """
    with _db_lock:
        conn = _get_db()
        conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        conn.commit()
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# 14. SEMANTIC ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def semantic_route(question: str) -> str:
    """
    Intelligent semantic router for intent classification.
    Routes greetings, goodbyes, memory queries, and routes all citizen, farmer,
    disaster relief, cooperative, legal, and scheme queries to the RAG pipeline.
    """
    q = question.lower().strip()

    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "namaste", "hii", "hello there", "pranam", "vanakkam"}
    if q in greetings or (len(q.split()) <= 2 and any(g in q for g in greetings)):
        return "GREETING"

    goodbyes = {"bye", "goodbye", "see you", "exit", "quit", "thanks bye", "thank you bye", "alvida"}
    if q in goodbyes:
        return "GOODBYE"

    memory_patterns = [
        "what did i ask", "what was my last question", "what did we discuss",
        "what did we talk about", "summarize our conversation", "conversation summary",
        "remember", "previous question", "earlier question", "last answer"
    ]
    if any(p in q for p in memory_patterns):
        return "MEMORY"

    explicit_out_of_scope = [
        "write python code", "write a python", "write a javascript", "solve math",
        "who won the world cup", "sing a song", "write a poem", "tell me a joke",
        "recipe for cake", "movie review of"
    ]
    if any(p in q for p in explicit_out_of_scope):
        return "OUT_OF_SCOPE"

    # Route all situational queries, farmer questions, crop losses, cooperative disputes,
    # legal queries, subsidies, and welfare schemes to SCHEME
    return "SCHEME"


# ══════════════════════════════════════════════════════════════════════════════
# 15. ANSWER GENERATION & PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """
You are "Sahakar AI" — an AI-powered Multilingual Cooperative Governance, Legal Assistance & Citizen Welfare Advisor developed under the Ministry of Cooperation and National Council for Cooperative Training (NCCT).

Your mission is to empower farmers, cooperative members, Primary Agricultural Credit Societies (PACS), rural stakeholders, entrepreneurs, students, and citizens with accurate, authoritative, and actionable guidance based on the government schemes database.

CORE DOMAINS & CAPABILITIES:

1. Agricultural Crisis & Crop Insurance (PMFBY & Crop Loss):
   - Whenever a farmer reports crop damage, heavy rainfall, flood, drought, or pest attack, explain the government's flagship crop insurance scheme: **Pradhan Mantri Fasal Bima Yojna (PMFBY)**.
   - Explain how PMFBY provides comprehensive financial coverage for crop loss caused by non-preventable natural risks (heavy rainfall, inundation, localized calamity, drought, pest attacks).
   - Detail the low farmer premium share (2% for Kharif crops, 1.5% for Rabi crops, 5% for commercial/horticultural crops).

2. Crop Loans, Financial Protection & Kisan Credit Card (KCC / PACS):
   - Explain the direct link between Crop Loans, Kisan Credit Card (KCC), and Crop Insurance:
     • For **Loanee Farmers** (farmers who took crop loans via Banks or PACS): PMFBY insurance coverage protects their loan liability so that crop loss compensation directly covers/settles the bank loan, preventing debt traps.
     • For **Non-Loanee Farmers**: The insurance claim amount is credited directly to their Aadhaar-linked bank account.
     • In cases of severe calamity, banks and cooperative societies provide restructuring/conversion of short-term crop loans into medium-term loans with interest subvention.

3. State-Level Crop Relief & Assistance Schemes (from Dataset):
   - Highlight applicable state-specific crop loss assistance schemes found in the context (such as *Mukhyamantri Kisan Sahay Yojana*, *Jharkhand Rajya Fasal Rahat Yojana*, *Bihar Rajya Fasal Sahayata Yojna*, and *SDRF/NDRF Natural Calamity Relief*).

4. Cooperative Governance & PACS Guidance:
   - Primary Agricultural Credit Societies (PACS) services (input distribution, credit, crop insurance enrollment, procurement).
   - Cooperative laws, by-laws, membership rights, election rules, and dispute resolution through the District Registrar / Cooperative Court.

RESPONSE STRUCTURE FOR CROP LOSS / FARMER QUERIES:
When responding to crop destruction or farmer financial distress:
1. **Flagship Government Scheme (PMFBY)**: Name the scheme, its objective, and coverage for heavy rainfall/calamity.
2. **Crop Loan & Financial Protection**: Explain how KCC / bank crop loans are protected by insurance and what financial relief is available.
3. **Immediate 72-Hour Action Plan**:
   • Step 1: Intimation within 72 hours via Crop Insurance App, portal (pmfby.gov.in), toll-free 14447, or nearest bank/PACS branch.
   • Step 2: Inform Block Agriculture Officer (BAO) / Revenue Patwari / PACS for Joint Assessment Survey.
   • Step 3: Keep photos of damaged fields, land records (7/12, Khasra), Aadhaar, bank passbook, and sowing certificate ready.
4. **Relevant State Schemes from Dataset**: Present matching state-level compensation/relief schemes with benefits and eligibility.

STRUCTURE & TONE:
• Structure answers cleanly with Markdown tables, bold headers, and numbered steps.
• Empathetic, supportive, professional, and accessible to rural communities.
• Ground all specific scheme parameters (name, category, eligibility, documents, benefits) strictly on the provided dataset context.
"""


def _generate_answer(
    rewritten_question: str,
    sub_queries: list[str],
    context: str,
    chat_history: list[dict],
    found_in_doc: bool,
) -> str:
    """Generate final grounded and situational advisory answer using LLM."""
    answer_plan = ""
    if sub_queries:
        answer_plan = "\n\nANSWER PLAN — ensure you address these aspects:\n" + "\n".join(f"• {q}" for q in sub_queries)

    not_found_hint = ""
    if not found_in_doc:
        not_found_hint = "\n\nNOTE: If specific scheme records are limited in context, combine verified general agricultural/cooperative/PMFBY procedures with available database information to guide the citizen effectively.\n"

    system_content = (
        _SYSTEM_PROMPT
        + answer_plan
        + not_found_hint
        + f"""

══════════════════════════════════════════════════════
CONTEXT FROM SCHEMES & WELFARE DATABASE
══════════════════════════════════════════════════════

{context}
"""
    )

    messages = [{"role": "system", "content": system_content}]

    # Include recent chat history
    for msg in chat_history[-6:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in ["user", "assistant"] and content:
            messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": rewritten_question})

    start = time.time()
    answer = _llm(messages, temperature=0.0, max_tokens=1500)
    print(f"[TIME] LLM Answer: {time.time()-start:.2f}s")
    return answer


# ══════════════════════════════════════════════════════════════════════════════
# 16. LANGGRAPH STATE SCHEMA & NODES
# ══════════════════════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    question: str
    chat_history: List[Dict[str, str]]
    route: str
    rewritten_question: str
    sub_queries: List[str]
    chunks: List[Dict[str, Any]]
    context: str
    answer: str
    found_in_doc: bool


def node_semantic_route(state: RAGState) -> dict:
    question = state["question"]
    chat_history = state.get("chat_history", [])
    pronouns = {"it", "they", "them", "their", "this", "that", "these", "those"}
    question_lower = question.lower()
    has_pronoun = any(p in question_lower.split() for p in pronouns)

    if has_pronoun and chat_history:
        route = "SCHEME"
    else:
        route = semantic_route(question)

    print(f"[LangGraph] Route: {route}")
    return {"route": route}


def node_rewrite_question(state: RAGState) -> dict:
    question = state["question"]
    chat_history = state.get("chat_history", [])
    rewritten = rewrite_question(question, chat_history)
    return {"rewritten_question": rewritten}


def node_decompose_question(state: RAGState) -> dict:
    rewritten = state["rewritten_question"]
    sub_queries = decompose_question(rewritten)
    print(f"[LangGraph] Sub-queries: {sub_queries}")
    return {"sub_queries": sub_queries}


def node_retrieve(state: RAGState) -> dict:
    rewritten = state["rewritten_question"]
    sub_queries = state.get("sub_queries", [])
    plan = sub_queries[:3] if sub_queries else [rewritten]

    evidence = []
    with ThreadPoolExecutor(max_workers=min(len(plan), 3)) as executor:
        future_map = {executor.submit(retrieve, step): step for step in plan}
        for future in as_completed(future_map):
            try:
                res = future.result()
                if res:
                    evidence.extend(res)
            except Exception as e:
                print(f"[LangGraph] Retrieval error: {e}")

    seen = set()
    unique = [c for c in evidence if c["chunk_id"] not in seen and not seen.add(c["chunk_id"])]
    final_chunks = unique[:RERANK_TOP_N * 2]
    found_in_doc = len(final_chunks) > 0
    return {"chunks": final_chunks, "found_in_doc": found_in_doc}


def node_build_context(state: RAGState) -> dict:
    chunks = state["chunks"]
    context = build_context(chunks, max_chars=CONTEXT_CHARS)
    return {"context": context}


def node_generate_answer(state: RAGState) -> dict:
    rewritten = state["rewritten_question"]
    sub_queries = state.get("sub_queries", [])
    context = state["context"]
    chat_history = state.get("chat_history", [])
    found_in_doc = state.get("found_in_doc", True)

    answer = _generate_answer(
        rewritten_question=rewritten,
        sub_queries=sub_queries,
        context=context,
        chat_history=chat_history,
        found_in_doc=found_in_doc,
    )

    if len(answer.split()) > 500:
        print("[LangGraph] Summarizing detailed answer...")
        answer = _llm(
            [
                {
                    "role": "system",
                    "content": "Summarize the answer keeping all emergency steps, deadlines, scheme names, eligibility rules, financial benefits, and documents intact. Keep under 350 words."
                },
                {"role": "user", "content": answer}
            ],
            temperature=0,
            max_tokens=900
        )

    return {"answer": answer}


def node_handle_greeting(state: RAGState) -> dict:
    return {
        "answer": "Namaste! I am **Sahakar AI** — your AI-powered Cooperative Governance, Legal Assistance & Citizen Welfare Advisor.\n\nI can assist you with:\n• **Agricultural & Crop Loss Support**: PMFBY crop insurance claims, heavy rainfall/flood damage assistance, and farming subsidies.\n• **Government Schemes & Subsidies**: Central & State welfare programs, eligibility criteria, and application processes.\n• **Cooperative Governance & PACS**: Primary Agricultural Credit Society services, by-laws, membership rights, and legal guidance.\n• **Financial Literacy & Credit**: KCC loans, interest subvention, and rural banking.\n• **Grievance Redressal**: Legal dispute resolution, filing complaints, and department contacts.\n\nHow may I assist you today?"
    }


def node_handle_goodbye(state: RAGState) -> dict:
    return {
        "answer": "Goodbye! Feel free to return whenever you need guidance on Cooperative Governance, Agricultural Support, or Government Schemes."
    }


def node_handle_memory(state: RAGState) -> dict:
    chat_history = state.get("chat_history", [])
    if not chat_history:
        return {"answer": "We haven't discussed anything yet in this session."}
    summary_lines = [f"{'You' if m['role']=='user' else 'Assistant'}: {m['content'][:150]}" for m in chat_history[-6:]]
    return {"answer": "Here is a summary of our recent conversation:\n\n" + "\n\n".join(summary_lines)}


def node_handle_out_of_scope(state: RAGState) -> dict:
    return {
        "answer": "That topic is outside the scope of Cooperative Governance, Agricultural Support, and Government Schemes. Please ask questions related to agricultural assistance, crop loss, PMFBY, PACS, cooperative by-laws, government subsidies, or citizen welfare programs."
    }


# ══════════════════════════════════════════════════════════════════════════════
# 17. LANGGRAPH STATEGRAPH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def should_route_to_scheme(state: RAGState) -> str:
    route = state["route"]
    if route == "GREETING":
        return "greeting"
    elif route == "GOODBYE":
        return "goodbye"
    elif route == "MEMORY":
        return "memory"
    elif route == "OUT_OF_SCOPE":
        return "out_of_scope"
    return "scheme"


rag_graph = StateGraph(RAGState)
rag_graph.add_node("semantic_route", node_semantic_route)
rag_graph.add_node("rewrite_question", node_rewrite_question)
rag_graph.add_node("decompose_question", node_decompose_question)
rag_graph.add_node("retrieve", node_retrieve)
rag_graph.add_node("build_context", node_build_context)
rag_graph.add_node("generate_answer", node_generate_answer)
rag_graph.add_node("greeting", node_handle_greeting)
rag_graph.add_node("goodbye", node_handle_goodbye)
rag_graph.add_node("memory", node_handle_memory)
rag_graph.add_node("out_of_scope", node_handle_out_of_scope)

rag_graph.set_entry_point("semantic_route")
rag_graph.add_conditional_edges("semantic_route", should_route_to_scheme, {
    "greeting": "greeting",
    "goodbye": "goodbye",
    "memory": "memory",
    "out_of_scope": "out_of_scope",
    "scheme": "rewrite_question"
})
rag_graph.add_edge("rewrite_question", "decompose_question")
rag_graph.add_edge("decompose_question", "retrieve")
rag_graph.add_edge("retrieve", "build_context")
rag_graph.add_edge("build_context", "generate_answer")
rag_graph.add_edge("greeting", END)
rag_graph.add_edge("goodbye", END)
rag_graph.add_edge("memory", END)
rag_graph.add_edge("out_of_scope", END)
rag_graph.add_edge("generate_answer", END)

memory_saver = MemorySaver()
rag_app = rag_graph.compile(checkpointer=memory_saver)


# ══════════════════════════════════════════════════════════════════════════════
# 18. MAIN ROUTE FUNCTION & ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def route_question(question: str, chat_history: Optional[list[dict]] = None) -> str:
    """Main RAG execution entrypoint using LangGraph."""
    if chat_history is None:
        chat_history = load_history()

    initial_state = {
        "question": question,
        "chat_history": chat_history,
        "route": "",
        "rewritten_question": "",
        "sub_queries": [],
        "chunks": [],
        "context": "",
        "answer": "",
        "found_in_doc": False,
    }

    config = {"configurable": {"thread_id": "default_session"}}
    final_state = rag_app.invoke(initial_state, config=config)
    return final_state["answer"]


def _wrap(text: str, width: int = 80) -> str:
    return "\n".join(textwrap.fill(line, width=width) if line else "" for line in text.splitlines())


if __name__ == "__main__":
    print("=" * 60)
    print("Government Schemes RAG Chatbot — type 'exit' to quit")
    print("=" * 60)

    history = load_history()
    if history:
        print(f"Loaded {len(history)} previous messages from SQLite memory.\n")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        answer = route_question(question, history)
        print(f"\nAssistant:\n{_wrap(answer)}\n")

        # Save to SQLite conversation memory
        save_message("user", question)
        save_message("assistant", answer)
        history = load_history()
