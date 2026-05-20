"""Local ChromaDB-backed RAG tools."""

import os
import time
import asyncio
import threading
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)

CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    str(Path(__file__).resolve().parents[1] / "data" / "chroma"),
)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "hugobank_data")

_EMBED_MODEL = "gemini-embedding-2"
_EMBED_DIMS = 1024
_EMBED_BATCH_SIZE = 100
_EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "6"))
_EMBED_RETRY_BASE_SECONDS = float(os.getenv("EMBED_RETRY_BASE_SECONDS", "10"))
_EMBED_REQUEST_INTERVAL_SECONDS = float(
    os.getenv("EMBED_REQUEST_INTERVAL_SECONDS", "0")
)

_embedding_cache: dict[str, list[float]] = {}
MAX_CACHE_SIZE = 200
_embed_request_lock = threading.Lock()
_next_embed_request_at = 0.0

COMMON_QUERIES = [
    "HugoBank digital banking Pakistan",
    "Savings Pots Money Pots savings goals",
    "EWA Earned Wage Access salary advance",
    "Wealthcare holistic financial wellbeing",
    "HugoBank waitlist sign up join",
    "HugoBank app launch iOS Android",
    "virtual card physical card debit credit",
    "fraud awareness security",
    "privacy policy terms of use",
    "careers HugoTribe team",
    "State Bank Pakistan in-principle approval IPA",
    "contact customer support",
]


def _google_api_key() -> Optional[str]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    prefix = "GOOGLE_API_KEY="
    return api_key[len(prefix):] if api_key.startswith(prefix) else api_key


_genai_client = genai.Client(api_key=_google_api_key())

def _get_client() -> chromadb.api.ClientAPI:
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

_client: Optional[chromadb.api.ClientAPI] = None
_collection = None

def get_collection():
    """Return the Chroma collection, creating client/collection on first use."""
    global _client, _collection
    if _collection is not None:
        return _collection
    if _client is None:
        _client = _get_client()
    _collection = _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _extract_vector(resp) -> list[float]:
    embeddings = resp.embeddings or []
    if not embeddings or not embeddings[0].values:
        raise RuntimeError("Gemini embedding response did not include vector values")
    return list(embeddings[0].values)


def _extract_vectors(resp) -> list[list[float]]:
    vectors: list[list[float]] = []
    for item in resp.embeddings or []:
        if not item.values:
            raise RuntimeError("Gemini embedding response contained an empty vector")
        vectors.append(list(item.values))
    return vectors


def _batched(items: list[str], batch_size: int = _EMBED_BATCH_SIZE):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "resource_exhausted" in message
        or "quota exceeded" in message
        or "429" in message
    )


def _sync_embed_gate():
    global _next_embed_request_at
    with _embed_request_lock:
        now = time.monotonic()
        wait_seconds = max(0.0, _next_embed_request_at - now)
        if _EMBED_REQUEST_INTERVAL_SECONDS > 0:
            _next_embed_request_at = max(_next_embed_request_at, now) + _EMBED_REQUEST_INTERVAL_SECONDS
    if wait_seconds > 0:
        time.sleep(wait_seconds)


async def _async_embed_gate():
    global _next_embed_request_at
    with _embed_request_lock:
        now = time.monotonic()
        wait_seconds = max(0.0, _next_embed_request_at - now)
        if _EMBED_REQUEST_INTERVAL_SECONDS > 0:
            _next_embed_request_at = max(_next_embed_request_at, now) + _EMBED_REQUEST_INTERVAL_SECONDS
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)


def _sync_embed_request(contents):
    for attempt in range(_EMBED_MAX_RETRIES + 1):
        try:
            _sync_embed_gate()
            return _genai_client.models.embed_content(
                model=_EMBED_MODEL,
                contents=contents,
                config={"output_dimensionality": _EMBED_DIMS},
            )
        except Exception as error:
            if not _is_quota_error(error) or attempt >= _EMBED_MAX_RETRIES:
                raise
            delay = _EMBED_RETRY_BASE_SECONDS * (2 ** attempt)
            print(
                f"⚠️ Gemini embed quota hit; retrying in {delay:.0f}s "
                f"(attempt {attempt + 1}/{_EMBED_MAX_RETRIES})"
            )
            time.sleep(delay)


async def _async_embed_request(contents):
    for attempt in range(_EMBED_MAX_RETRIES + 1):
        try:
            await _async_embed_gate()
            return await _genai_client.aio.models.embed_content(
                model=_EMBED_MODEL,
                contents=contents,
                config={"output_dimensionality": _EMBED_DIMS},
            )
        except Exception as error:
            if not _is_quota_error(error) or attempt >= _EMBED_MAX_RETRIES:
                raise
            delay = _EMBED_RETRY_BASE_SECONDS * (2 ** attempt)
            print(
                f"⚠️ Gemini embed quota hit; retrying in {delay:.0f}s "
                f"(attempt {attempt + 1}/{_EMBED_MAX_RETRIES})"
            )
            await asyncio.sleep(delay)


async def _async_embed_many(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch in _batched(texts):
        resp = await _async_embed_request(batch)
        vectors.extend(_extract_vectors(resp))
    return vectors

async def _async_embed(query: str) -> list[float]:
    cache_key = query.strip().lower()
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    resp = await _async_embed_request(query)
    vector = _extract_vector(resp)

    if len(_embedding_cache) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]

    _embedding_cache[cache_key] = vector
    return vector

def _sync_embed(query: str) -> list[float]:
    cache_key = query.strip().lower()
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    resp = _sync_embed_request(query)
    vector = _extract_vector(resp)
    if len(_embedding_cache) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]
    _embedding_cache[cache_key] = vector
    return vector


def sync_embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    uncached_texts: list[str] = []
    results_by_index: list[Optional[list[float]]] = [None] * len(texts)

    for index, text in enumerate(texts):
        cache_key = text.strip().lower()
        cached = _embedding_cache.get(cache_key)
        if cached is not None:
            results_by_index[index] = cached
            continue
        uncached_texts.append(text)

    if uncached_texts:
        generated_vectors: list[list[float]] = []
        for batch in _batched(uncached_texts):
            resp = _sync_embed_request(batch)
            generated_vectors.extend(_extract_vectors(resp))

        if len(generated_vectors) != len(uncached_texts):
            raise RuntimeError(
                "Gemini embedding response count did not match the requested texts"
            )
        uncached_iter = iter(zip(uncached_texts, generated_vectors))
        for index, text in enumerate(texts):
            if results_by_index[index] is not None:
                continue
            original_text, vector = next(uncached_iter)
            cache_key = original_text.strip().lower()
            if len(_embedding_cache) >= MAX_CACHE_SIZE:
                oldest_key = next(iter(_embedding_cache))
                del _embedding_cache[oldest_key]
            _embedding_cache[cache_key] = vector
            results_by_index[index] = vector

    if any(vector is None for vector in results_by_index):
        raise RuntimeError("Failed to resolve embeddings for one or more texts")
    return [vector for vector in results_by_index if vector is not None]

def _chroma_query(vector: list[float], top_k: int) -> dict:
    collection = get_collection()
    return collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

def _iter_matches(results: dict):
    ids_batch = results.get("ids") or [[]]
    docs_batch = results.get("documents") or [[]]
    metas_batch = results.get("metadatas") or [[]]
    dists_batch = results.get("distances") or [[]]
    if not ids_batch or not ids_batch[0]:
        return
    docs = docs_batch[0] if docs_batch else []
    metas = metas_batch[0] if metas_batch else []
    dists = dists_batch[0] if dists_batch else []
    for i in range(len(ids_batch[0])):
        doc = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else 1.0
        score = 1.0 - float(dist)
        yield doc or (meta or {}).get("text", ""), (meta or {}), score

async def prewarm_embeddings():
    try:
        vectors = await _async_embed_many(COMMON_QUERIES)
        warm_vector: Optional[list[float]] = None
        for text, vector in zip(COMMON_QUERIES, vectors):
            _embedding_cache[text.strip().lower()] = vector
            if warm_vector is None:
                warm_vector = vector
        print(f"✅ Pre-warmed {len(COMMON_QUERIES)} embedding cache entries")

        t0 = time.perf_counter()
        collection = await asyncio.to_thread(get_collection)
        count = await asyncio.to_thread(collection.count)
        if warm_vector is not None and count > 0:
            await asyncio.to_thread(_chroma_query, warm_vector, 1)
        warm_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"🔥 [RAG] Chroma collection '{CHROMA_COLLECTION}' ready "
            f"(docs={count}, warmup_ms={warm_ms:.0f}, path={CHROMA_DB_PATH})"
        )
    except Exception as e:
        print(f"⚠️ Embedding pre-warm failed (non-fatal): {e}")

def retrieve_context(query: str, top_k: int = 3, min_score: float = 0.35) -> str:
    try:
        query_vector = _sync_embed(query)
        results = _chroma_query(query_vector, top_k)

        context_chunks = []
        seen_content = set()

        for text_content, metadata, score in _iter_matches(results):
            if score < min_score:
                continue
            category = metadata.get("category", "General")
            subcategory = metadata.get("subcategory", "")

            content_hash = hash(text_content[:100])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            if text_content:
                context_chunks.append(
                    f"[{category} - {subcategory}]\n{text_content}"
                )

        return "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

async def search_knowledge_base(query: str, top_k: int = 10, min_score: float = 0.28) -> dict:
    try:
        start_time = time.time()
        print(f"\n🔍 RAG SEARCH: '{query}'")

        vector = await _async_embed(query)
        embed_ms = (time.time() - start_time) * 1000

        results = await asyncio.to_thread(_chroma_query, vector, top_k)
        elapsed = time.time() - start_time
        print(f"🔍 RAG SEARCH completed in {elapsed:.2f}s (embed: {embed_ms:.0f}ms)")

        matches = list(_iter_matches(results))
        if not matches:
            return {"success": False, "message": "No relevant information found in the knowledge base.", "results": [], "context": ""}

        relevant_matches = [m for m in matches if m[2] >= min_score]

        if not relevant_matches:
            print(f"⚠️ RAG: All {len(matches)} results below threshold ({min_score})")
            return {"success": False, "message": "No relevant information found in the knowledge base.", "results": [], "context": ""}

        context_chunks = []
        knowledge_results = []
        total_chars = 0
        MAX_TOTAL_CHARS = 5500
        seen_content = set()

        for text_content, metadata, score in relevant_matches:
            if total_chars >= MAX_TOTAL_CHARS:
                break

            content_hash = hash(text_content[:100])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            category = metadata.get("category", "General")
            subcategory = metadata.get("subcategory", "")
            
            knowledge_results.append({
                "text": text_content,
                "category": category,
                "subcategory": subcategory,
                "score": score
            })

            if text_content:
                remaining = MAX_TOTAL_CHARS - total_chars
                text = text_content[:remaining]
                total_chars += len(text)
                context_chunks.append(f"[{category}]\n{text}")

        combined_context = "\n---\n".join(context_chunks)

        print(f"✅ RAG: Found {len(context_chunks)} results ({total_chars} chars) in {elapsed:.2f}s")

        return {
            "success": True,
            "message": "Found relevant information.",
            "context": combined_context,
            "num_results": len(knowledge_results)
        }

    except Exception as e:
        print(f"⚠️ search_knowledge_base error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "An error occurred while searching the knowledge base.",
            "context": ""
        }
