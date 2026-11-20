"""Find the passages most relevant to a question.

Two retrievers are combined (hybrid search):
  * vector search  - finds passages with similar MEANING (embeddings)
  * BM25 search    - finds passages with the same KEYWORDS
Their rankings are merged with Reciprocal Rank Fusion (RRF).

Storage modes
  * local  : one persistent database in ./vividmind_db (your own machine)
  * public : one private in-memory collection PER VISITOR SESSION, deleted
             automatically after a few hours (used on Hugging Face Spaces)
"""
import functools
import re
import time

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "vividmind_db"
COLLECTION = "vividmind"
SESSION_TTL_S = 6 * 60 * 60  # public sessions are wiped after 6 hours


def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


# ------------------------------------------------------------ storage
@functools.lru_cache(maxsize=1)
def _ephemeral_client():
    return chromadb.EphemeralClient()


def cleanup_stale_sessions(client=None):
    """Delete public-session collections older than SESSION_TTL_S (best effort)."""
    client = client or _ephemeral_client()
    try:
        now = time.time()
        for c in client.list_collections():
            name = c if isinstance(c, str) else c.name
            if not name.startswith("vm_"):
                continue
            created = float((client.get_collection(name).metadata or {}).get("created", now))
            if now - created > SESSION_TTL_S:
                client.delete_collection(name)
                invalidate_index(name)
    except Exception:
        pass


def get_collection(session_id=None, public=False, path: str = DB_PATH):
    """local: shared persistent collection. public: private collection per session."""
    if not public:
        client = chromadb.PersistentClient(path=path)
        return client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    client = _ephemeral_client()
    name = f"vm_{session_id}"
    try:
        return client.get_collection(name)
    except Exception:
        cleanup_stale_sessions(client)
        return client.create_collection(
            name, metadata={"hnsw:space": "cosine", "created": time.time()}
        )


def list_docs(collection) -> dict:
    """Return {filename: number_of_chunks}."""
    counts = {}
    for m in collection.get(include=["metadatas"])["metadatas"]:
        counts[m["file"]] = counts.get(m["file"], 0) + 1
    return counts


def delete_doc(collection, filename: str):
    collection.delete(where={"file": filename})
    invalidate_index(collection.name)


# ------------------------------------------------------------ BM25 cache
_BM25_CACHE = {}


def invalidate_index(collection_name: str):
    """Forget cached keyword indexes for a collection (call after add/delete)."""
    for key in [k for k in _BM25_CACHE if k[0] == collection_name]:
        _BM25_CACHE.pop(key, None)


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25(collection, where, files):
    """Build the BM25 index once and reuse it until the documents change."""
    key = (collection.name, tuple(sorted(files)) if files else ())
    if key in _BM25_CACHE:
        return _BM25_CACHE[key]
    data = collection.get(where=where, include=["documents"])
    ids, docs = data["ids"], data["documents"]
    bm25 = BM25Okapi([_tokenize(d) for d in docs]) if ids else None
    if len(_BM25_CACHE) > 64:
        _BM25_CACHE.clear()
    _BM25_CACHE[key] = (ids, bm25)
    return ids, bm25


# ------------------------------------------------------------ overview
OVERVIEW_PATTERN = re.compile(
    r"(summar\w*|overview|main character|protagonist|key points|main points|main idea|tl;?dr"
    r"|what(?:'s| is) (?:this|the|it)[\w\s]{0,20}about|about (?:this|the) (?:file|document|pdf|book|story|paper))",
    re.IGNORECASE,
)


def is_overview_question(question: str) -> bool:
    """Broad questions (summary, main character...) match no single passage."""
    return bool(OVERVIEW_PATTERN.search(question))


def _chunk_no(doc_id: str) -> int:
    m = re.search(r"-c(\d+)$", doc_id)
    return int(m.group(1)) if m else 0


def _where(files):
    if not files:
        return None
    if len(files) == 1:
        return {"file": files[0]}
    return {"file": {"$in": list(files)}}


def overview_passages(collection, files=None, n: int = 10):
    """Return n passages spread evenly across the document(s), in reading order."""
    data = collection.get(where=_where(files), include=["documents", "metadatas"])
    items = sorted(
        zip(data["ids"], data["documents"], data["metadatas"]),
        key=lambda x: (x[2]["file"], x[2]["page"], _chunk_no(x[0])),
    )
    if not items:
        return []
    if len(items) > n:
        idx = np.linspace(0, len(items) - 1, n).astype(int)
        items = [items[i] for i in idx]
    return [
        {"id": i, "text": d, "file": m["file"], "page": m["page"],
         "similarity": 0.0, "overview": True}
        for i, d, m in items
    ]


# ------------------------------------------------------------ search
def _vector_rank(q_emb, collection, n, where):
    n = max(1, min(n, collection.count()))
    res = collection.query(query_embeddings=[q_emb.tolist()], n_results=n, where=where)
    return res["ids"][0]


def _bm25_rank(question, collection, n, where, files):
    ids, bm25 = _get_bm25(collection, where, files)
    if bm25 is None:
        return []
    scores = bm25.get_scores(_tokenize(question))
    order = np.argsort(scores)[::-1][:n]
    return [ids[i] for i in order if scores[i] > 0]


def _rrf(rankings, k: int = 60):
    """Reciprocal Rank Fusion: merge several ranked id lists into one."""
    fused = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def retrieve(question, collection, embedder, k=5, files=None, hybrid=True):
    """Return the top-k passages as dicts: id, text, file, page, similarity."""
    if collection.count() == 0:
        return []

    where = _where(files)
    q_emb = embedder.encode([question], normalize_embeddings=True)[0]
    pool = max(k * 3, 10)

    vec_ids = _vector_rank(q_emb, collection, pool, where)
    if hybrid:
        ranked = _rrf([vec_ids, _bm25_rank(question, collection, pool, where, files)])
    else:
        ranked = vec_ids

    top_ids = ranked[:k]
    if not top_ids:
        return []

    got = collection.get(ids=top_ids, include=["documents", "metadatas", "embeddings"])
    by_id = {
        i: (d, m, e)
        for i, d, m, e in zip(
            got["ids"], got["documents"], got["metadatas"], got["embeddings"]
        )
    }
    hits = []
    for i in top_ids:
        doc, meta, emb = by_id[i]
        hits.append(
            {
                "id": i,
                "text": doc,
                "file": meta["file"],
                "page": meta["page"],
                # embeddings are normalised, so a dot product = cosine similarity
                "similarity": float(np.dot(q_emb, np.asarray(emb))),
            }
        )
    return hits
