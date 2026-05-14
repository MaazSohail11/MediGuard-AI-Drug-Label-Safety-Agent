"""
rag_retriever.py
Queries the ChromaDB vector store for the most relevant FDA drug-label chunks.

Standalone test:
    python src/rag_retriever.py
"""

from pathlib import Path
from typing import List, Dict

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Run:  pip install chromadb sentence-transformers")

PROJECT    = Path(__file__).parent.parent
DB_PATH    = PROJECT / "vector_db" / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level singletons (lazy-loaded)
_client     = None
_collection = None
_embedder   = None


def _load():
    global _client, _collection, _embedder
    if _collection is not None:
        return
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Vector DB not found: {DB_PATH}\n"
            "Run  python src/build_vector_db.py  first."
        )
    _client     = chromadb.PersistentClient(path=str(DB_PATH))
    _collection = _client.get_or_create_collection("drug_labels")
    _embedder   = SentenceTransformer(EMBED_MODEL)


def retrieve_context(query: str, top_k: int = 5) -> List[Dict]:
    """
    Returns top_k relevant chunks as a list of dicts:
        drug_name, brand_name, generic_name, section_name,
        source_id, text, distance
    """
    _load()
    embed  = _embedder.encode([query]).tolist()
    result = _collection.query(
        query_embeddings=embed,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        chunks.append({
            "drug_name":    meta.get("drug_name",    ""),
            "brand_name":   meta.get("brand_name",   ""),
            "generic_name": meta.get("generic_name", ""),
            "section_name": meta.get("section_name", ""),
            "source_id":    meta.get("source_id",    ""),
            "text":         doc,
            "distance":     round(float(dist), 4),
        })
    return chunks


if __name__ == "__main__":
    query = "buspirone MAOI serotonin syndrome contraindication"
    print(f"Test query: {query}\n")
    results = retrieve_context(query, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['drug_name']} — {r['section_name']}  (dist={r['distance']})")
        print(f"     {r['text'][:250]}\n")
