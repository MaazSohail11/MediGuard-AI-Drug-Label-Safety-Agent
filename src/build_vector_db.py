"""
build_vector_db.py
Embeds drug_knowledge.csv safety sections into a local ChromaDB vector store.

Usage:
    python src/build_vector_db.py              # embed up to 60 000 chunks (default)
    python src/build_vector_db.py --max 20000  # smaller / faster demo build
    python src/build_vector_db.py --full       # embed ALL chunks (very slow on CPU)
    python src/build_vector_db.py --reset      # wipe DB before building
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run:  pip install chromadb sentence-transformers tqdm")
    sys.exit(1)

PROJECT    = Path(__file__).parent.parent
PROC_DIR   = PROJECT / "data" / "processed"
DB_PATH    = PROJECT / "vector_db" / "chroma_db"
DB_PATH.mkdir(parents=True, exist_ok=True)

EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE   = 256
COLLECTION   = "drug_labels"

# Only safety-relevant sections go into the vector DB
SAFETY_SECTIONS = {
    "warnings", "warnings_and_cautions", "boxed_warning",
    "contraindications", "drug_interactions", "precautions",
    "adverse_reactions", "overdosage", "pediatric_use",
    "geriatric_use", "pregnancy",
}


def make_document(row: pd.Series) -> str:
    return (
        f"Drug: {row['drug_name']}\n"
        f"Section: {row['section_name']}\n"
        f"Text: {row['section_text']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build MediGuard ChromaDB vector store")
    parser.add_argument("--max",   type=int, default=60_000,
                        help="Max chunks to embed (default: 60000)")
    parser.add_argument("--full",  action="store_true",
                        help="Embed ALL chunks — may take 30+ minutes on CPU")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing collection before building")
    args = parser.parse_args()

    csv_path = PROC_DIR / "drug_knowledge.csv"
    if not csv_path.exists():
        print(f"Not found: {csv_path}")
        print("Run  python src/preprocess_json.py  first.")
        sys.exit(1)

    print("Loading drug_knowledge.csv …")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    print(f"  Total rows in CSV    : {len(df):,}")

    df = df[df["section_name"].isin(SAFETY_SECTIONS)].copy()
    print(f"  Safety-section rows  : {len(df):,}")

    if not args.full:
        df = df.head(args.max)
        print(f"  Limiting to          : {len(df):,} rows  (pass --full to embed all)")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Connecting to ChromaDB …")
    client = chromadb.PersistentClient(path=str(DB_PATH))

    if args.reset:
        try:
            client.delete_collection(COLLECTION)
            print("  Existing collection deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Existing vectors in DB: {collection.count():,}")

    # Build parallel lists for ChromaDB upsert
    docs      = [make_document(row) for _, row in df.iterrows()]
    # IDs must be unique — combine source_id + section to guarantee uniqueness
    ids       = [
        f"{row['source_id']}__{row['section_name']}"
        for _, row in df.iterrows()
    ]
    metadatas = [
        {
            "source_id":    row["source_id"],
            "drug_name":    row["drug_name"],
            "brand_name":   row["brand_name"],
            "generic_name": row["generic_name"],
            "section_name": row["section_name"],
            "route":        row["route"],
        }
        for _, row in df.iterrows()
    ]

    total = len(docs)
    print(f"\nEmbedding and upserting {total:,} chunks (batch_size={BATCH_SIZE}) …")

    for start in tqdm(range(0, total, BATCH_SIZE)):
        end      = min(start + BATCH_SIZE, total)
        b_docs   = docs[start:end]
        b_ids    = ids[start:end]
        b_metas  = metadatas[start:end]
        embeds   = embedder.encode(b_docs, show_progress_bar=False).tolist()
        collection.upsert(
            documents=b_docs,
            embeddings=embeds,
            metadatas=b_metas,
            ids=b_ids,
        )

    print(f"\n[OK] Vector DB ready.  Total vectors: {collection.count():,}")
    print(f"  Path: {DB_PATH}")


if __name__ == "__main__":
    main()
