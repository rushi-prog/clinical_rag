from src.ingestion.data_loader import load_trials
from src.ingestion.chunker import chunk_text
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import VectorStore


def build_index():

    df = load_trials(
        "data/clinical_trials.csv"
    )

    documents = []
    metadata = []

    for _, row in df.iterrows():

        text = f"""
Title: {row['Study Title']}

Condition: {row['Conditions']}

Intervention: {row['Interventions']}

Summary: {row['Brief Summary']}
"""

        documents.append(text)

        metadata.append(
            {
                "nct_id": row["NCT Number"],
                "title": row["Study Title"]
            }
        )

    chunks = []

    chunk_metadata = []

    for doc, meta in zip(
        documents,
        metadata
    ):

        doc_chunks = chunk_text(
            doc,
            chunk_size=50
        )

        chunks.extend(
            doc_chunks
        )

        chunk_metadata.extend(
            [meta] * len(doc_chunks)
        )

    print(
        f"Created {len(chunks)} chunks"
    )

    embeddings = embed_texts(
        chunks
    )

    store = VectorStore(
        dimension=384
    )

    store.add_documents(
        embeddings,
        chunks,
        chunk_metadata
    )

    store.save(
        "data/indexes/clinical_trials"
    )

    bm25_store = BM25Store()

    bm25_store.build(
        chunks
    )

    bm25_store.save(
        "data/indexes/clinical_trials"
    )

    print(
        f"Indexed {len(chunks)} chunks"
    )
if __name__ == "__main__":

    build_index()