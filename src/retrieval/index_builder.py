from src.ingestion.data_loader import load_trials
from src.ingestion.chunker import chunk_text
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import VectorStore


def build_index():

    df = load_trials(
        "data/clinical_trials.csv"
    )

    chunks = []

    for abstract in df["abstract"]:

        chunks.extend(
            chunk_text(
                abstract,
                chunk_size=50
            )
        )

    embeddings = embed_texts(
        chunks
    )

    store = VectorStore(
        dimension=384
    )

    store.add_documents(
        embeddings,
        chunks
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