from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import VectorStore


store = VectorStore(
    dimension=384
)

store.load(
    "data/indexes/clinical_trials"
)

query = input(
    "\nEnter query: "
)

query_embedding = embed_texts(
    [query]
)[0]

results = store.search(
    query_embedding,
    k=3
)

print("\nTop Results:\n")

for i, result in enumerate(results, 1):

    print(f"{i}. {result}")