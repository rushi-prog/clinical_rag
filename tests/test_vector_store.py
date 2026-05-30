import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import VectorStore


documents = [

    "Metformin reduces blood sugar levels.",

    "Aspirin reduces cardiovascular risk.",

    "Football is a popular sport."

]

embeddings = embed_texts(
    documents
)

store = VectorStore(
    dimension=384
)

store.add_documents(
    embeddings,
    documents
)

query = "diabetes medication"

query_embedding = embed_texts(
    [query]
)[0]

results = store.search(
    query_embedding,
    k=2
)

print("\nRetrieved Documents:\n")

for doc in results:

    print(doc)