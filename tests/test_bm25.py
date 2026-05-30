from src.retrieval.bm25_store import BM25Store


documents = [

    "Metformin reduces blood sugar levels",

    "Aspirin reduces cardiovascular risk",

    "Football is a popular sport"
]

store = BM25Store()

store.build(
    documents
)

results = store.search(
    "blood sugar",
    k=2
)

print("\nBM25 Results:\n")

for doc, score in results:

    print(
        f"{score:.2f}",
        doc
    )