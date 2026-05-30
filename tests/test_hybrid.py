from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever


documents = [

    "Metformin reduces blood sugar levels",

    "Aspirin reduces cardiovascular risk",

    "Football is a popular sport"
]


embeddings = embed_texts(
    documents
)


vector_store = VectorStore(
    dimension=384
)

vector_store.add_documents(
    embeddings,
    documents
)


bm25_store = BM25Store()

bm25_store.build(
    documents
)


retriever = HybridRetriever(
    vector_store,
    bm25_store
)


results = retriever.search(
    "blood sugar treatment"
)


print("\nHybrid Results:\n")

for doc, score in results:

    print(
        round(score, 5),
        doc
    )