from src.retrieval.reranker import Reranker


query = (
    "Can metformin reduce blood sugar?"
)

documents = [

    "Blood sugar is an important health metric.",

    "Metformin reduces blood sugar levels.",

    "Football is a popular sport."
]

reranker = Reranker()

results = reranker.rerank(
    query,
    documents
)

print("\nReranked Results:\n")

for doc, score in results:

    print(
        round(float(score), 3),
        doc
    )