import pandas as pd

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever


def evaluate():

    vector_store = VectorStore(
        dimension=384
    )

    vector_store.load(
        "data/indexes/clinical_trials"
    )

    bm25_store = BM25Store()

    bm25_store.load(
        "data/indexes/clinical_trials"
    )

    retriever = HybridRetriever(
        vector_store,
        bm25_store
    )

    eval_df = pd.read_csv(
        "data/evaluation_queries.csv"
    )

    hits = 0

    reciprocal_ranks = []

    for _, row in eval_df.iterrows():

        query = row["query"]

        target = row["relevant_doc"]

        results = retriever.search(
            query,
            k=5
        )

        docs = [
            doc
            for doc, score
            in results
        ]

        found_rank = None

        for rank, doc in enumerate(
            docs,
            start=1
        ):

            if target.lower() in doc.lower():

                found_rank = rank

                break

        if found_rank:

            hits += 1

            reciprocal_ranks.append(
                1 / found_rank
            )

        else:

            reciprocal_ranks.append(
                0
            )

    recall_at_5 = (
        hits / len(eval_df)
    )

    mrr = (
        sum(reciprocal_ranks)
        /
        len(reciprocal_ranks)
    )

    print(
        f"Recall@5: {recall_at_5:.2f}"
    )

    print(
        f"MRR: {mrr:.2f}"
    )


if __name__ == "__main__":

    evaluate()