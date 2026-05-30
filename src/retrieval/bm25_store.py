from rank_bm25 import BM25Okapi
import pickle

class BM25Store:

    def __init__(self):

        self.documents = []

        self.tokenized_docs = []

        self.bm25 = None

    def build(self, documents):

        self.documents = documents

        self.tokenized_docs = [
            doc.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

    def search(
        self,
        query,
        k=3
    ):

        query_tokens = (
            query.lower()
            .split()
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked = sorted(
            zip(
                self.documents,
                scores
            ),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:k]

    def save(self, path):

        with open(
            f"{path}.bm25",
            "wb"
        ) as f:

            pickle.dump(
            {
                "documents": self.documents,
                "tokenized_docs": self.tokenized_docs
            },
            f
        )
    
    def load(self, path):

        with open(
        f"{path}.bm25",
        "rb"
        ) as f:

            data = pickle.load(f)

        self.documents = data["documents"]

        self.tokenized_docs = data["tokenized_docs"]

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )