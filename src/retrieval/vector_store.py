import faiss
import numpy as np
import pickle


class VectorStore:

    def __init__(self, dimension):

        self.dimension = dimension

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.documents = []

    def add_documents(
        self,
        embeddings,
        documents
    ):

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        self.index.add(embeddings)

        self.documents.extend(
            documents
        )

    def search(
        self,
        query_embedding,
        k=3
    ):

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for idx in indices[0]:

            results.append(
                self.documents[idx]
            )

        return results

    def save(self, path):

        faiss.write_index(
            self.index,
            f"{path}.index"
        )

        with open(
            f"{path}.pkl",
            "wb"
        ) as f:

            pickle.dump(
                self.documents,
                f
            )

    def load(self, path):

        self.index = faiss.read_index(
            f"{path}.index"
        )

        with open(
            f"{path}.pkl",
            "rb"
        ) as f:

            self.documents = pickle.load(f)