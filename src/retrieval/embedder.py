from sentence_transformers import SentenceTransformer


# Load once when application starts
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def embed_texts(texts):

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings


if __name__ == "__main__":

    sample_texts = [
        "Metformin reduces blood sugar levels.",
        "Aspirin reduces cardiovascular risk."
    ]

    embeddings = embed_texts(sample_texts)

    print("Shape:", embeddings.shape)

    print("\nFirst 5 values:")
    print(embeddings[0][:5])