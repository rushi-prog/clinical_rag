import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from src.retrieval.embedder import embed_texts
from sklearn.metrics.pairwise import cosine_similarity

texts = [
    "heart attack",
    "myocardial infarction",
    "football match"
]

embeddings = embed_texts(texts)

scores = cosine_similarity(embeddings)

print(scores)