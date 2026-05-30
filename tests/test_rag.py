from src.generation.rag_chain import (
    generate_answer
)


query = (
    "How does metformin help diabetic patients?"
)

docs = [

    """
    Metformin reduced HbA1c
    levels in elderly diabetic
    patients.
    """,

    """
    Metformin improved glucose
    control and showed a good
    safety profile.
    """
]

answer = generate_answer(
    query,
    docs
)

print("\nAnswer:\n")

print(answer)