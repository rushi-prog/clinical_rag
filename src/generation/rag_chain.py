import os

from groq import Groq

from dotenv import load_dotenv

load_dotenv()


client = Groq(
    api_key=os.getenv(
        "GROQ_API_KEY"
    )
)


def generate_answer(
    query,
    retrieved_docs
):

    context = "\n\n".join(
        retrieved_docs
    )

    prompt = f"""
You are a clinical research assistant.

Use ONLY the provided context.

If the answer is not present,
say so.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return (
        response
        .choices[0]
        .message
        .content
    )