def chunk_text(text, chunk_size=50):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    sample = """
    Metformin reduced HbA1c levels in elderly diabetic patients.
    """

    print(
        chunk_text(sample, chunk_size=5)
    )