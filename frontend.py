import requests
import streamlit as st

st.set_page_config(
    page_title="Clinical Trial RAG",
    page_icon="🧬"
)

st.title("🧬 Clinical Trial RAG")

question = st.text_input(
    "Ask a clinical question"
)

if st.button("Ask"):

    if question:

        with st.spinner(
            "Searching..."
        ):

            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={
                    "question": question
                }
            )

            result = response.json()

        st.subheader("Answer")

        st.write(
            result["answer"]
        )

        st.subheader("Sources")

        for source in result["sources"]:

            st.info(source)