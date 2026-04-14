import streamlit as st
import requests

st.title("RAG Q&A System")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        response = requests.get(
            "http://127.0.0.1:8000/ask",
            params={"query": query}
        )

        data = response.json()

        st.write("### Answer")
        st.write(data["answer"])

        st.write("### Confidence")
        st.write(data["confidence"])

        st.write("### Sources")
        for s in data["sources"]:
            st.write(f"{s['source']} (chunk {s['chunk_id']})")