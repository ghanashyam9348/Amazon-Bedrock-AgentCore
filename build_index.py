"""
Run this locally once to pre-compute the FAISS index from lauki_qna.csv.
The saved index is bundled into the Docker image so the runtime loads it
instantly without making any Bedrock embedding calls at startup.

Usage:
    uv run python build_index.py
"""
import csv
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS


def load_faq_csv(path: str) -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            docs.append(Document(page_content=f"Q: {q}\nA: {a}"))
    return docs


if __name__ == "__main__":
    print("Loading FAQ CSV...")
    docs = load_faq_csv("./data/custom_faq.csv")

    emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = splitter.split_documents(docs)

    print(f"Embedding {len(chunks)} chunks locally via fastembed...")
    store = FAISS.from_documents(chunks, emb)

    store.save_local("faiss_index")
    print("Done! FAISS index saved to ./faiss_index/")
    print("Commit the faiss_index/ directory so it gets bundled into the Docker image.")
