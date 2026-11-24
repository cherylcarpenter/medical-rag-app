"""Build the compact vector index from the Merck Manual PDF (run locally only).

Writes index/embeddings.npy and index/chunks.json.gz, which are committed.
The app loads them into an in-memory Chroma collection at startup.
"""
import gzip
import json
import os
import sys

import numpy as np
import pymupdf
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "/Users/cherylcarpenter/Documents/PythonPrj/NLP/project/medical_diagnosis_manual.pdf"
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")
BATCH = 256


def size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    doc = pymupdf.open(pdf_path)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=200
    )

    records = []
    page_count = 0
    for i, page in enumerate(doc):
        page_no = i + 1
        page_count += 1
        text = page.get_text()
        for j, chunk in enumerate(splitter.split_text(text)):
            if not chunk.strip():
                continue
            records.append({"id": f"p{page_no}-c{j}", "page": page_no, "text": chunk})

    ef = DefaultEmbeddingFunction()
    vectors = []
    next_report = 1000
    for s in range(0, len(records), BATCH):
        batch = [r["text"] for r in records[s : s + BATCH]]
        vectors.extend(np.asarray(v, dtype=np.float32) for v in ef(batch))
        if len(vectors) >= next_report:
            print(f"Embedded {len(vectors)}/{len(records)} chunks")
            next_report += 1000
    embeddings = np.stack(vectors).astype(np.float32)

    os.makedirs(INDEX_DIR, exist_ok=True)
    emb_path = os.path.join(INDEX_DIR, "embeddings.npy")
    chunks_path = os.path.join(INDEX_DIR, "chunks.json.gz")
    np.save(emb_path, embeddings)
    with gzip.open(chunks_path, "wt", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Pages: {page_count}")
    print(f"Chunks: {len(records)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"embeddings.npy: {size_mb(emb_path):.1f} MB")
    print(f"chunks.json.gz: {size_mb(chunks_path):.1f} MB")


if __name__ == "__main__":
    main()
