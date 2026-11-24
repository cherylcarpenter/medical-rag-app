try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import gzip
import json
import os

import chromadb
import groq
import numpy as np
import streamlit as st
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import Groq

MODEL = "openai/gpt-oss-20b"
TOP_K = 3
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")
COLLECTION = "merck_manual"

qna_system_message = """You are an AI assistant for healthcare professionals. Your task is to answer medical questions using ONLY the context provided to you, which is retrieved from the Merck Manual of Diagnosis and Therapy.

Guidelines:
- Base your answer strictly on the information in the provided context. Do not use any prior or outside knowledge.
- If the context does not contain enough information to answer the question, respond with: "I don't have enough information in the manual to answer this question."
- Do not fabricate symptoms, treatments, drug names, dosages, or protocols.
- Provide clear, concise, and clinically accurate answers grounded in the retrieved text."""

qna_user_message_template = """Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:"""

groundedness_rater_system_message = """You are an expert at evaluating if an answer is grounded in a given context. \nYour task is to act as an impartial judge and evaluate the groundedness of the answer based on the provided context. \nBegin your evaluation by providing a score for groundedness on a scale of 1 to 5, where 1 means the answer is not at all grounded in the context and 5 means the answer is completely grounded in the context. \nAfter the score, provide a comprehensive explanation for your rating, avoiding any external knowledge.\nFocus solely on whether the information in the answer can be directly inferred from the provided context. Label your score as the Groundedness Score"""

relevance_rater_system_message = """You are an expert at evaluating if an answer is relevant to a given question. \nYour task is to act as an impartial judge and evaluate the relevance of the answer to the provided question. \nBegin your evaluation by providing a score for relevance on a scale of 1 to 5, where 1 means the answer is not at all relevant to the question and 5 means the answer is completely relevant to the question. \nAfter the score, provide a comprehensive explanation for your rating, avoiding any external knowledge.\nFocus solely on whether the answer directly addresses the question asked. Label your score as the Relevance Score"""

rater_user_message_template = "###Question\n{question}\n\n###Context\n{context}\n\n###Answer\n{answer}"

EXAMPLES = [
    "What is the protocol for managing sepsis in a critical care unit?",
    "What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?",
    "What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?",
    "What treatments are recommended for a person who has sustained a physical injury to brain tissue, resulting in temporary or permanent impairment of brain function?",
    "What are the necessary precautions and treatment steps for a person who has fractured their leg during a hiking trip, and what should be considered for their care and recovery?",
    "What treatment options are available for managing hypertension?",
]


@st.cache_resource(show_spinner="Loading the Merck Manual index…")
def get_collection():
    with gzip.open(os.path.join(INDEX_DIR, "chunks.json.gz"), "rt", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        COLLECTION,
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    step = 1000
    for i in range(0, len(chunks), step):
        j = i + step
        part = chunks[i:j]
        collection.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[{"page": c["page"]} for c in part],
            embeddings=embeddings[i:j].tolist(),
        )
    return collection


@st.cache_resource
def get_client():
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    key = key or os.environ.get("GROQ_API_KEY")
    if not key:
        st.error(
            "GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml (or the "
            "Streamlit Cloud secrets) or set it as an environment variable."
        )
        st.stop()
    return Groq(api_key=key)


def retrieve(q):
    res = get_collection().query(query_texts=[q], n_results=TOP_K)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [(d, m.get("page")) for d, m in zip(docs, metas)]


def chat(system, user, max_tokens):
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_completion_tokens=max_tokens,
        reasoning_effort="low",
        include_reasoning=False,
    )
    return (resp.choices[0].message.content or "").strip()


st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺")
st.title("🩺 Medical Q&A — Merck Manual RAG")
st.caption(
    "Retrieval-Augmented Generation over the Merck Manual. Embeddings: all-MiniLM-L6-v2 + "
    "Chroma. LLM: openai/gpt-oss-20b via Groq."
)
st.info(
    "⏳ **Heads up:** this is a demo running on a free server that goes to sleep when idle. "
    "Your first question may take 30–60 seconds while it wakes up and loads the manual's index — "
    "after that, answers come back in a few seconds."
)
st.warning(
    "Educational demo only — not medical advice. Answers come solely from retrieved "
    "passages of the Merck Manual and may be incomplete. Always consult a qualified clinician."
)

example = st.selectbox(
    "Try an example question",
    [""] + EXAMPLES,
    format_func=lambda x: "— choose an example —" if x == "" else x,
)
question = st.text_area("Your question", value=example, key=f"question_{example}")
score = st.checkbox(
    "Score this answer (groundedness & relevance)",
    value=False,
    help="Makes 2 extra LLM calls.",
)

if st.button("Ask", type="primary"):
    question = question.strip()
    if not question:
        st.info("Please enter a question.")
    else:
        try:
            with st.spinner("Searching the manual…"):
                results = retrieve(question)
            context = "\n\n---\n\n".join(t for t, _ in results)

            with st.spinner("Generating answer…"):
                answer = chat(
                    qna_system_message,
                    qna_user_message_template.format(context=context, question=question),
                    1024,
                )
            st.subheader("Answer")
            st.markdown(answer)

            with st.expander("Sources (top 3 retrieved passages)"):
                for text, page in results:
                    st.markdown(f"**Page {page}**")
                    snippet = text if len(text) <= 600 else text[:600] + "…"
                    st.caption(snippet)

            if score:
                rater_user = rater_user_message_template.format(
                    question=question, context=context, answer=answer
                )
                with st.spinner("Scoring answer…"):
                    g = chat(groundedness_rater_system_message, rater_user, 512)
                    r = chat(relevance_rater_system_message, rater_user, 512)
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Groundedness")
                    st.markdown(g)
                with c2:
                    st.subheader("Relevance")
                    st.markdown(r)
        except groq.APIError as e:
            st.error(f"LLM request failed: {e}")
        except Exception as e:
            st.error(f"LLM request failed: {e}")
