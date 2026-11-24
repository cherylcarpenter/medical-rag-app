# Medical RAG Assistant

A small Streamlit app that answers medical questions using Retrieval-Augmented Generation over the Merck Manual of Diagnosis and Therapy. Educational demo only, not medical advice.

## Stack
- Streamlit UI
- ChromaDB (cosine) with all-MiniLM-L6-v2 embeddings
- LLM: openai/gpt-oss-20b via Groq

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-build.txt
python build_index.py            # optional argument: path to the PDF; writes index/
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml  # then add your key
streamlit run app.py
```

## Deploy
Streamlit Community Cloud, Python 3.11. Add the secret `GROQ_API_KEY` in the app settings.

## Notes
- `index/` (`chunks.json.gz` + `embeddings.npy`) is committed prebuilt. At startup the app loads it into an in-memory Chroma collection without re-embedding the chunks.
- `chroma_db/` is no longer used and is gitignored.
- The source PDF is not committed (copyright).
- This is an educational demo, not medical advice. Consult a qualified clinician.
