---
title: VividMind
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# 🧠 VividMind

Chat with your PDFs. Every answer is grounded in your documents and cited by page.

**Live demo:** _add your Hugging Face Space link_

## Features
- **Hybrid retrieval:** BM25 keyword search + vector search, merged with Reciprocal Rank Fusion
- **Page-level citations** with match scores and retrieval latency on every answer
- **Strictness control:** refuses to answer instead of hallucinating when nothing relevant is found
- **Overview mode** for summary-style questions that no single passage can answer
- Multi-document library with a per-document search filter, follow-up questions and streamed answers
- **Private per-visitor sessions** in the hosted version, with bring-your-own Gemini API key
- Evaluation script (Hit@k, MRR) to compare vector-only and hybrid retrieval

## Architecture
```
PDF -> PyMuPDF (per page) -> sentence-aware chunks -> MiniLM embeddings -> ChromaDB
Question -> vector search + BM25 -> RRF fusion -> top-k passages -> Gemini -> cited answer
```

## Tech stack
Python, Streamlit, ChromaDB, Sentence-Transformers (all-MiniLM-L6-v2), rank-bm25, PyMuPDF, Google Gemini API

## Run locally
```
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # then add your Gemini API key and model name
streamlit run app.py
```

## Evaluation
```
python eval/run_eval.py
```
Write your own questions in `eval/questions.json` (see the file for the format).

| Metric | Vector only | Hybrid |
|---|---|---|
| Hit@1 | _fill in_ | _fill in_ |
| MRR | _fill in_ | _fill in_ |

_Test set: N questions over a ___-page document._

## Screenshots
_Add 2-3 screenshots or a short GIF here._

## Limitations
- Scanned PDFs (images only) are not supported; there is no OCR
- Tables and multi-column layouts can be extracted out of order
- Answers depend on the Gemini model and its free-tier rate limits
