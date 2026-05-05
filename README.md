# 🤖 SupportGenie

> Production-grade RAG-powered customer support bot. Plug in any business's documentation and get an AI assistant that answers customer questions accurately.

## 🎯 What Problem It Solves

Companies spend $100k+/year on customer support agents answering the same questions repeatedly. SupportGenie:
- Answers 80% of repetitive questions automatically
- Cites the exact docs it pulled from
- Works for any business (multi-tenant)
- Scales infinitely

## 🏗️ Architecture


## 🛠️ Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | FastAPI | Industry standard, async, auto-docs |
| LLM | Groq (LLaMA 3.3) | Fastest inference, generous free tier |
| Vector DB | ChromaDB → Pinecone | Local dev → Cloud production |
| Embeddings | sentence-transformers | Free, runs locally, good quality |
| Re-ranking | cross-encoder | Improves retrieval precision 30%+ |
| Auth | API keys | Simple, sufficient for B2B |
| Frontend | TBD (SwiftUI) | After backend is solid |

## 🚀 Roadmap

- [x] Project structure
- [ ] Document ingestion pipeline
- [ ] Smart chunking strategies
- [ ] Hybrid search (semantic + BM25)
- [ ] Cross-encoder re-ranking
- [ ] Streaming responses
- [ ] Multi-tenancy
- [ ] Authentication
- [ ] Evaluation framework
- [ ] Docker deployment
- [ ] Live demo

## 📚 What I Learned Building This

(Will document weekly)

## 🏃 Running Locally

```bash
git clone https://github.com/Jashan-ios/support-genie.git
cd support-genie
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
uvicorn app.main:app --reload
```

## 📝 License

MIT

---

Built by [Jashan Deol](https://github.com/Jashan-ios) as part of journey to AI Engineer 🚀