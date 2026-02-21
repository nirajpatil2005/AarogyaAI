<p align="center">
  <h1 align="center">🏥 MEDORBY</h1>
  <p align="center"><strong>Privacy-First Medical AI with LLM Council, RAG, and Federated Learning</strong></p>
  <p align="center">
    <em>Your medical data never leaves your device. Period.</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Next.js-16-black?logo=next.js" />
  <img src="https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
  <img src="https://img.shields.io/badge/Privacy-First-brightgreen?logo=shield" />
</p>

---

## What is MEDORBY?

MEDORBY is a **privacy-first medical AI assistant** that uses a multi-model **LLM Council** to provide clinical reasoning about symptoms. Unlike traditional health chatbots, MEDORBY:

- **Never sends your raw medical data to the cloud** — all PII is stripped client-side
- **Runs a local ML classifier** on your device before any API calls
- **Retrieves relevant medical knowledge** from a FAISS-indexed knowledge base (RAG)
- **Lets you upload medical reports** that stay on your machine
- **Improves over time** through Federated Learning without sharing your data

> ⚠️ **Disclaimer**: MEDORBY is a research project and NOT a replacement for professional medical advice. Always consult a qualified healthcare provider for medical decisions.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **LLM Council** | 3 AI models deliberate in parallel → peer review → chairman synthesis |
| 🔒 **Zero PII Exposure** | Names, dates, contacts stripped in-browser before any network call |
| 🔍 **RAG Knowledge Base** | FAISS-indexed AHA/ACC medical guidelines for context-aware responses |
| 🩺 **Local ML Classifier** | On-device symptom classification — no cloud calls needed |
| 📄 **Medical Report Upload** | Upload PDF/DOCX/TXT reports, indexed locally for personalised context |
| 🏥 **Hospital DB** | SQLite edge storage for anonymised consultation records |
| 🤝 **Federated Learning** | Your feedback improves the global model via DP-noised gradients |
| 🔐 **Encrypted Storage** | Local medical history encrypted with AES-256-GCM |
| ⚡ **Real-Time Streaming** | Server-Sent Events for live council progress updates |
| 🚨 **Red-Flag Engine** | Instant emergency detection — no LLM needed for critical cases |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 16 (App Router), React 19, Vanilla CSS, Lucide Icons |
| **Backend** | FastAPI + Uvicorn (Python 3.12+) |
| **LLM Council** | Groq Cloud — Llama 3.3 70B, Llama 3.1 8B, Qwen3 32B |
| **RAG Engine** | FAISS + TF-IDF Vectorizer (scikit-learn) |
| **ML Classifier** | TF-IDF + Logistic Regression (scikit-learn) |
| **Local Storage** | IndexedDB + AES-256-GCM (client), SQLite (server) |
| **Privacy** | Client-side PII sanitizer + Differential Privacy |
| **Federated Learning** | FedAvg aggregator + DP noise injection |
| **Streaming** | Server-Sent Events (SSE) |

---

## 📂 Project Structure

```
MVP/
├── README.md                          ← You are here
├── FEATURES.md                        ← Detailed feature documentation
├── ARCHITECTURE.md                    ← System architecture & implementation
├── .gitignore
├── start_backend.ps1                  ← One-click backend launcher
├── start_frontend.ps1                 ← One-click frontend launcher
│
├── backend/
│   ├── main.py                        ← FastAPI entry point (all API routes)
│   ├── config.py                      ← Pydantic settings (.env loader)
│   ├── requirements.txt               ← Python dependencies
│   ├── .env.example                   ← Environment variable template
│   │
│   ├── core/                          ← Core utilities
│   │   └── red_flag_engine.py         ← Deterministic emergency detection
│   │
│   ├── council/                       ← LLM Council module
│   │   ├── groq_client.py             ← Groq API client + model config
│   │   └── orchestrator.py            ← 3-stage orchestration logic
│   │
│   ├── ml/                            ← Machine Learning module
│   │   └── symptom_classifier.py      ← Local symptom classifier (TF-IDF + LR)
│   │
│   ├── rag/                           ← RAG module
│   │   ├── engine.py                  ← FAISS-based retrieval engine
│   │   └── report_processor.py        ← PDF/DOCX/TXT report processor
│   │
│   ├── storage/                       ← Storage module
│   │   └── hospital_db.py             ← SQLite hospital database
│   │
│   ├── federated/                     ← Federated Learning module
│   │   ├── aggregator.py              ← FedAvg gradient aggregator
│   │   └── dp_privacy.py              ← Differential privacy utilities
│   │
│   ├── knowledge_base/                ← Curated medical knowledge
│   │   └── heart_knowledge.json       ← 15 AHA/ACC/WHO heart documents
│   │
│   ├── data/                          ← Runtime data (auto-created, gitignored)
│   │   └── hospital_local.db
│   └── user_reports/                  ← Uploaded reports (gitignored)
│
└── frontend/
    ├── package.json
    ├── next.config.ts
    ├── .env.local                     ← Backend URL config
    │
    └── src/
        ├── app/
        │   ├── layout.tsx             ← Root layout + fonts
        │   ├── page.tsx               ← Landing page + consultation UI
        │   ├── globals.css            ← Full design system (vanilla CSS)
        │   └── favicon.ico
        ├── components/
        │   └── ChatInterface.tsx      ← Main consultation interface
        └── lib/
            ├── sanitizer.ts           ← Client-side PII sanitizer
            ├── local_learning.ts      ← Federated learning client
            └── storage.ts             ← Encrypted IndexedDB storage
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** with `pip`
- **Node.js 18+** with `npm`
- **Groq API Key** — Get one free at [console.groq.com](https://console.groq.com)

### 1. Clone & Setup Backend

```bash
# Navigate to project
cd MVP/backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### 2. Start Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
[HospitalDB] Database initialized.
[SymptomClassifier] Trained on 64 examples, 5 categories.
[RAG] Indexed 16 documents.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 3. Setup & Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Open the App

Visit **[http://localhost:3000](http://localhost:3000)** in your browser. 🎉

### One-Click Start (Windows)

You can also use the PowerShell launcher scripts:

```powershell
.\start_backend.ps1    # Launches backend in a new window
.\start_frontend.ps1   # Launches frontend in a new window
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + model info |
| `POST` | `/api/triage` | Deterministic red-flag check (no LLM) |
| `POST` | `/api/council` | Full council deliberation (SSE streaming) |
| `POST` | `/api/classify` | Local ML symptom classification |
| `POST` | `/api/rag/retrieve` | RAG knowledge base retrieval |
| `GET` | `/api/rag/stats` | RAG engine statistics |
| `POST` | `/api/reports/upload` | Upload medical report (PDF/DOCX/TXT) |
| `GET` | `/api/reports` | List uploaded reports |
| `DELETE` | `/api/reports/{id}` | Delete a report |
| `POST` | `/api/reports/analyze/{id}` | AI analysis of a report |
| `GET` | `/api/hospital/stats` | Hospital DB statistics |
| `GET` | `/api/hospital/records` | Anonymised consultation records |
| `POST` | `/api/federated/update` | Submit DP-noised model update |
| `GET` | `/api/federated/adapter` | Download latest global adapter |
| `GET` | `/api/federated/status` | Aggregator status |

---

## 🔒 Privacy Guarantees

| Guarantee | How |
|-----------|-----|
| ✅ **No PII sent to cloud** | Client-side regex sanitizer strips names, emails, phone numbers, dates of birth before any API call |
| ✅ **Emergency detection is local** | Red-Flag Engine runs deterministic rules — zero cloud dependency |
| ✅ **ML classification is local** | TF-IDF + Logistic Regression model runs on your machine |
| ✅ **Reports stay on your device** | Uploaded PDFs/DOCX never leave `user_reports/` on your filesystem |
| ✅ **Encrypted local storage** | IndexedDB data encrypted with AES-256-GCM; only you hold the key |
| ✅ **Federated learning is private** | Gradient clipping + Gaussian DP noise before any update leaves the device |
| ✅ **Anonymised DB records** | Hospital DB stores symptom hashes, never raw text |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  Built with ❤️ for privacy-respecting medical AI
</p>
