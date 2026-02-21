# 🌟 MEDORBY — Features & Specialties

> A deep dive into what makes MEDORBY unique and how each feature is implemented.

---

## Table of Contents

1. [LLM Council (Multi-Model Deliberation)](#1-llm-council-multi-model-deliberation)
2. [Client-Side PII Sanitization](#2-client-side-pii-sanitization)
3. [Red-Flag Engine (Emergency Detection)](#3-red-flag-engine-emergency-detection)
4. [Local ML Symptom Classifier](#4-local-ml-symptom-classifier)
5. [RAG Knowledge Base (FAISS + TF-IDF)](#5-rag-knowledge-base-faiss--tf-idf)
6. [Medical Report Upload & Analysis](#6-medical-report-upload--analysis)
7. [Hospital Database (Local Edge Storage)](#7-hospital-database-local-edge-storage)
8. [Federated Learning with Differential Privacy](#8-federated-learning-with-differential-privacy)
9. [Encrypted Local Storage (IndexedDB + AES-256)](#9-encrypted-local-storage-indexeddb--aes-256)
10. [Real-Time SSE Streaming](#10-real-time-sse-streaming)

---

## 1. LLM Council (Multi-Model Deliberation)

### What Makes It Special

Instead of relying on a single AI model, MEDORBY employs a **council of 3 diverse AI models** that deliberate in a structured 3-stage protocol. This mimics how real medical boards operate — multiple physicians independently analyse a case, discuss their findings, and reach a consensus.

### Council Members

| Role | Model | Purpose |
|------|-------|---------|
| **Member A** | Llama 3.3 70B | Large-scale general clinical reasoning |
| **Member B** | Llama 3.1 8B Instant | Ultra-fast, diverse perspective from a smaller model |
| **Member C** | Qwen3 32B | Strong cross-architecture reasoning (non-Llama diversity) |
| **Reviewer** | Llama 3.1 8B | Fast peer ranking of the three divergence outputs |
| **Chairman** | Llama 3.3 70B | Final synthesis into a unified clinical answer |

### 3-Stage Protocol

```
Stage 1: DIVERGENCE (Parallel)
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Member A    │  │  Member B    │  │  Member C    │
│  Llama 70B   │  │  Llama 8B    │  │  Qwen3 32B   │
│              │  │              │  │              │
│ Independent  │  │ Independent  │  │ Independent  │
│  reasoning   │  │  reasoning   │  │  reasoning   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
Stage 2: CONVERGENCE (Peer Review)
┌──────────────────────────────────────────────────┐
│  Reviewer (Llama 8B)                             │
│  Ranks A, B, C by clinical accuracy              │
│  Output: ranking + brief reasoning               │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
Stage 3: SYNTHESIS (Chairman)
┌──────────────────────────────────────────────────┐
│  Chairman (Llama 70B)                            │
│  Merges top-ranked response into final answer    │
│  Output: differentials, next steps, confidence   │
└──────────────────────────────────────────────────┘
```

### How It's Implemented

- **Module**: `backend/council/`
- **Files**: `groq_client.py` (API client + model config), `orchestrator.py` (3-stage logic)
- **API**: All three stages run via `asyncio.gather` for parallel fan-out
- **Streaming**: Each stage emits SSE events so the frontend shows real-time progress
- **Error Handling**: If any model fails, fallback JSON is used so the pipeline continues

---

## 2. Client-Side PII Sanitization

### What Makes It Special

All personally identifiable information (PII) is **stripped in the browser** before any data touches the network. The backend never receives raw patient names, dates of birth, phone numbers, or email addresses.

### What Gets Sanitised

| PII Type | Pattern | Replacement |
|----------|---------|-------------|
| Names | Capitalised words near patient/name | `[REDACTED_NAME]` |
| Email | Standard email regex | `[REDACTED_EMAIL]` |
| Phone | Various phone number formats | `[REDACTED_PHONE]` |
| Dates | DD/MM/YYYY, MM-DD-YY, etc. | `[REDACTED_DATE]` |
| SSN/ID numbers | Numeric ID patterns | `[REDACTED_ID]` |

### How It's Implemented

- **Module**: `frontend/src/lib/sanitizer.ts`
- **Method**: Multi-pass regex replacement pipeline
- **Integration**: `buildSanitizedPrompt()` is called before every API request
- **Transparency**: The UI shows a "PII stripped before sending" badge

---

## 3. Red-Flag Engine (Emergency Detection)

### What Makes It Special

Before any LLM is called, a **deterministic rule-based engine** checks for medical emergencies. This runs in milliseconds and has **zero false-negative tolerance** — it errs on the side of caution.

### What It Checks

**Vital Signs**:
- Heart rate < 40 or > 150 bpm → Emergency
- SpO₂ < 90% → Emergency  
- Systolic BP > 180 or < 80 mmHg → Emergency

**Symptom Keywords**:
- "crushing chest pain", "can't breathe", "sudden severe headache"
- "loss of consciousness", "coughing blood", "stroke symptoms"
- And many more emergency-indicator phrases

### How It's Implemented

- **Module**: `backend/core/red_flag_engine.py`
- **API**: `POST /api/triage` — returns instantly, no LLM call
- **Frontend**: If emergency detected, a full-screen overlay instructs the user to call 112/911
- **Design**: Deterministic rules = 100% reproducible, auditable, explainable

---

## 4. Local ML Symptom Classifier

### What Makes It Special

A **TF-IDF + Logistic Regression** model runs entirely on the server (no cloud ML API). It classifies symptoms into cardiac categories and severity levels before the LLM council even starts, providing an initial triage that guides the council.

### Classification Categories

| Category | Severity | Example Symptoms |
|----------|----------|------------------|
| `cardiac_emergency` | Critical | Crushing chest pain, radiating to arm, sweating |
| `cardiac_chronic` | Moderate | Persistent fatigue, swollen ankles, exercise intolerance |
| `cardiac_arrhythmia` | Moderate | Palpitations, irregular heartbeat, dizziness |
| `cardiac_risk` | Low-Moderate | Family history concerns, high cholesterol, smoking |
| `non_cardiac` | Low | Common cold, muscle pain, headache |

### How It's Implemented

- **Module**: `backend/ml/symptom_classifier.py`
- **Training**: Pre-trained on 64 curated medical examples at startup
- **Vectorizer**: `TfidfVectorizer` with 4096-dim feature space and medical n-grams
- **Model**: `LogisticRegression` with `multinomial` loss
- **Output**: Category, label, severity, confidence, description, and recommended action
- **API**: `POST /api/classify`
- **Integration**: Classification runs as a pre-stage in the council pipeline and results are displayed to the user

---

## 5. RAG Knowledge Base (FAISS + TF-IDF)

### What Makes It Special

MEDORBY uses **Retrieval-Augmented Generation (RAG)** to inject relevant medical knowledge into the LLM prompt. Instead of relying solely on the model's training data, the system retrieves up-to-date, curated medical information from a local FAISS index.

### Knowledge Sources

The knowledge base contains **15 curated documents** from authoritative sources:

| Topic | Source |
|-------|--------|
| Coronary Artery Disease (CAD) | AHA/ACC Guidelines |
| Acute Myocardial Infarction (Heart Attack) | ESC Guidelines |
| Heart Failure | ACC/AHA/HFSA Guidelines |
| Atrial Fibrillation | AHA/ACC/HRS Guidelines |
| Hypertension | ACC/AHA Guidelines |
| Cardiac Biomarkers (Troponin, BNP) | IFCC Recommendations |
| Cardiac Medications | AHA Pharmacology Guidelines |
| Heart Valve Disease | ACC/AHA VHD Guidelines |
| Cardiomyopathy | AHA Scientific Statement |
| Cardiac Rehabilitation | AHA/AACVPR Guidelines |
| Sudden Cardiac Arrest | AHA ACLS Guidelines |
| Peripheral Artery Disease | AHA/ACC PAD Guidelines |
| Pulmonary Hypertension | ESC/ERS Guidelines |
| Congenital Heart Disease | AHA/ACC ACHD Guidelines |
| Cardiovascular Risk Assessment | ACC/AHA Risk Calculator |

### How It's Implemented

- **Module**: `backend/rag/engine.py`
- **Indexing**: TF-IDF vectorisation → FAISS `IndexFlatIP` (inner product similarity)
- **Data**: `backend/knowledge_base/heart_knowledge.json` (curated JSON documents)
- **User Reports**: Uploaded medical reports are also indexed into the same FAISS index
- **Retrieval**: Top-k most relevant documents are retrieved based on cosine similarity
- **Augmentation**: Retrieved context is appended to the LLM prompt before council deliberation
- **API**: `POST /api/rag/retrieve`, `GET /api/rag/stats`

### RAG Pipeline Flow

```
User Symptoms
     │
     ▼
┌──────────────┐       ┌──────────────────────┐
│  TF-IDF      │──────►│  FAISS Index          │
│  Vectorize   │       │  (Knowledge Base +    │
│  Query       │       │   User Reports)       │
└──────────────┘       └──────────┬───────────┘
                                  │
                          Top-K Results
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Augmented Prompt        │
                    │  = Symptoms + RAG Context│
                    │       │                  │
                    │       ▼                  │
                    │  LLM Council             │
                    └─────────────────────────┘
```

---

## 6. Medical Report Upload & Analysis

### What Makes It Special

Users can upload their actual medical reports (PDF, DOCX, TXT). These reports:
- **Never leave the device** — stored locally in `user_reports/`
- **Are indexed into the RAG engine** — providing personalised medical context
- **Can be analysed by the full LLM council** — getting AI insights on lab results

### Supported Formats

| Format | Library |
|--------|---------|
| PDF | PyPDF2 |
| DOCX | python-docx |
| TXT | Built-in Python |

### How It's Implemented

- **Module**: `backend/rag/report_processor.py`
- **Storage**: Files saved to `backend/user_reports/` with UUID-based naming
- **Index**: `reports_index.json` maintains metadata (filename, word count, upload date)
- **RAG Integration**: After upload, the RAG engine rebuilds its index to include the new report
- **Analysis API**: `POST /api/reports/analyze/{id}` sends report through classifier + RAG + council
- **Frontend**: Collapsible panel with upload button, file list, and delete functionality

---

## 7. Hospital Database (Local Edge Storage)

### What Makes It Special

MEDORBY maintains a **local SQLite database** that simulates a hospital's electronic health records system. All data is **anonymised** — symptom hashes are stored instead of raw text.

### What Gets Stored

| Table | Contents |
|-------|----------|
| `medical_records` | Consultation records (category, severity, symptom hash, council summary, confidence) |
| `report_embeddings` | Chunk-level embeddings for uploaded reports |
| `federated_contributions` | Logged federated learning gradient submissions |

### How It's Implemented

- **Module**: `backend/storage/hospital_db.py`
- **Database**: SQLite (file: `backend/data/hospital_local.db`)
- **Anonymisation**: Symptoms are SHA-256 hashed before storage
- **Auto-Init**: Database schema is created automatically on first import
- **API**: `GET /api/hospital/stats`, `GET /api/hospital/records`
- **Federated Learning**: Consultation records can be used as training signals for federated model updates

---

## 8. Federated Learning with Differential Privacy

### What Makes It Special

MEDORBY implements **Federated Learning (FedAvg)** so that user feedback improves the global model **without sharing any raw medical data**. Every gradient update is protected by **Differential Privacy** (gradient clipping + Gaussian noise).

### Privacy Protections

| Protection | Method |
|------------|--------|
| **Gradient Clipping** | L2 norm clipping to bound sensitivity |
| **Gaussian Noise** | Calibrated noise injection (σ = 1.1) |
| **Minimum Clients** | Aggregation only triggers after 3+ client updates |
| **Anonymisation** | Only gradient hashes stored, no raw model weights |

### How It's Implemented

- **Module**: `backend/federated/`
- **Files**: `aggregator.py` (FedAvg + client management), `dp_privacy.py` (noise injection)
- **Client**: `frontend/src/lib/local_learning.ts` — computes adapter deltas + applies DP noise
- **Workflow**:
  1. User provides thumbs-up/thumbs-down feedback on council output
  2. Client computes a pseudo-gradient from the feedback
  3. DP noise is injected client-side
  4. Noised gradient is sent to `/api/federated/update`
  5. Server aggregates after 3+ updates using FedAvg
  6. Updated global adapter is available at `/api/federated/adapter`

---

## 9. Encrypted Local Storage (IndexedDB + AES-256)

### What Makes It Special

Medical consultation history is stored in the browser using **IndexedDB** with **AES-256-GCM encryption** via the Web Crypto API. The encryption key is derived per-session — even if someone accesses the browser's storage, they can't read the medical data.

### How It's Implemented

- **Module**: `frontend/src/lib/storage.ts`
- **Encryption**: Web Crypto API → `AES-256-GCM`
- **Key Derivation**: PBKDF2 from a user-specific seed
- **Data Stored**: Consultation history, council outputs, feedback records
- **Access**: Only the running application can decrypt stored data

---

## 10. Real-Time SSE Streaming

### What Makes It Special

The council process is streamed to the frontend in real-time using **Server-Sent Events (SSE)**. Users see each stage's progress as it happens — classification, RAG retrieval, divergence, convergence, and synthesis.

### SSE Event Stages

| Stage | Event | Data |
|-------|-------|------|
| `classification` | ML classifier result | Category, severity, confidence |
| `rag_retrieval` | Knowledge base search | Documents found, topics |
| `divergence` | Running → Complete | 3 member responses |
| `convergence` | Running → Complete | Peer review ranking |
| `synthesis` | Running → Complete | Final consensus |
| `done` | Pipeline finished | — |

### How It's Implemented

- **Backend**: `council_event_generator()` in `main.py` yields SSE events as async generator
- **Frontend**: `ChatInterface.tsx` reads the SSE stream with `fetch` + `ReadableStream` API
- **UI**: Progress bar with stage indicators, animated transitions between stages

---

## 🔮 Future Enhancements

| Feature | Status | Description |
|---------|--------|-------------|
| Multi-language support | Planned | Support for Hindi, Spanish, and more |
| Voice input | Planned | Speech-to-text for symptom description |
| Wearable integration | Research | Direct vital signs from smartwatches |
| Advanced RAG | Research | Dense embeddings via sentence-transformers |
| Multi-organ knowledge | Planned | Expand beyond cardiac to respiratory, neurological, etc. |
| HIPAA compliance audit | Planned | Full compliance review for clinical deployment |

---

<p align="center">
  📖 See <a href="ARCHITECTURE.md">ARCHITECTURE.md</a> for the detailed system architecture and implementation guide.
</p>
