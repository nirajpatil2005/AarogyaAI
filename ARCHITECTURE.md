# 🏗️ MEDORBY — System Architecture & Implementation

> Technical deep-dive into how MEDORBY is built, the design decisions behind each component, and the end-to-end data flow.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Data Flow Pipeline](#2-data-flow-pipeline)
3. [Backend Architecture](#3-backend-architecture)
4. [Frontend Architecture](#4-frontend-architecture)
5. [Council Protocol Design](#5-council-protocol-design)
6. [RAG System Design](#6-rag-system-design)
7. [ML Pipeline Design](#7-ml-pipeline-design)
8. [Storage Architecture](#8-storage-architecture)
9. [Privacy Architecture](#9-privacy-architecture)
10. [Federated Learning Architecture](#10-federated-learning-architecture)
11. [API Design](#11-api-design)
12. [Deployment Guide](#12-deployment-guide)
13. [Design Decisions & Trade-offs](#13-design-decisions--trade-offs)

---

## 1. High-Level Architecture

MEDORBY follows a **"Local-First, Cloud-Consensus"** architecture. Sensitive operations run locally; only sanitised data reaches cloud LLMs for clinical reasoning.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EDGE (User's Device)                            │
│                                                                         │
│  ┌───────────┐  ┌──────────────┐  ┌────────────────┐  ┌────────────┐  │
│  │ Browser   │  │ PII          │  │ Encrypted      │  │ Federated  │  │
│  │ (Next.js) │  │ Sanitizer    │  │ IndexedDB      │  │ Learning   │  │
│  │           │  │ (client-side)│  │ (AES-256-GCM)  │  │ Client     │  │
│  └─────┬─────┘  └──────┬───────┘  └────────────────┘  └──────┬─────┘  │
│        │               │                                      │        │
│        │    Sanitised   │                        DP-noised     │        │
│        │    Prompt      │                        Gradients     │        │
└────────│───────────────│──────────────────────────────────────│────────┘
         │               │                                      │
         ▼               ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     SERVER (FastAPI Backend)                             │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│  │ Red-Flag │  │ ML       │  │ RAG      │  │ Hospital │  │ FedAvg   ││
│  │ Engine   │  │ Classifier│  │ Engine   │  │ DB       │  │Aggregator││
│  │ (core/)  │  │ (ml/)    │  │ (rag/)   │  │(storage/)│  │(federated│|
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘│
│       │              │              │              │                    │
│       ▼              ▼              ▼              │                    │
│  ┌──────────────────────────────────────────┐     │                    │
│  │         Council Orchestrator              │     │                    │
│  │         (council/)                        │     │                    │
│  │                                           │     │                    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │     │                    │
│  │  │Member A │ │Member B │ │Member C │    │     │                    │
│  │  │Llama 70B│ │Llama 8B │ │Qwen3 32B│    │     │                    │
│  │  └────┬────┘ └────┬────┘ └────┬────┘    │     │                    │
│  │       └───────────┼───────────┘          │     │                    │
│  │                   ▼                      │     │                    │
│  │  ┌──────────────────────────────┐        │     │                    │
│  │  │ Reviewer → Chairman          │        │     │                    │
│  │  │ (Peer Review → Synthesis)    │        │     │                    │
│  │  └──────────────────────────────┘        │     │                    │
│  └──────────────────┬───────────────────────┘     │                    │
│                     │                              │                    │
│                     ▼                              ▼                    │
│              SSE Stream ──────────────► Store Anonymised Record         │
│                     │                                                   │
└─────────────────────│───────────────────────────────────────────────────┘
                      │
                      ▼
               ┌──────────────┐
               │  Browser UI  │
               │  (Real-time) │
               └──────────────┘
```

### Architectural Principles

| Principle | Implementation |
|-----------|---------------|
| **Privacy by Design** | PII never leaves the browser; all storage encrypted |
| **Local-First** | Emergency detection, classification, and storage run locally |
| **Cloud-Consensus** | Only sanitised prompts reach cloud LLMs; multiple models cross-verify |
| **Fail-Safe** | If any model fails, the pipeline continues with fallback data |
| **Modular** | Each subsystem is an independent Python package with clear interfaces |
| **Observable** | SSE streaming gives real-time visibility into every pipeline stage |

---

## 2. Data Flow Pipeline

The complete request lifecycle from user input to displayed results:

```
Step 1: USER INPUT
  │  Symptoms + optional vitals + optional age/sex
  │
  ▼
Step 2: PII SANITIZATION (Client-Side — sanitizer.ts)
  │  Regex-based removal of names, emails, phones, dates
  │  Output: sanitised_prompt (safe to send over network)
  │
  ▼
Step 3: RED-FLAG TRIAGE (Server — core/red_flag_engine.py)
  │  Deterministic vital sign checks + keyword matching
  │  If EMERGENCY → return immediately (no LLM calls)
  │  If SAFE → continue to Step 4
  │
  ▼
Step 4: LOCAL ML CLASSIFICATION (Server — ml/symptom_classifier.py)
  │  TF-IDF vectorisation → Logistic Regression prediction
  │  Output: category, severity, confidence
  │  Streamed to frontend via SSE
  │
  ▼
Step 5: RAG CONTEXT RETRIEVAL (Server — rag/engine.py)
  │  TF-IDF vectorise query → FAISS similarity search
  │  Top-K relevant documents from knowledge base + user reports
  │  Output: context string appended to prompt
  │  Streamed to frontend via SSE
  │
  ▼
Step 6: COUNCIL DELIBERATION (Server — council/orchestrator.py)
  │
  │  6a. DIVERGENCE — 3 models reason in parallel
  │      Augmented prompt (symptoms + RAG context) sent to:
  │      - Member A (Llama 3.3 70B)
  │      - Member B (Llama 3.1 8B)
  │      - Member C (Qwen3 32B)
  │
  │  6b. CONVERGENCE — Reviewer ranks responses
  │      Compact summaries sent to Reviewer (Llama 8B)
  │      Output: ranking [A, B, C] + reasoning
  │
  │  6c. SYNTHESIS — Chairman produces final answer
  │      Top-ranked response + ranking sent to Chairman (Llama 70B)
  │      Output: final_differentials, recommended_next_steps,
  │              confidence, red_flag, summary
  │
  ▼
Step 7: POST-PROCESSING (Server — storage/hospital_db.py)
  │  Anonymised consultation record stored in SQLite
  │  Symptom hash (SHA-256), not raw text
  │
  ▼
Step 8: DISPLAY (Frontend — ChatInterface.tsx)
  │  Classification card, RAG context, council consensus
  │  Individual member responses (expandable)
  │
  ▼
Step 9: FEEDBACK (Optional)
     User provides thumbs-up/thumbs-down
     → DP-noised gradient → Federated aggregator
```

---

## 3. Backend Architecture

### Module Structure

```
backend/
├── main.py              ← FastAPI entry point, all route definitions
├── config.py            ← Pydantic BaseSettings (.env loader)
│
├── core/                ← Core utilities (no external API calls)
│   ├── __init__.py
│   └── red_flag_engine.py
│
├── council/             ← LLM Council (Groq API)
│   ├── __init__.py
│   ├── groq_client.py   ← AsyncGroq client, model config, parallel queries
│   └── orchestrator.py  ← 3-stage protocol: diverge → converge → synthesise
│
├── ml/                  ← Machine Learning (local, no API)
│   ├── __init__.py
│   └── symptom_classifier.py  ← TF-IDF + Logistic Regression classifier
│
├── rag/                 ← Retrieval-Augmented Generation (local)
│   ├── __init__.py
│   ├── engine.py        ← FAISS index + TF-IDF vectoriser + retrieval
│   └── report_processor.py  ← PDF/DOCX/TXT text extraction + storage
│
├── storage/             ← Persistent storage (local SQLite)
│   ├── __init__.py
│   └── hospital_db.py   ← Schema, CRUD operations, stats
│
└── federated/           ← Federated Learning
    ├── __init__.py
    ├── aggregator.py    ← FedAvg aggregation, client management
    └── dp_privacy.py    ← Gradient clipping + Gaussian noise injection
```

### Key Design Decisions

1. **Flat `main.py`**: All routes in one file for simplicity. Each route delegates to the appropriate module.
2. **Module `__init__.py` exports**: Each package exports its public API, so `main.py` imports are clean.
3. **No ORM**: Raw SQLite for the hospital DB — simpler, fewer dependencies, better performance for this use case.
4. **Singleton patterns**: `get_rag_engine()`, `get_classifier()`, and `get_settings()` use caching/singletons to avoid re-initialisation.

### Dependency Graph

```
main.py
  ├── config.py (settings)
  ├── core/red_flag_engine.py (no deps)
  ├── council/
  │   ├── groq_client.py ← config.py
  │   └── orchestrator.py ← groq_client.py
  ├── ml/symptom_classifier.py (sklearn, numpy)
  ├── rag/
  │   ├── engine.py (faiss, sklearn)
  │   └── report_processor.py (PyPDF2, python-docx)
  ├── storage/hospital_db.py (sqlite3)
  └── federated/
      ├── aggregator.py (numpy)
      └── dp_privacy.py (numpy)
```

---

## 4. Frontend Architecture

### Component Hierarchy

```
app/layout.tsx                 ← Root layout (fonts, meta tags)
  └── app/page.tsx             ← Landing page
        ├── Hero Section       ← Title, CTA, animated stats
        ├── Pipeline Section   ← 5-step visual pipeline
        ├── Features Grid      ← 9 feature cards
        ├── Council Grid       ← 5 AI member cards
        ├── ChatInterface.tsx  ← Main consultation component
        │     ├── Report Upload Panel
        │     ├── Demographics Form
        │     ├── Vitals Form
        │     ├── Symptoms Textarea
        │     ├── Classification Card
        │     ├── RAG Context Card
        │     ├── Stage Progress Bar
        │     ├── Consensus Result Card
        │     ├── Member Tabs (expandable)
        │     └── Feedback Buttons
        └── Footer             ← Tech stack badges, disclaimer
```

### Frontend Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Next.js | 16.1.6 | React framework (App Router) |
| React | 19.2.3 | UI library |
| Lucide React | 0.574+ | Icon library |
| Vanilla CSS | — | Full design system (no Tailwind) |

### Design System

The CSS is organised in `globals.css` using BEM-like naming:

```css
/* Block */
.result-card { ... }

/* Element */
.result-card__header { ... }
.result-card__title { ... }

/* Modifier */
.stage-item--running { ... }
.stage-item--complete { ... }
```

Key design tokens:
- **Background**: `#020617` (near-black) with glassmorphism overlays
- **Primary**: `#3b82f6` (blue), `#10b981` (green), `#8b5cf6` (violet)
- **Typography**: Inter (Google Fonts) system stack
- **Animations**: CSS keyframes for fade-in, pulse, spin

---

## 5. Council Protocol Design

### Why a Multi-Model Council?

Single-model medical AI has inherent limitations:
- **Hallucination risk**: One model may confidently assert incorrect diagnoses
- **Training bias**: Each model has different training data coverage
- **No cross-verification**: A single model can't catch its own errors

The council approach solves these by:
- **Diversity**: 3 different model architectures (Llama 70B, Llama 8B, Qwen3 32B)
- **Peer Review**: Independent ranking catches outlier/hallucinated responses
- **Consensus**: Chairman synthesises only the top-ranked reasoning

### Stage Details

#### Divergence (Parallel Fan-Out)

```python
# council/groq_client.py
async def query_council_parallel(sanitized_prompt: str) -> dict[str, str]:
    tasks = {
        name: query_groq(model, [system_msg, user_msg])
        for name, model in COUNCIL_MODELS.items()
        if name in ("member_a", "member_b", "member_c")
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
```

- All 3 models run **simultaneously** via `asyncio.gather`
- Each returns structured JSON: `{differentials, next_steps, confidence, red_flag}`
- Exceptions are caught and replaced with fallback JSON

#### Convergence (Peer Review)

```python
# council/orchestrator.py — Compact prompt design
summary_lines = "\n".join(
    f"  {anon_map[m]}: {_summarise_response(divergence_results[m])}"
    for m in members
)
```

- Member responses are **anonymised** (A, B, C) to prevent model bias
- Only **compact summaries** are sent (not full JSON), reducing latency
- Reviewer model outputs a ranking + brief reasoning

#### Synthesis (Chairman)

- Only the **top-ranked** member's full response is sent to the Chairman
- Chairman produces: `final_differentials`, `recommended_next_steps`, `confidence`, `summary`
- Temperature is set to **0.2** for deterministic, focused output

---

## 6. RAG System Design

### Indexing Pipeline

```
Knowledge Base JSON ──┐
                      ├──► TF-IDF Vectoriser ──► FAISS IndexFlatIP
User Report Text ─────┘        (4096-dim)
```

### Retrieval Pipeline

```
Query Text ──► TF-IDF Transform ──► FAISS Search ──► Top-K Results
                                                          │
                                                          ▼
                                                    Context String
                                                    (appended to LLM prompt)
```

### Technical Details

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Vectoriser** | TF-IDF (4096-dim) | Fast, no GPU required, good for medical vocabulary |
| **Index** | FAISS `IndexFlatIP` | Exact inner product search, optimal for small corpora |
| **Similarity** | Cosine (via L2-normalised IP) | Standard for text retrieval |
| **Fallback** | TF-IDF (not sentence-transformers) | `sentence-transformers` has import issues; TF-IDF is reliable |

### Why TF-IDF Instead of Dense Embeddings?

Dense embeddings (e.g., from `sentence-transformers`) are generally better for semantic search. However:
1. **Dependency issues**: `sentence-transformers` has known import failures in some environments
2. **Speed**: TF-IDF index builds in milliseconds vs. minutes for dense models
3. **Medical vocabulary**: TF-IDF naturally handles medical terminology without needing fine-tuned embeddings
4. **Future upgrade path**: The RAG engine is architected to swap in dense embeddings when available

---

## 7. ML Pipeline Design

### Training Data

The classifier is pre-trained at startup using 64 curated examples across 5 categories:

```python
# ml/symptom_classifier.py
TRAINING_DATA = [
    ("crushing chest pain radiating to left arm sweating nausea", "cardiac_emergency"),
    ("persistent fatigue swollen ankles difficulty breathing when lying down", "cardiac_chronic"),
    ("heart skipping beats irregular pulse dizziness lightheaded", "cardiac_arrhythmia"),
    # ... 64 total examples
]
```

### Model Architecture

```
Input Text → TfidfVectorizer(max_features=4096) → LogisticRegression(C=1.0, multinomial)
                                                         │
                                                         ▼
                                                  {category, severity, confidence,
                                                   description, action, probabilities}
```

### Why Logistic Regression?

| Factor | Logistic Regression | Deep Learning |
|--------|-------------------|---------------|
| **Training time** | ~50ms | Minutes to hours |
| **Inference time** | <1ms | 10-100ms |
| **Data needed** | ~60 examples | 1000s+ examples |
| **Interpretability** | High (coefficients) | Low (black box) |
| **GPU required** | No | Yes |
| **Accuracy (for this task)** | Good enough for triage routing | Marginal improvement |

The classifier's role is triage routing, not final diagnosis — the LLM council provides the detailed clinical reasoning.

---

## 8. Storage Architecture

### Dual-Storage Strategy

| Storage | Location | Encryption | Contents |
|---------|----------|------------|----------|
| **IndexedDB** | Browser | AES-256-GCM | Consultation history, preferences |
| **SQLite** | Server (`data/`) | None (local file) | Anonymised medical records, federated contributions |
| **File System** | Server (`user_reports/`) | None (local file) | Uploaded medical reports |
| **FAISS Index** | Server (`faiss_index/`) | None (memory/disk) | Vector index for RAG retrieval |

### SQLite Schema

```sql
-- Anonymised consultation records
CREATE TABLE medical_records (
    id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL,        -- 'consultation' | 'report'
    category TEXT,                     -- from classifier
    severity TEXT,                     -- 'critical' | 'moderate' | 'low'
    symptoms_hash TEXT,                -- SHA-256 hash (never raw text)
    council_summary TEXT,              -- anonymised council output
    confidence REAL,
    timestamp TEXT NOT NULL,
    metadata TEXT                      -- JSON for extension fields
);

-- Federated learning audit trail
CREATE TABLE federated_contributions (
    id TEXT PRIMARY KEY,
    record_id TEXT,
    gradient_hash TEXT,
    dp_noise_level REAL,
    contributed_at TEXT NOT NULL,
    aggregation_round INTEGER,
    status TEXT DEFAULT 'pending'      -- 'pending' | 'aggregated'
);
```

---

## 9. Privacy Architecture

### Privacy Layers

```
Layer 1: CLIENT-SIDE PII SANITISATION
  ├── Regex removal of names, emails, phones, dates, IDs
  ├── Runs entirely in the browser (sanitizer.ts)
  └── Zero PII reaches the network

Layer 2: DETERMINISTIC TRIAGE
  ├── Red-Flag Engine uses rule-based logic (no ML/LLM)
  ├── Emergency detection requires zero cloud calls
  └── Auditable, reproducible, explainable

Layer 3: LOCAL ML CLASSIFICATION
  ├── TF-IDF + LogReg runs on the server (no cloud ML)
  ├── No training data leaves the device
  └── Classification informs but doesn't replace LLM

Layer 4: ANONYMISED STORAGE
  ├── SHA-256 symptom hashes (not raw text) in SQLite
  ├── AES-256-GCM encrypted local history in browser
  └── Uploaded reports never transmitted anywhere

Layer 5: DIFFERENTIAL PRIVACY (Federated Learning)
  ├── L2 gradient clipping (bound sensitivity)
  ├── Gaussian noise injection (σ = 1.1)
  ├── Only aggregate after 3+ client updates
  └── Formal (ε, δ)-differential privacy guarantee
```

### Threat Model

| Threat | Mitigation |
|--------|------------|
| Cloud provider reads medical data | PII stripped client-side; only sanitised prompts sent |
| Network eavesdropper | HTTPS + no PII in transit |
| Local device theft | IndexedDB encrypted with AES-256-GCM |
| Model update reveals data | Differential privacy noise on all gradients |
| Server compromise | SQLite contains only hashes, not raw symptoms |

---

## 10. Federated Learning Architecture

### FedAvg Protocol

```
Client 1 ──► DP(∇θ₁) ──┐
                         │
Client 2 ──► DP(∇θ₂) ──┼──► Aggregator ──► θ_global = Σ(DP(∇θᵢ)) / N
                         │
Client 3 ──► DP(∇θ₃) ──┘
```

### Implementation Flow

1. **User Feedback**: Thumbs-up or thumbs-down on council output
2. **Pseudo-Gradient**: `local_learning.ts` computes a gradient-like vector from the feedback text
3. **DP Noise**: Gradient is clipped (L2 norm) and Gaussian noise is added client-side
4. **Submission**: Noised gradient sent to `POST /api/federated/update`
5. **Aggregation**: Server buffers updates; FedAvg runs when ≥3 clients have submitted
6. **Distribution**: Updated adapter available at `GET /api/federated/adapter`

### Privacy Budget

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Clip norm (C) | 1.0 | Maximum L2 norm of any single gradient |
| Noise multiplier (σ) | 1.1 | Gaussian noise std relative to clip norm |
| Min clients (K) | 3 | Minimum updates before aggregation |
| Privacy (ε) | ~2-3 | Per-round privacy budget (approximate) |

---

## 11. API Design

### REST Conventions

- All endpoints use JSON request/response bodies
- SSE streaming for long-running council deliberations
- Standard HTTP status codes (200, 400, 404, 500)
- CORS configured for `localhost:3000` (frontend)

### Request Models

```python
class SymptomRequest(BaseModel):
    sanitized_prompt: str          # PII-free text from sanitizer.ts
    vitals: Optional[dict] = None  # {heart_rate, spo2, systolic_bp}

class FederatedUpdateRequest(BaseModel):
    client_id: str                 # Unique client identifier
    gradients: list[float]         # DP-noised adapter delta
```

### SSE Event Format

```json
{"stage": "classification", "status": "complete", "data": {...}}
{"stage": "rag_retrieval", "status": "complete", "data": {...}}
{"stage": "divergence", "status": "running"}
{"stage": "divergence", "status": "complete", "data": {...}}
{"stage": "convergence", "status": "running"}
{"stage": "convergence", "status": "complete", "data": {...}}
{"stage": "synthesis", "status": "running"}
{"stage": "synthesis", "status": "complete", "data": {...}}
{"stage": "done"}
```

---

## 12. Deployment Guide

### Development (Local)

```bash
# Terminal 1: Backend
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Add your GROQ_API_KEY
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### Environment Variables

#### Backend (`backend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | — | Groq Cloud API key |
| `VLLM_BASE_URL` | No | `http://localhost:8001/v1` | vLLM endpoint (if using) |
| `VLLM_MODEL_NAME` | No | `meta-llama/Llama-3-8B-Instruct` | vLLM model (if using) |
| `FEDERATED_SECRET_KEY` | No | `change_this_secret_key` | Secret for federated auth |

#### Frontend (`frontend/.env.local`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_BACKEND_URL` | No | `http://localhost:8000` | Backend API URL |

### Production Considerations

| Concern | Recommendation |
|---------|----------------|
| **HTTPS** | Use a reverse proxy (nginx) with Let's Encrypt |
| **CORS** | Restrict `allow_origins` to your production domain |
| **Rate Limiting** | Add API rate limiting (e.g., `slowapi`) |
| **Monitoring** | Add structured logging + metrics (Prometheus/Grafana) |
| **Database** | Consider PostgreSQL for production hospital DB |
| **Secrets** | Use proper secret management (HashiCorp Vault, AWS Secrets Manager) |

---

## 13. Design Decisions & Trade-offs

### Why Groq Cloud Instead of Self-Hosted?

| Factor | Groq Cloud | Self-Hosted (vLLM) |
|--------|-----------|-------------------|
| **Latency** | ~200ms per model call | 2-5s per model call |
| **Cost** | Free tier available | GPU hardware costs |
| **Privacy** | Sanitised data only | Full control |
| **Reliability** | 99.9% uptime | Self-managed |
| **Setup** | API key only | GPU setup, model download |

**Decision**: Use Groq for speed and accessibility. PII sanitization ensures privacy despite cloud calls.

### Why 3 Models Instead of 1 or 5?

- **1 model**: No error checking, hallucination risk
- **3 models**: Optimal diversity-to-latency ratio; majority voting is meaningful
- **5+ models**: Diminishing returns; latency increases; cost multiplies

### Why TF-IDF Instead of Dense Embeddings for RAG?

- **Reliability**: TF-IDF has zero dependency issues
- **Speed**: Index builds in milliseconds
- **Medical domain**: Term-level matching works well for medical queries
- **Upgrade path**: Architecture supports swapping to dense embeddings later

### Why SQLite Instead of PostgreSQL?

- **Local-first**: No server setup required
- **Privacy**: Data stays in a local file, not a network database
- **Simplicity**: Zero configuration, zero maintenance
- **Portability**: Database is a single file that moves with the project

### Why SSE Instead of WebSockets?

- **Simplicity**: One-directional stream is sufficient (server → client)
- **Compatibility**: Works with standard HTTP; no upgrade handshake needed
- **Resilience**: Automatic reconnection built into browser `EventSource` API
- **Fetch API**: Can be consumed with standard `fetch` + `ReadableStream`

---

<p align="center">
  📖 See <a href="FEATURES.md">FEATURES.md</a> for detailed feature documentation.<br>
  📖 See <a href="README.md">README.md</a> for setup instructions.
</p>
