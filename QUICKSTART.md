# AarogyaAI - Phase 0 Complete ✅

This document provides a quickstart guide for developers working with the Phase 0 implementation.

## What's Implemented (Phase 0)

✅ **Core Architecture**
- Project scaffolding with proper Python package structure
- JSON schemas for all data structures
- Pydantic models with validation
- Privacy-first design patterns

✅ **Privacy & Security Components**
- **Sanitizer Module**: Removes all PHI before cloud transmission
  - Email, phone, SSN, date removal
  - Relative time descriptors
  - Numeric value rounding
  - Fingerprinting for auditing
  - 18 comprehensive unit tests
  
- **Red-Flag Engine**: Local emergency detection
  - Immediate emergency rules (chest pain, stroke, etc.)
  - Urgent symptom detection
  - Combination rules (cardiac risk, sepsis)
  - Vital sign thresholds
  - 29 comprehensive unit tests

✅ **REST API**
- FastAPI application with health endpoints
- Query endpoint with emergency detection
- Sanitization validation endpoint
- 13 API tests covering all endpoints

✅ **Development Infrastructure**
- pytest configuration
- black/ruff code formatting
- Docker and docker-compose setup
- .gitignore for secure development

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/nirajpatil2005/AarogyaAI.git
cd AarogyaAI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_sanitizer.py -v
```

### 3. Start Local Server

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --port 8000

# Or use the convenience script
python -m app.main
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Query with routine symptoms
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "test-001",
    "symptoms": ["mild headache", "runny nose"]
  }'

# Query with emergency symptoms
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "test-002",
    "symptoms": ["severe chest pain", "shortness of breath"]
  }'

# Validate text for PHI
curl -X POST "http://localhost:8000/validate-sanitization?text=Patient has fever"
```

## API Examples

### Routine Query Response

```json
{
  "status": "SUCCESS",
  "urgency_level": "routine",
  "red_flag": false,
  "message": "Query processed successfully. Data sanitized and ready for analysis.",
  "recommendations": [
    "Monitor your symptoms",
    "Stay hydrated and rest",
    "Contact healthcare provider if symptoms persist or worsen",
    "This is not a substitute for professional medical advice"
  ],
  "fingerprint": "abc123...",
  "timestamp": "2026-02-18T08:54:37.895204"
}
```

### Emergency Response

```json
{
  "status": "EMERGENCY",
  "urgency_level": "immediate",
  "red_flag": true,
  "message": "IMMEDIATE EMERGENCY: Detected critical symptoms: severe chest pain...",
  "recommendations": [
    "SEEK IMMEDIATE MEDICAL ATTENTION",
    "Call emergency services (911) or go to nearest emergency room",
    "Do not delay - this may be a medical emergency"
  ],
  "fingerprint": null,
  "timestamp": "2026-02-18T08:54:33.286371"
}
```

## Code Quality

### Run Formatters

```bash
# Format code with black
black app/ llm/ ml/ federated/ tests/

# Check with ruff
ruff check app/ llm/ ml/ federated/ tests/

# Auto-fix ruff issues
ruff check --fix app/ llm/ ml/ federated/ tests/
```

### Type Checking (Coming Soon)

```bash
# Will be added in next iteration
mypy app/ llm/ ml/ federated/
```

## Docker Deployment

```bash
# Build and start services
cd infra
docker-compose up --build

# Access the API
curl http://localhost:8000/health

# Stop services
docker-compose down
```

## Architecture Overview

```
AarogyaAI/
├── app/              # FastAPI application
│   └── main.py       # REST API endpoints
├── llm/              # LLM and council components
│   ├── schemas.py    # Pydantic models
│   ├── sanitizer.py  # PHI removal (CRITICAL for privacy)
│   ├── red_flags.py  # Emergency detection
│   └── council_client.py  # Stub for Phase 2
├── ml/               # Machine learning components
│   └── feature_extractor.py  # Feature engineering
├── federated/        # Federated learning (Phase 3)
│   └── client.py     # Flower-based FL client stub
├── schemas/          # JSON schema definitions
├── tests/            # Comprehensive test suite
└── infra/            # Docker and deployment configs
```

## Key Privacy Features

1. **No PHI Leaves Local Node**
   - All patient data stays encrypted locally
   - Sanitizer removes names, dates, identifiers
   - Fingerprinting for audit trails

2. **Red-Flag Detection is Local**
   - Emergency detection happens on-device
   - No network delay for critical decisions
   - Conservative clinical rules

3. **Sanitized Cloud Communication**
   - Only sanitized, de-identified data sent to cloud
   - Schema validation on all payloads
   - Automatic PHI detection and rejection

## Testing

**Total Tests: 61 (All Passing ✅)**

- Sanitizer: 18 tests
- Red-Flag Engine: 29 tests  
- API: 13 tests
- ML: 1 test

All tests include:
- Positive cases (normal operation)
- Negative cases (no false positives)
- Edge cases (boundary conditions)
- Failure modes (error handling)

## Next Steps (Phase 1)

Phase 1 will implement:
- Local ML model training with XGBoost
- Encrypted local storage (SQLCipher)
- Vector search with FAISS
- Local-only prediction pipeline
- Enhanced feature extraction

## Contributing

See `instructions.md` for:
- Programming standards
- Security guidelines
- Testing requirements
- PR checklist

## Security

**CRITICAL**: Never commit:
- Patient data
- API keys
- Secrets
- .env files

All sensitive data must stay in `.gitignore`.

## Support

For issues or questions:
1. Check existing tests for usage examples
2. Review `instructions.md` for guidelines
3. Open an issue on GitHub

---

**License**: See LICENSE file
**Version**: 0.1.0 (Phase 0)
**Status**: ✅ Phase 0 Complete - Ready for Phase 1
