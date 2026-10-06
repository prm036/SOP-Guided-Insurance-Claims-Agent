# SOP-Guided Insurance Claims Agent

An agentic conversational AI that follows a strict Standard Operating Procedure (SOP)
workflow while maintaining natural, empathetic conversation.

For an in-depth technical deep-dive on the design, methodology, and workings of this application, please read [DOCS](DOCS.md).

## Architecture

The agent uses a **SOP Harness Pattern** — a deterministic state machine controls
*what* the LLM is allowed to do, while the LLM controls *how* it talks.

```
User Message
  -> Scope Guard         (reject off-topic)
  -> Phase Gate          (determine available tools/data)
  -> Context Assembly    (build phase-specific LLM context)
  -> LLM Invocation      (generate response + function calls)
  -> Tool Processing     (verify identity, store memory, resolve intent...)
  -> Gate Check          (advance phase if conditions met)
  -> Response Assembly   (return response + updated state)
```

### SOP Phases

| Phase | Freedom Level | Gate |
|-------|--------------|------|
| **VERIFY_ID** | Strict | >=3 PII fields verified server-side |
| **RESOLVE_INTENT** | Flexible | Intent + claim resolved |
| **PROCESS_CASE** | Flexible (grounded) | User satisfied |
| **POST_PROCESS** | Semi-strict | Email decision made |

### Key Components

| Module | Purpose |
|--------|---------|
| `agent/harness.py` | Master orchestrator - wires everything together |
| `agent/state_machine.py` | Deterministic phase controller with hard gates |
| `agent/identity_verifier.py` | Server-side PII matching with fuzzy name support |
| `agent/memory.py` | Cross-phase memory (stores hints for later phases) |
| `agent/scope_guard.py` | Two-tier scope classification (keywords + LLM fallback) |
| `agent/emotion_tracker.py` | Emotion detection with de-escalation guidance |
| `agent/context_assembler.py` | Phase-gated data injection into LLM context |
| `llm/tools.py` | Function-calling schemas with phase-gated availability |
| `llm/prompts.py` | Multi-layer prompt architecture |

## Quick Start

### Option 1: Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 3. Run
python app.py
```

Open http://localhost:5000

### Option 2: Docker

```bash
# 1. Set your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 2. Build and run
docker compose up --build
```

Open http://localhost:5000

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key for the LLM provider |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `OPENAI_BASE_URL` | OpenAI default | Custom API endpoint (for Azure, Ollama, etc.) |
| `PORT` | `5000` | Server port |

## Test Scenario

Use this test input to verify the full workflow:

> "I'm the policyholder. My name is Margaret Chen, policy POL-9921.
> I'm calling about my denied healthcare claim from January.
> DOB is 1985-03-15, SSN last four is 4472."

**Expected behaviour:**
1. Agent extracts name, policy number, DOB, SSN last 4
2. Server verifies 3+ PII fields -> identity confirmed
3. Agent uses cross-phase memory ("denied healthcare claim from January") to auto-resolve intent
4. Agent explains denial reason (missing pathology report + office note) from grounded claim data
5. Agent offers email summary before closing

## Project Structure

```
insurance_claims/
├── fixtures/               # Test data (policyholders, claims, guidelines)
├── agent/
│   ├── harness.py          # Master orchestrator
│   ├── state_machine.py    # Phase controller + gates
│   ├── identity_verifier.py # Server-side PII matching
│   ├── memory.py           # Cross-phase memory
│   ├── scope_guard.py      # Scope classification
│   ├── emotion_tracker.py  # Emotion intelligence
│   ├── context_assembler.py # Phase-gated data injection
│   └── email_generator.py  # Email summary formatter
├── llm/
│   ├── client.py           # OpenAI-compatible API wrapper
│   ├── prompts.py          # Multi-layer prompt architecture
│   └── tools.py            # Phase-gated function calling schemas
├── data/
│   └── store.py            # Fixture data loader + queries
├── templates/
│   └── index.html          # Basic chat UI
├── app.py                  # Flask entry point
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
