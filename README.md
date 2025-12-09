# Homework Helper

A multi-agent AI system for academic work featuring specialized agents for report writing, homework solving, and tutoring over uploaded materials.

## What it Does

Homework Helper is an AI-powered academic assistant with three specialized agents:

1. **Report Writing Agent**: Human-in-the-loop workflow (research notes → outline → draft → feedback incorporation) using uploaded documents and web search, with precise citations.  
2. **Homework Solver Agent**: Upload a homework PDF, auto-extract each question, and solve step-by-step.  
3. **Tutor Agent**: Q&A over uploaded study materials with retrieval-augmented answers and summaries.

The system uses RAG over uploaded PDFs plus live web search, and keeps per-session isolation so documents from other sessions are not mixed.

## Quick Start

1) Create and activate a virtual environment  
Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```
macOS / Linux:
```
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Set environment variables
```
# Windows (PowerShell)
$env:OPENAI_API_KEY="your-openai-key"
$env:TAVILY_API_KEY="your-tavily-key"          # optional, enables web search
$env:WOLFRAM_ALPHA_APPID="your-wolfram-appid"  # optional, enables Wolfram Alpha

# macOS / Linux
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"
export WOLFRAM_ALPHA_APPID="your-wolfram-appid"
```

4) Run the application
```
streamlit run src/app.py
```

5) Open your browser to `http://localhost:8501`

## Video Links

- **Project Demo (3-5 min)**: [videos/demo.mp4](videos/demo.mp4) - Overview of features and capabilities
- **Technical Walkthrough (5-10 min)**: [videos/technical.mp4](videos/technical.mp4) - Architecture and implementation details

## Features

### Report Writing Agent
- Staged, human-in-loop flow: Research Notes → Outline → Draft → Feedback → Revised draft
- Dual sourcing: uploaded documents (session-scoped) + web search (even when docs exist)
- Explicit citation formats: `[Source: <filename>, p.<page>]` and `[Source: <title>, <url>]`
- Longer, detailed sections (targeting 1,000–1,500 words)
- Logs for tool calls and stage completion to aid debugging

### Homework Solver Agent
- Upload a homework PDF, auto-extract questions, and solve each sequentially
- Wolfram Alpha integration for symbolic/math reliability
- Progress feedback (counts, per-question status)
- Stores extracted text and solutions in session state; re-runnable per session

### Tutor Agent
- Q&A over uploaded study materials with retrieval-augmented answers
- Summaries of uploaded docs to onboard the user
- Session-isolated retrieval so only current-session docs are used

### RAG & Infra
- ChromaDB with `session_id` metadata to isolate runs (`./chroma_data`)
- OpenAI `text-embedding-3-small` for chunk embeddings
- PyMuPDF loaders for PDFs; graceful error surfacing in UI

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          Streamlit Web UI                          │
│  Agent selection + session state (session_id)                      │
└──────────────┬─────────────────────┬─────────────────────┬─────────┘
               │                     │                     │
               │                     │                     │
               ▼                     ▼                     ▼
      ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
      │ Report Agent   │     │ Homework Agent │     │ Tutor Agent    │
      │ (LangGraph)    │     │ (LangGraph)    │     │ (LangGraph)    │
      └──────┬─────────┘     └──────┬─────────┘     └──────┬─────────┘
             │                     │                     │
             ▼                     ▼                     ▼
      Tools / Calls         Tools / Calls         Tools / Calls
      - Retrieval tool      - Question extractor  - Retrieval tool
      - Web search (Tavily) - Wolfram Alpha       - Document summarizer
      - Outline / Draft     - Retrieval tool

               └──────────────────────┬──────────────┬──────────────┘
                                      ▼
                            ┌────────────────────┐
                            │  ChromaDB Vector   │
                            │  Store (sessioned) │
                            │  text-embedding-3  │
                            └────────────────────┘
```

## Technology Stack

- **LLM**: OpenAI GPT-5.1 (multimodal)
- **Agent Framework**: LangGraph
- **Vector Store**: ChromaDB
- **Embeddings**: OpenAI text-embedding-3-small
- **Web UI**: Streamlit
- **Web Search**: Tavily API
- **PDF Processing**: PyMuPDF