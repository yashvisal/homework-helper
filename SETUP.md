# Setup Instructions

Follow these steps to run the Multi-Agent Homework Helper locally.

## 1) Clone and enter the project
```
git clone git@github.com:yashvisal/homework-helper.git
cd homework-helper
```

## 2) Create and activate a virtual environment
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

## 3) Install dependencies
```
pip install -r requirements.txt
```

## 4) Set required API keys
At minimum you need an OpenAI API key. Set keys in your shell before running:

Windows (PowerShell):
```
$env:OPENAI_API_KEY="your-openai-key"
$env:TAVILY_API_KEY="your-tavily-key"          # optional, enables web search
$env:WOLFRAM_ALPHA_APPID="your-wolfram-appid"  # optional, enables Wolfram Alpha
```

macOS / Linux:
```
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"          # optional, enables web search
export WOLFRAM_ALPHA_APPID="your-wolfram-appid"  # optional, enables Wolfram Alpha
```

## 5) Run the Streamlit app
```
streamlit run src/app.py
```

Then open http://localhost:8501 in your browser.

## Notes
- Uploaded documents are indexed per session in ChromaDB under `./chroma_data`.
- Web search (Tavily) is used alongside document RAG; Wolfram Alpha powers the homework solver’s math steps.
- If you change dependencies, re-run `pip install -r requirements.txt`.
# Setup Instructions

This guide will help you set up and run the Homework Helper application.

## Prerequisites

- Python 3.10 or higher
- OpenAI API key with GPT-5.1 access
- (Optional) Tavily API key for web search functionality

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/homework-helper.git
cd homework-helper
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root (or set environment variables):

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: Tavily API Key (for web search in Report Agent)
TAVILY_API_KEY=tvly-your-tavily-api-key-here

# Optional: Wolfram Alpha API Key (for math/STEM reasoning in Homework Solver)
WOLFRAM_ALPHA_APPID=your-wolfram-alpha-appid-here
```

**Getting API Keys:**

- **OpenAI**: Sign up at [platform.openai.com](https://platform.openai.com) and create an API key
- **Tavily**: Sign up at [tavily.com](https://tavily.com) for free web search API access
- **Wolfram Alpha**: Sign up at [wolframalpha.com](https://products.wolframalpha.com/api/) for API access

### 5. Run the Application

```bash
streamlit run src/app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Usage Guide

### Using the Report Writing Agent

1. Select "Report Writer" in the sidebar
2. (Optional) Upload PDF documents as sources
3. Enter your topic or research question
4. Use quick action buttons:
   - **Research Topic**: Gather information from documents and web
   - **Generate Outline**: Create a structured outline
   - **Write Full Draft**: Generate complete report with citations
5. Chat to refine, revise, or ask follow-up questions

### Using the Homework Solver Agent

1. Select "Homework Solver" in the sidebar
2. (Optional) Upload textbook/notes PDFs for reference
3. Either:
   - Upload an image of your homework problem
   - Type the problem directly in chat
4. Click "Solve This Problem" or send message
5. (Optional) Upload your work for feedback analysis

### Document Upload

- Supported formats: PDF
- Documents are chunked and stored in the vector database
- The knowledge base persists between sessions
- Use "Clear Knowledge Base" to reset

## Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Ensure `OPENAI_API_KEY` is set in your environment or `.env` file
- Check that the key is valid and has GPT-5.1 access

**"Web search not available"**
- This is optional; the Report Agent works without it
- Set `TAVILY_API_KEY` to enable web search

**Slow response times**
- GPT-5.1 calls may take 5-15 seconds depending on complexity
- Reduce chunk size or number of retrieved documents if needed

**ChromaDB errors**
- Delete the `chroma_data` directory to reset the vector store
- Ensure write permissions in the project directory

### Running Evaluation

To run the evaluation notebook:

```bash
cd notebooks
jupyter notebook evaluation.ipynb
```

Make sure the API keys are configured before running evaluation cells.

## Project Structure

```
homework-helper/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── report_agent.py
│   │   └── homework_agent.py
│   ├── tools/            # Agent tools
│   │   ├── retrieval.py
│   │   ├── web_search.py
│   │   └── vision.py
│   ├── vectorstore/      # Vector database
│   │   └── store.py
│   └── app.py            # Streamlit application
├── data/                 # Uploaded documents
├── docs/                 # Documentation
├── videos/               # Demo videos
├── requirements.txt      # Dependencies
└── README.md            # Project overview
```