# Technical RAG Exam API

## Project contents
- `app.py`: FastAPI service, semantic retrieval, evidence extraction and abstention.
- `knowledge_base.json`: the 50 supplied passages, in their original order.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Hugging Face Spaces Docker deployment.
- `.gitignore`: excludes virtual environments and secrets.

## Run locally on Windows PowerShell

Use Python 3.11 if available:

```powershell
cd C:\Sanjitaa\IIT\TDS\rag_exam
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API docs and
`http://127.0.0.1:8000/health` to check readiness.

## Test from a second PowerShell window

```powershell
$body = @{
    question = "What quorum does Raft require to commit a log entry?"
    top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8000/query" `
    -ContentType "application/json" `
    -Body $body
```

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/spaces.
2. Choose Docker as the SDK.
3. Choose CPU hardware and Public visibility if the grader must access it without credentials.
4. Upload `app.py`, `requirements.txt`, `Dockerfile`, `knowledge_base.json`, and `README.md`.
5. Wait for the Docker build to complete.
6. Test `https://YOUR-SPACE-NAME.hf.space/health`.
7. Submit `https://YOUR-SPACE-NAME.hf.space/query`.

The Dockerfile listens on port 7860, which is the expected Spaces port.

## Configuration

Optional environment variables:
- `MIN_PASSAGE_SCORE`: abstention threshold; default 0.34.
- `DEFAULT_TOP_K`: default retrieval count; default 3.
- `MAX_TOP_K`: maximum accepted top_k; default 10.
- `MAX_CONCURRENT_REQUESTS`: concurrent inference limit; default 4.
- `EMBEDDING_MODEL`: embedding model name.

Tune the abstention threshold only after evaluating representative answerable and
unanswerable questions. The returned confidence is a heuristic, not a calibrated
probability.

## Scope and limitations

This is an extractive RAG baseline. It uses dense semantic retrieval and selects
supporting sentences from the supplied passages; it does not call a generative LLM.
That design means retrieved passages are not inserted into an LLM prompt, reducing
prompt-injection risk. It does not guarantee detection of every conceivable
instruction-like string, and difficult multi-hop questions still need evaluation.
