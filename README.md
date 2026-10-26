# AI-Powered Document Intelligence Platform

An AI- and NLP-powered Streamlit application for summarizing and analyzing PDF,
DOCX, and TXT documents. It extracts structured information, visualizes key
findings, and supports document-based questions using Ollama and TinyLlama.

## Features

- Upload PDF, DOCX, and TXT documents
- Generate summaries and explanations
- Extract structured information
- Analyze document content and visualize keywords and numbers
- Ask questions about an uploaded document
- Store document metadata and feedback in MySQL

## Requirements

- Python 3.10 or newer
- MySQL server
- [Ollama](https://ollama.com/) with the `tinyllama` model (or another model
  configured with `OLLAMA_MODEL`)

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Create the MySQL database:

   ```sql
   CREATE DATABASE document_intelligence;
   ```

   The application creates its tables on startup. Configure the connection
   using environment variables. `DB_PASSWORD` is required:

   ```powershell
   $env:DB_HOST = "localhost"
   $env:DB_PORT = "3306"
   $env:DB_USER = "root"
   $env:DB_PASSWORD = "<your-mysql-password>"
   $env:DB_NAME = "document_intelligence"
   ```

3. Start Ollama and download a model:

   ```powershell
   ollama pull tinyllama
   ```

   To use a different installed model, set `OLLAMA_MODEL` before starting the
   application.

4. Run the application:

   ```powershell
   python -m streamlit run app.py
   ```

5. Open <http://localhost:8501>.

## Notes

- Uploaded documents and generated outputs are stored locally and excluded from
  Git.
- Scanned PDFs may require OCR before their text can be analyzed.
- AI-generated results may contain errors; verify important information.
