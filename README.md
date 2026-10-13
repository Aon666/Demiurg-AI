# 🗺️ Demiurg-AI

> **Local-First RAG Assistant for TTRPG Worldbuilding & Lore Management**

**Demiurg-AI** is a local Retrieval-Augmented Generation (RAG) system engineered specifically for managing and querying extensive TTRPG lore and worldbuilding documentation. It combines local Large Language Models (LLMs) and vector databases with flexible metadata filtering to deliver accurate, hallucination-free answers strictly grounded in source Markdown documents.

---

## 🏗️ Architecture & Features

* **Local & Privacy-Centric Pipeline:** Operates 100% locally via **Ollama** (`llama3.2:3b` & `nomic-embed-text`), eliminating external API costs and data exposure.
* **Intelligent Ingestion Pipeline:** 
  * Parses YAML frontmatter (regions, domains, tags).
  * **Header-Aware Chunking:** Splits text along Markdown headings (`#`, `##`, `###`) to preserve semantic context.
  * **Incremental Updates (MD5 Hashes):** Tracks file hashes in PostgreSQL to re-embed only new or modified files during re-runs.
* **Vector Database & Dynamic Query Builder:**
  * Powered by **PostgreSQL** with the `pgvector` extension.
  * Dynamic SQL Query Builder inside the FastAPI backend allows flexible, runtime JSONB metadata filtering (e.g., filter by region, category, or tags).
* **Strict Prompting Against Hallucinations:** Hardened system prompting strictly restricts the LLM to synthesized answers based solely on retrieved chunks.
* **FastAPI Backend & Streamlit Frontend:** Decoupled architecture featuring a REST API with automatic Swagger documentation and an extensible web user interface.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Backend Framework:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Vector Database:** PostgreSQL + `pgvector` (via Docker)
* **LLM & Embeddings Orchestration:** Ollama (`llama3.2:3b`, `nomic-embed-text`)
* **Database Driver:** `psycopg` (v3) with native `pgvector` registration
* **Parsing:** `python-frontmatter`