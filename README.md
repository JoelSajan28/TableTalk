# TableTalk — Conversational Analytics for Your Excel Data

**TabulaRAG** is a web application that transforms your Excel files into an intelligent, query-ready database.  
Each worksheet becomes its own MongoDB collection, pivot tables are generated automatically, and relations between tables are detected.  
A built-in LLM-powered chatbot lets you ask natural language questions — even across multiple sheets — and provides answers backed by your original documents.

## ✨ Features
- **Excel to Database** — Each worksheet is ingested into MongoDB as a separate collection.
- **Auto Pivot Tables** — Automatically create useful pivot views for quick insights.
- **Relation Discovery** — Detect and store relationships between tables for joins.
- **Conversational Queries** — Use natural language to query your data via an LLM + RAG pipeline.
- **Multi-Table Reasoning** — Ask questions that require connecting data across collections.
- **Source Doclinks** — Trace answers back to the exact sheet, row, and cell in your original file.

## 🛠 Tech Stack
- **Frontend:** Next.js, Tailwind CSS, React Query
- **Backend:** Python (pandas, FastAPI), MongoDB, Redis
- **AI Layer:** LLM (OpenAI/Groq/Llama 3), Atlas Vector Search
- **Storage:** S3/R2 for file uploads
- **ETL:** pandas for schema inference, pivot creation, and relation detection

## 🚀 Quick Start
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/tabularag.git
   cd tabularag
   ```
2. Start services:
   ```bash
   docker-compose up --build
   ```
3. Open http://localhost:3000 and upload your first Excel file.

📌 Roadmap
 - Custom pivot table builder in the UI
 - User-confirmed relation editing
 - Support for CSV/Google Sheets
 - Fine-tuned SQL/NoSQL query generation model

Why TabulaRAG?
Because your spreadsheets deserve more than static cells — they deserve to talk back.
