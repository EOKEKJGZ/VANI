# 🏛️ Sahakaar Sathi VANI Platform
### *Voice-enabled Assistance for National Inclusion*

**Smart India Hackathon (SIH) — Problem Statement ID: 26088**  
**Ministry / Department:** Ministry of Cooperation / National Council for Cooperative Training (NCCT)  
**Theme:** Agriculture, FoodTech & Rural Development  

---

## 🌾 Overview

**Sahakaar Sathi VANI** is an AI-powered, multilingual cooperative governance and legal assistance platform designed for cooperative members, farmers, and rural stakeholders across India. It addresses critical information asymmetries regarding:
- **PMFBY 72-Hour Crop Loss Emergency Assistance**
- **PACS (Primary Agricultural Credit Societies) Model By-Laws & Governance**
- **Kisan Credit Card (KCC) Crop Loans & Debt Restructuring**
- **Cooperative Dispute Resolution & Grievance Redressal Mechanisms**
- **Central & State Welfare and Subsidy Schemes (3,300+ Verified Programs)**

---

## 🛠️ Architecture & Technology Stack

- **Retrieval-Augmented Generation (RAG)**:
  - **Dense Vector Retrieval**: Qdrant (`BAAI/bge-base-en-v1.5` embeddings, cosine metric)
  - **Lexical Keyword Search**: Custom In-Memory BM25 index with token stemming
  - **Hybrid Reciprocal Rank Fusion (RRF)**: Weighted dense (65%) + sparse (35%) scoring
  - **Reranking**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  - **Agentic Workflow**: LangGraph state machine with automatic intent classification, query decomposition, and context grounding
- **Backend**: FastAPI with async thread-safe execution and SQLite conversation persistence
- **Frontend**: Responsive Public-Service Ledger UI with Vanilla CSS, Web Speech API (STT & TTS), and accessibility font resizer (`A-`, `A`, `A+`)

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Clone and Setup
```bash
git clone https://github.com/EOKEKJGZ/VANI.git
cd VANI
```

### 3. Create & Activate Virtual Environment
```bash
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```

### 6. Run the Platform
```bash
python server.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 📄 License & Attribution
Developed for the **Smart India Hackathon (SIH)**. All rights reserved.
