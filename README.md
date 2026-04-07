# 🛒 Multilingual RAG E-Commerce Customer Support Chatbot

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange.svg)](https://gradio.app/)

A multilingual customer service chatbot for e-commerce using Retrieval-Augmented Generation (RAG). Supports English, Chinese, French, Spanish, and **code-switching** (mixed language queries).

**Course:** AIG230 Natural Language Processing - Final Project
**Team:** Group 5 (Zhihuai Wang & Stephane Donald Njike Paho)
**Institution:** Seneca Polytechnic

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🌐 **Multilingual Support** | English, 中文, Français, Español |
| 🔀 **Code-Switching** | Handle mixed language queries like "我的fridge坏了" |
| 😊 **Sentiment Analysis** | Detect angry customers for escalation |
| 📚 **Source Citations** | Always cite knowledge base sources |
| 🔔 **Auto-Escalation** | Transfer angry customers to human support |
| 🚀 **Fast Deployment** | Docker-ready, one-click setup |

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Evaluation](#evaluation)
- [Demo Guide](#demo-guide)
- [GenAI Declaration](#genai-declaration)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                         │
│         Gradio Web (7860)    │    FastAPI (8000)           │
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
                ▼                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Engine                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Language    │  │  Sentiment   │  │  Retrieval   │      │
│  │  Detection   │  │  Analysis    │  │  (ChromaDB)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         LLM (ZhipuAI GLM-4.7 / Qwen2.5-7B)           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              Knowledge Base (40+ Documents)                  │
│  Product FAQs (multilingual) │ Policies │ Code-Switching │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| Embedding Model | BAAI/bge-m3 (100+ languages) |
| Vector Database | ChromaDB |
| LLM | ZhipuAI GLM-4.7 (primary) / Qwen2.5-7B (fallback) |
| Frontend | Gradio 4.x |
| Backend API | FastAPI |
| Framework | LangChain |

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone <repo-url>
cd aig230_final_project_g5

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
echo "ZAI_API_KEY=your_zhipuai_key" > .env

# 4. Build vector store
python src/vector_store.py

# 5. Run Gradio demo
python app.py
```

Open http://localhost:7860 in your browser.

---

## 📦 Installation

### Prerequisites

- Python 3.11+
- 8GB RAM minimum
- API key (ZhipuAI or HuggingFace)

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
ZAI_API_KEY=your_zhipuai_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here  # Optional
EOF

# Build vector store (first time only)
python src/vector_store.py
```

### Docker Deployment

```bash
# Build and run
docker-compose up

# Or manually
docker build -t rag-chatbot .
docker run -p 7860:7860 -p 8000:8000 \
  -e ZAI_API_KEY=your_key \
  rag-chatbot
```

---

## 💻 Usage

### Gradio Web Interface

```bash
python app.py
```

**Example Queries:**
- "我的 fridge 噪音大，保修怎么走？" (Chinese)
- "How do I return a broken washer?" (English)
- "Mon téléphone ne s'allume pas, que faire ?" (French)
- "¿Cómo solicito una devolución?" (Spanish)
- "我的laptop screen flickering，warranty能cover吗？" (Code-switching)

### FastAPI REST API

```bash
python api.py
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Full chat with metadata |
| POST | `/chat/simple` | Simple answer only |
| GET | `/sources` | List KB sources |
| GET | `/languages` | Supported languages |

**Example API Call:**

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "我的fridge坏了"}
)

print(response.json())
# {
#   "answer": "...",
#   "sources": ["kb/FAQ-Appliances.md"],
#   "detected_lang": "中文",
#   "is_code_switching": true,
#   "sentiment": {"score": 0.0, "is_angry": false},
#   "escalated": false
# }
```

---

## 📁 Project Structure

```
aig230_final_project_g5/
├── app.py                 # Gradio frontend
├── api.py                 # FastAPI backend
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker config
├── docker-compose.yml     # Multi-service
├── README.md              # This file
│
├── src/
│   ├── rag_pipeline.py    # Core RAG engine
│   ├── vector_store.py    # ChromaDB builder
│   ├── sentiment.py       # Sentiment analysis
│   ├── rag_eval.py        # Evaluation module
│   ├── gen_amazon_kb.py   # KB generator
│   ├── data_gen.py        # Test data generator
│   └── update_kb.py       # KB updater
│
├── kb/                    # Knowledge Base
│   ├── FAQ-General-*.md   # General FAQs (EN/ZH/ES/FR)
│   ├── FAQ-Appliances-*.md
│   ├── FAQ-Clothing-*.md
│   ├── FAQ-Electronics-*.md
│   ├── FAQ-Furniture-*.md
│   ├── FAQ-Headphones-*.md
│   ├── FAQ-Kitchen-*.md
│   ├── FAQ-SmartHome-EN.md
│   ├── Refund-Policy.md
│   ├── Exchange-Policy.md
│   ├── Shipping-Policy.md
│   ├── Shipping-Rates.md
│   ├── Warranty-Terms.md
│   ├── Coupon-Policy.md
│   ├── CodeSwitch-ENZH.md
│   ├── CodeSwitch-FRES.md
│   └── Customer-Complaints.md
│
├── data/
│   ├── chroma_db/         # Vector database
│   ├── testset.csv        # Test dataset
│   ├── eval_results.json  # Evaluation results
│   └── eval_report.md     # Evaluation report
│
└── docs/
    ├── proposals/         # Project proposals
    ├── Final Report.md    # Final project report
    └── Progress Report.md # Progress report
```

---

## 📊 Evaluation

### Run Evaluation

```bash
python src/rag_eval.py
```

### Metrics

| Metric | Baseline | RAG | Improvement |
|--------|----------|-----|-------------|
| Answer Coverage | 62% | ~85% | +37% |
| Source Citation | 0% | ~90% | +90% |
| Hallucination Rate | 25% | <10% | -60% |

### Test Dataset

- **30 test questions** covering 10 language combinations
- **5 categories:** appliances, electronics, shipping, returns, payment
- **3 sentiment scenarios:** neutral, angry, escalation

---

## 🎬 Demo Guide (5-8 minutes)

### Demo Script

1. **Introduction (30 sec)**
   - Problem: Multilingual e-commerce support
   - Solution: RAG chatbot with code-switching

2. **Feature Demo (4 min)**
   - Show multilingual queries (EN, ZH, FR, ES)
   - Demonstrate code-switching detection
   - Show sentiment analysis (angry customer → escalation)
   - Display source citations

3. **Comparison (2 min)**
   - RAG vs Pure LLM comparison
   - Show hallucination reduction

4. **Q&A (1-2 min)**

### Key Talking Points

- **Novelty:** Code-switching support for multicultural markets
- **Practicality:** 20-30% sales loss reduction for small sellers
- **Technology:** BGE-M3 embeddings, ChromaDB, GLM-4.7 LLM

---

## 📝 GenAI Declaration

As per AIG230 course requirements, we declare the following GenAI usage:

| Component | AI Contribution | Human Contribution |
|-----------|-----------------|-------------------|
| **KB Documents** (kb/*.md) | Grammar check | Content writing, Q&A design |
| **RAG Pipeline** (src/rag_pipeline.py) | Debugging assistance | Architecture design, logic implementation |
| **Sentiment Module** (src/sentiment.py) | None | 100% human-written |
| **Gradio UI** (app.py) | Initial template | Customization, styling |
| **API** (api.py) | Code generation | Endpoint design |
| **Documentation** | Grammar check | All content |
| **Test Cases** | None | 100% human-curated |

**Estimated GenAI content: <5%** (primarily code debugging and grammar checking)

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `ZAI_API_KEY not found` | Create `.env` file with your key |
| `ChromaDB not found` | Run `python src/vector_store.py` |
| `Out of memory` | Use mock mode: `RAGEngine(use_mock=True)` |

### Environment Variables

```bash
# Required for ZhipuAI
ZAI_API_KEY=your_key_here

# Optional for HuggingFace
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## 📜 License

This project is for educational purposes (AIG230 course at Seneca Polytechnic).

---

## 👥 Authors

**Group 5**
- Zhihuai Wang
- Stephane Donald Njike Paho

**Instructor:** Prof. David Quispe
**Course:** AIG230 Natural Language Processing
**Term:** Winter 2026

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - RAG framework
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - Multilingual embeddings
- [ZhipuAI](https://open.bigmodel.cn/) - GLM-4.7 LLM
- [Gradio](https://gradio.app/) - Web interface
