# AIG230 NLP – One-Page Project Proposal

**Title**  
Multilingual RAG E-Commerce Customer Service Chatbot

**Problem**  
Ever dealt with a bot giving irrelevant answers? Spend 20 mins explaining, it says "transfer to human"—explain AGAIN, then rage quit... Happens daily on Shopify/Amazon. I want to use native Chinese in diverse Toronto (English+Chinese mixes common). Bots fail on mixed langs/rules (returns/shipping), causing wrong info, angry customers, refunds, overload. Sellers lose 20-30% sales (small ones hit hard). We wanna fix for multicultural sellers.

**Solution**  
Build RAG chatbot for multi-langs. Multilingual embedding (BAAI/bge-m3, 100+ langs) vectorizes queries, matches KB (FAQs/policies: Eng/Chi/Fr/Sp). + Chat history (LangChain) to LLM (Qwen2.5-7B) for natural user-lang replies. Features: citations, intent detect (refunds), code-switching. No fine-tuning.

**Objectives**  
1. 25% better than basic bots (relevance scores).  
2. Eng/Chi/Fr/Sp (+more), incl. mixed.  
3. Cut hallucinations w/ citations.  
4. Real-time web demo.

**Deliverables**  
- Gradio web chat (real-time/history/citations).  
- KB: 10-15 multi-lang docs.  
- Open-source GitHub repo.  
- Report: tests/comparisons/results.  
- 5-8 min live demo pres.

**Datasets**  
- **Main**: Multilingual Amazon Reviews (MARC), https://registry.opendata.aws/amazon-reviews-ml/ or Hugging Face "amazon-reviews-ml" (appliances/electronics: Eng/Fr/Sp/Chi). Gen FAQs/policies from reviews.  
- **Extra**: McAuley Lab Amazon Reviews'23; HF rjac/e-commerce-customer-support-qa (QA ex.). + 20-30 our synthetic Qs (e.g., "Fridge noisy, warranty?"). Retrieval-only.

**Demo**  
Live Gradio: Audience asks Eng/Chi/Fr/Sp/mixed (e.g., "Washer broken? Return? Shipping?"). Bot replies right lang/sources/context. vs. basic LLM. Laptop/cloud, 5-8 mins.

**Team:** Group 5 - Zhihuai Wang & Stephane Donald Njike Paho  
**Date:** Feb 11, 2026