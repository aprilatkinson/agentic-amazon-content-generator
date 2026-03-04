# Agentic Amazon Content Generator

An **AI-powered agent system for generating compliant Amazon product listings and PDP content** using structured product evidence, validation guardrails, and an autonomous repair loop.

This project demonstrates an **agentic architecture** where specialized agents collaborate to produce **accurate, policy-compliant e-commerce content** from product pages and marketplace listings.

---

# Overview

Modern e-commerce content must satisfy multiple constraints simultaneously:

* SEO optimization
* marketplace compliance (Amazon listing rules)
* brand voice consistency
* factual accuracy
* multilingual output

This system solves the problem using a **multi-agent workflow** that extracts evidence-backed product facts and generates validated listing content.

---

# Key Features

• **Agentic workflow architecture**
• **Evidence-based fact extraction** from product pages
• **Amazon listing generation** (title, bullets, description)
• **Automated FAQ generation**
• **AI visibility snippets** for search assistants
• **Validation guardrails** to enforce compliance
• **Autonomous repair loop** when validation fails
• **Multilingual output support**
• **Webhook integration with n8n automation**

---

# System Architecture

The system is composed of specialized agents that cooperate sequentially.

```
User Input
   │
   ▼
Archivist Agent
(Data ingestion & scraping)
   │
   ▼
Fact Extraction Engine
(Evidence spans + structured facts)
   │
   ▼
Scribe Agent
(Content generation using extracted facts)
   │
   ▼
Guard Agent
(Validation of policy rules)
   │
   ▼
Planner Agent
(Decides whether to repair or finalize)
   │
   ▼
Final Output
```

Each generated claim must be traceable to **source evidence spans** extracted from the PDP or Amazon listing.

---

# Input

The system accepts the following inputs:

| Input                  | Description                       |
| ---------------------- | --------------------------------- |
| Amazon URL             | Existing product listing          |
| PDP URL                | Official product page             |
| Reviews (optional)     | Customer review text              |
| Keyword CSV (optional) | SEMrush / Helium10 keyword export |
| Output language        | Target output language            |

---

# Output

The API returns structured listing content:

```
amazon_title
bullets
description
faqs
ai_visibility_snippets
validation_report
product_facts
```

Each content element is validated before finalization.

---

# Validation Guardrails

The **Guard agent** enforces compliance rules including:

* Character limits
* No unsupported marketing claims
* No keyword stuffing
* Source traceability
* Numeric consistency
* FAQ sentence constraints

If validation fails, the system enters a **repair loop** until the content passes or the retry budget is exhausted.

---

# Technology Stack

Backend

* Python
* FastAPI
* BeautifulSoup (scraping)
* Regex fact extraction

AI Layer

* LLM content generation
* structured validation
* repair loop

Data / Retrieval

* Qdrant (vector search)
* RAG guideline retrieval

Automation

* n8n webhook integration

---

# Project Structure

```
agentic-amazon-content-generator
│
├── app
│   └── main.py
│
├── ui
│   └── client.html
│
├── PROJECT_STATE.md
├── CHANGELOG.md
└── README.md
```

---

# Running the System Locally

Install dependencies:

```
pip install -r requirements.txt
```

Start the API server:

```
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

---

# Test the API

Example request:

```
curl -X POST http://127.0.0.1:8001/generate \
-H "Content-Type: application/json" \
-d '{
  "amazon_url": "https://www.amazon.de/dp/B00008XV49",
  "pdp_url": "https://www.wmf.com/de/en/cutlery-set-boston-cromargan-30-piece-3201112492.html",
  "output_lang": "de"
}'
```

---

# Example Output

The API returns:

* Amazon optimized title
* 5 compliant bullet points
* Product description
* Customer FAQs
* AI search snippets
* Validation report

All generated statements are traceable to **source evidence**.

---

# Example Use Cases

E-commerce automation
Product catalog scaling
Marketplace listing generation
AI search visibility optimization
Content governance for large product catalogs

---

# Future Improvements

* Category-aware title builders
* PIM integration
* Advanced review theme analysis
* Multi-marketplace support
* Semantic fact extraction
---

# Author

April Atkinson
AI Systems & Agent Architecture
