CHANGELOG

Agentic Amazon Content Generator

2026-03-03 — Architecture Stabilization
Core Architecture

FastAPI backend confirmed

/generate (JSON mode) working

/generate_upload (multipart mode) working

Run store implemented with /runs/{run_id} endpoint

UI (client.html) fully functional

Knowledge Layer

Qdrant used for guidelines only

Product text is run-scoped (not persisted)

Validation pipeline integrated with:

No hype rule

Character limits

No new claim terms

Traceability

Input Expansion

Added support for:

product_info_file

helium10_csv

reviews_file

Helium 10 treated as volatile ranking signal

Reviews treated as FAQ discovery signal only

Output Improvements

FAQ rendering fixed in UI

Structured JSON output confirmed

Validation report exposed in frontend

Helium 10 preview rendering added

Known Gaps

Source text cleaner needs strengthening (promo leakage)

Helium 10 analyzer not fully optimized

Keyword-to-fact mapping incomplete

QDRANT_URL must be set to avoid validation runtime error

Future Planned Upgrades

Deterministic Helium 10 analyzer module

Keyword-to-fact semantic mapping

AI visibility snippet output

Structured claim ledger visualization

Section-level repair loop (LangGraph upgrade)

How to Use This File

Each time:

You change architecture

You add validation logic

You modify keyword logic

You alter generation rules

Add a new dated section.