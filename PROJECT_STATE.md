# PROJECT_STATE.md
# Agentic Amazon Content Generator
Current Architecture State (Authoritative)

1. Core Goal

Build a deterministic, compliant, AI-visibility-optimized Amazon content generator that:

Uses evidence-only product facts

Integrates Helium 10 raw CSV intelligently

Optionally uses review exports for FAQ discovery

Enforces Amazon + brand guidelines

Prevents hallucinated claims

Supports run-scoped processing

Exposes structured JSON via API

Can be orchestrated via n8n

2. Tech Stack (Current Implementation)

Backend:

FastAPI

Qdrant (guidelines only)

Local run store

Deterministic validators

Evidence span extraction

Frontend:

Static HTML form (/ui/client.html)

Multipart POST → /generate_upload

Endpoints:

/generate (JSON URL mode)

/generate_upload (multipart mode)

/runs/{run_id}

3. Knowledge Rules

Stored in Qdrant:

Amazon formatting rules

Character limits

Forbidden/hype terms

Compliance constraints

NOT stored in Qdrant:

Product text

Helium 10 data

Reviews

All product data is run-scoped only.

4. Input Modes

Supported inputs:

Amazon URL

PDP URL

product_info_file (new products)

helium10_csv (raw export)

reviews_file (optional)

Helium 10:

Used only for ranking + phrasing guidance

Never introduces new specs

Reviews:

Used for FAQ question discovery

Answers must map to product facts

5. Generation Principles

All output must:

Be fact-grounded

Map to extracted evidence spans

Respect char limits

Avoid hype

Avoid unsupported claims

Avoid keyword stuffing

Title:
Brand + Product + Core Attribute + Material + Set Size (if supported)

Bullets:
Fact → Benefit framing → Evidence-backed

Description:
Structured sections (facts + usage + material)

FAQs:

Evidence FAQs

Decision FAQs (explicit “Not specified” if unsupported)

Optional:
ai_visibility_snippets (LLM-answerable blocks)

6. Helium 10 Handling

System must auto-detect columns:

Keyword Phrase

Search Volume

IQ Score

Competition proxy

Normalization rules:

"1,846" → 1846

"-" → null

">500" → 500

Score formula:
score = log(1+volume) × IQ_weight × inverse_competition

Top keywords:

Influence prioritization

Influence phrasing

Never create new claims

7. Validation Strategy

Checks:

Character limits

Forbidden terms

No new claim terms

Claim traceability

Section completeness

Keyword stuffing detection

Repair loop:
Max 2 attempts (future LangGraph upgrade)

8. AI Visibility Strategy

Outputs must optimize for:

Human readability
Search visibility
LLM answer extraction

Content must be:

Structured

Dense but clean

Fact-first

Extractable

Non-ambiguous

9. Current Known Gaps

Source text cleaner needs strengthening

Helium10 analyzer module incomplete

Keyword-to-fact mapping needs refinement

Guidelines must load (QDRANT_URL required)

10. Project Direction

This is not just an AI writer.

It is a:
AI Visibility + Compliance Engine
with deterministic grounding and validation.

How To Use This

When opening a new chat, paste:

“Load the following as authoritative system state:”
Then paste this file.

If you want next, we can:

Upgrade Helium 10 analyzer design

Redesign generation logic for stronger AI visibility

Implement deterministic keyword-to-fact mapping

Improve source text cleaning pipeline

Your system is now architecturally serious.