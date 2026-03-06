# amazon_agent_clean/app/main.py
#
# Agentic Amazon Content Generator
# Stack: FastAPI + LangGraph + RAG (Qdrant) + OpenAI + n8n + simple HTML UI
#
# Fix Brief implemented (all) + robustness improvements based on review:
# 2.1 Hype validator uses word boundaries (no "best" match inside "Besteck")
# 2.2 Finish/color detection gated to PDP-only when pdp_len > 0
#     - Robust evidence mapping for gated facts (no fragile re-search)
#     - Dimension/weight/capacity gating kept; evidence is stable
# 2.3 Traceability improvements:
#     - hyphen/space normalization
#     - set_size numeric equivalence requires count context to avoid false positives
# 2.4 Language selection end-to-end (EN/DE/FR/ES)
# 2.5 n8n webhook POST on PASS + reduced payload by default
#     - Env toggle for full payload: N8N_SEND_FULL_PAYLOAD=1
# 2.6 Content quality + governance:
#     - Prompt encourages buyer decision factors + scenarios without invented specs
#     - FAQ answers 2–5 sentences (deterministically validated)
#     - ai_visibility_snippets validated (hype/claim terms + numeric hallucinations)
#     - Disallow ALL CAPS shouting (deterministically validated)
#
# Outstanding hardening implemented now:
# A) _sentence_count() made less fragile (abbreviations + decimals guarded)
# B) snippet numeric validator covers single digits with unit/context (still avoids noisy flags)

import os, re, uuid, math, csv, io, json, logging, hashlib
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from app.services.evidence import find_best_span

logger = logging.getLogger("amazon_agent")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Agentic Amazon Content Generator (LangGraph)")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "amazon_guidelines")
MAX_REPAIR_ATTEMPTS = 2   # up to 2 repair loops → 3 total LLM calls max

BASE_DIR = Path(__file__).resolve().parent.parent
UI_DIR   = BASE_DIR / "ui"
app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# n8n webhook (Fix 2.5)
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").strip()
N8N_SEND_FULL_PAYLOAD = os.getenv("N8N_SEND_FULL_PAYLOAD", "").strip() in ("1", "true", "TRUE", "yes", "YES")

# ---------------------------------------------------------------------------
# Run store
# ---------------------------------------------------------------------------

RUN_STORE: Dict[str, Dict[str, Any]] = {}
RUN_LOCK  = Lock()

def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"

def run_init(run_id: str, amazon_url: str, pdp_url: str) -> None:
    with RUN_LOCK:
        RUN_STORE[run_id] = {"run_id": run_id, "created_at": _now(),
                             "status": "RUNNING", "current_node": "init",
                             "event_log": [], "result": None}

def run_event(run_id: str, elf: str, stage: str, message: str,
              meta: Optional[Dict[str, Any]] = None) -> None:
    ev = {"ts": _now(), "elf": elf, "stage": stage,
          "message": message, "meta": meta or {}}
    with RUN_LOCK:
        if run_id in RUN_STORE:
            RUN_STORE[run_id]["event_log"].append(ev)
            RUN_STORE[run_id]["current_node"] = stage

def run_done(run_id: str, result: Dict[str, Any]) -> None:
    with RUN_LOCK:
        if run_id in RUN_STORE:
            RUN_STORE[run_id]["status"] = "DONE"
            RUN_STORE[run_id]["current_node"] = "finalize"
            RUN_STORE[run_id]["result"] = result

def event_log_snapshot(run_id: str) -> List[Dict[str, Any]]:
    with RUN_LOCK:
        return list((RUN_STORE.get(run_id) or {}).get("event_log") or [])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    amazon_url: Optional[str] = None
    pdp_url:    Optional[str] = None
    output_lang: Optional[str] = "en"

# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # inputs
    run_id:           str
    amazon_url:       str
    pdp_url:          str
    output_lang:      str
    source_mode:      str
    raw_source_text:  str
    helium_rows:      List[Dict[str, str]]
    reviews_text:     str
    # fetched / cleaned
    source_text:        str
    amazon_source_text: str
    pdp_clean_text:     str
    # RAG + analytics
    guidelines:      List[Dict[str, Any]]
    helium_analysis: Dict[str, Any]
    product_facts:   Dict[str, Any]
    review_themes:   List[Dict[str, Any]]
    blueprint:       Dict[str, Any]
    # generation + validation
    listing:            Dict[str, Any]
    validation_report:  Dict[str, Any]
    validation_status:  str
    # counters
    generation_count: int
    repair_count:     int
    next_action:      str
    # outputs / audit
    keyword_coverage_report: Dict[str, Any]
    claim_ledger:            List[Dict[str, Any]]
    google_doc_url:          str
    # errors
    runtime_errors: List[Dict[str, Any]]

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def get_openai_client() -> Optional[OpenAI]:
    k = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=k) if k else None

def get_qdrant_client() -> Optional[QdrantClient]:
    url = os.getenv("QDRANT_URL")
    if not url:
        return None
    k = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=k, check_compatibility=False) if k else QdrantClient(url=url, check_compatibility=False)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_str(x: Any, max_len: int = 20000) -> str:
    return ("" if x is None else str(x)).replace("\x00", "")[:max_len]

def _norm_for_match(s: str) -> str:
    if not s:
        return ""
    x = s.casefold()
    x = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212\-]+", " ", x)  # hyphen variants → space
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _ci(hay: str, needle: str) -> bool:
    n = (needle or "").strip()
    if not n:
        return False
    return _norm_for_match(needle) in _norm_for_match(hay or "")

def _digits(s: str) -> str:
    return "".join(re.findall(r"\d+", s or ""))

def _coerce_output_lang(lang: Optional[str]) -> str:
    x = (lang or "en").strip().lower()
    if x in ("en", "de", "fr", "es"):
        return x
    if x.upper() in ("EN", "DE", "FR", "ES"):
        return x.lower()
    return "en"

# --- Sentence counting (hardened) ---
_ABBR_TOKENS = [
    "e.g.", "i.e.", "etc.", "vs.", "approx.", "min.", "max.",
    "z.b.", "bzw.", "u.a.", "ca.", "d.h."
]

def _sentence_count(text: str) -> int:
    """
    Less fragile heuristic sentence counter:
    - protects common abbreviations (e.g., z.B., etc.) so they don't inflate counts
    - protects decimal points in numbers (e.g., 1.5) so they don't split sentences
    - splits on sentence end punctuation followed by whitespace
    Deterministic and cheap; designed to reduce false FAILs in FAQ validator.
    """
    t = (text or "").strip()
    if not t:
        return 0

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t)

    # Protect decimals: 1.5 -> 1<DOT>5
    t = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", t)

    # Protect known abbreviations
    # Case-insensitive replace of "." with "<DOT>" inside these tokens
    low = t.lower()
    for ab in _ABBR_TOKENS:
        if ab in low:
            # Replace in a case-preserving-ish way by doing regex on the literal token letters + dots
            pat = re.escape(ab)
            repl = ab.replace(".", "<DOT>")
            t = re.sub(pat, repl, t, flags=re.IGNORECASE)
            low = t.lower()

    # Split on sentence endings followed by whitespace (or end of string)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]

    # Cleanup placeholders
    cleaned = []
    for p in parts:
        p = p.replace("<DOT>", ".")
        if p:
            cleaned.append(p)

    # If text ends with no punctuation, still count as 1 sentence
    return len(cleaned) if cleaned else (1 if t else 0)

# ---------------------------------------------------------------------------
# Multilingual source cleaner (unchanged)
# ---------------------------------------------------------------------------

_PRICE_PATS = [
    r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:€|\$|£|¥|₹|CHF|USD|EUR|GBP|SEK|NOK|DKK|PLN|CZK|HUF|RON)\b",
    r"\b(?:€|\$|£|¥|₹|CHF|USD|EUR|GBP)\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\b",
]

_UNIV_NAV = [
    r"skip to (?:main )?content", r"back to top",
    r"add to (?:cart|basket|bag|wishlist)", r"buy now",
    r"out of stock", r"in stock",
    r"free (?:shipping|delivery|returns?)",
    r"\b(?:home|menu|search|login|sign ?in|sign ?up|register|logout|account|cart|basket)\b",
    r"©\s*\d{4}", r"all rights reserved", r"privacy policy",
    r"terms (?:of (?:use|service))?",
    r"cookie(?:s| policy| settings| preferences)?",
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}",
    r"tel(?:efon)?[:\s]+[\d\s()\-+]+",
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    r"https?://\S+",
    r"(?:www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/\S*)?",
    r"\b(?:qty|quantity|pcs|units?)\s*:\s*\d+",
    r"\b(?:sku|ean|isbn|asin|barcode)\s*[:\-]?\s*[A-Z0-9\-]+",
    r"\b\d+(?:\.\d+)?\s*(?:stars?|out of \d)",
    r"\bsold by\b", r"\bships from\b",
    r"\breturns?\s+(?:policy|within|accepted)\b",
]

_LANG_PATS: Dict[str, List[str]] = {
    "de": [r"\bjetzt\b",r"\bangebot\b",r"\bsparen\b",r"\brabatt\b",r"\bgratis\b",
           r"\bkostenlos\b",r"\buvp\b",r"\bpreis\b",r"\bversandkostenfrei\b",
           r"\blieferung\b",r"\bzum warenkorb\b",r"\bkaufen\b",r"\bbestellen\b",
           r"\bim angebot\b",r"\bdatenschutz\b",r"\bimpressum\b",r"\bagb\b",
           r"\bnewsletter\b",r"\banmelden\b",r"\bregistrieren\b",
           r"zum inhalt springen",r"\bwiderrufsrecht\b",r"\bwiderruf\b"],
    "en": [r"\bdeal\b",r"\bprime\b",r"\bblack friday\b",r"\bcyber monday\b",
           r"\bsale\b",r"\bdiscount\b",r"\bsave\b",r"\bpromo(?:tion)?\b",
           r"\bcoupon\b",r"\boffer\b",r"\blimited time\b",r"\bnewsletter\b",
           r"\bsubscribe\b",r"\bunsubscribe\b",r"\bterms and conditions\b",
           r"\bprivacy\b",r"skip to content"],
    "fr": [r"\bmaintenant\b",r"\boffre\b",r"\bpromotion\b",r"\bsoldes?\b",
           r"\blivraison gratuite\b",r"\bexpédition\b",r"\bachat\b",
           r"\bcommander\b",r"\bpanier\b",r"\beconomisez\b",r"\bremise\b",
           r"\bmentions légales\b",r"\bpolitique de confidentialité\b",
           r"\bcgv\b",r"\baller au contenu\b",r"\bnewsletter\b"],
    "it": [r"\badesso\b",r"\bofferta\b",r"\bpromo(?:zione)?\b",r"\bsconti?\b",
           r"\bspedizione gratuita\b",r"\bacquista\b",r"\bcarrello\b",
           r"\brisparmia\b",r"\bnewsletter\b",r"\biscriviti\b",
           r"\bpolitica sulla privacy\b",r"\bcondizioni di vendita\b",
           r"\bvai al contenuto\b"],
    "es": [r"\bahora\b",r"\boferta\b",r"\bpromo(?:ción)?\b",r"\bdescuento\b",
           r"\benvío gratis\b",r"\bcomprar\b",r"\bcarrito\b",r"\bahorrar\b",
           r"\bnewsletter\b",r"\bsuscribirse\b",r"\bpolítica de privacidad\b",
           r"\btérminos y condiciones\b",r"\bir al contenido\b"],
    "nl": [r"\bnu\b",r"\baanbieding\b",r"\bkorting\b",r"\bgratis verzending\b",
           r"\bbestellen\b",r"\bwinkelwagen\b",r"\bbesparen\b",r"\bnewsletter\b",
           r"\binschrijven\b",r"\bprivacybeleid\b",r"\balgemene voorwaarden\b",
           r"\bga naar inhoud\b"],
    "pt": [r"\bagora\b",r"\boferta\b",r"\bpromoção\b",r"\bdesconto\b",
           r"\bfrete grátis\b",r"\bcomprar\b",r"\bcarrinho\b",r"\bpoupar\b",
           r"\bnewsletter\b",r"\bpolitica de privacidade\b",
           r"\btermos e condições\b",r"\bir para o conteúdo\b"],
    "pl": [r"\bteraz\b",r"\boferta\b",r"\bpromocja\b",r"\bzniżka\b",
           r"\bdarmowa dostawa\b",r"\bkup\b",r"\bkoszyk\b",r"\boszczędzaj\b",
           r"\bnewsletter\b",r"\bpolityka prywatności\b",r"\bregulamin\b",
           r"\bprzejdź do treści\b"],
    "sv": [r"\bnu\b",r"\berbjudande\b",r"\brea\b",r"\brabatt\b",
           r"\bgratis frakt\b",r"\bköp\b",r"\bvagn\b",r"\bnewsletter\b",
           r"\bintegritetspolicy\b",r"\bvillkor\b"],
    "ja": [r"今すぐ",r"セール",r"割引",r"送料無料",r"カートに入れる",r"購入する",r"お買い得",r"クーポン"],
    "zh": [r"立即购买",r"促销",r"折扣",r"免费配送",r"加入购物车",r"优惠",r"限时",r"特价"],
    "ko": [r"지금 구매",r"할인",r"무료 배송",r"장바구니",r"프로모션"],
    "ar": [r"اشتر الآن",r"خصم",r"شحن مجاني",r"عرض",r"ترقية"],
}

def _detect_lang(text: str) -> str:
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        return detect(text[:2000]) or "en"
    except Exception:
        return "en"

def _pats_for_lang(lang: str) -> List[str]:
    pats = list(_UNIV_NAV) + list(_PRICE_PATS)
    pats += _LANG_PATS.get("en", [])
    if lang and lang != "en" and lang in _LANG_PATS:
        pats += _LANG_PATS[lang]
    return pats

def clean_source_text(text: str, lang: Optional[str] = None) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    detected = lang or _detect_lang(t)
    for p in _pats_for_lang(detected):
        try:
            t = re.sub(p, " ", t, flags=re.IGNORECASE)
        except re.error:
            continue
    return re.sub(r"\s+", " ", t).strip()[:12000]

# ---------------------------------------------------------------------------
# Source fetch
# ---------------------------------------------------------------------------

def fetch_page_text(url: str, timeout_s: int = 20) -> str:
    r = requests.get(url, headers={
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8,fr;q=0.7,es;q=0.6",
    }, timeout=timeout_s)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()

# ---------------------------------------------------------------------------
# Upload parsers
# ---------------------------------------------------------------------------

def parse_text_upload(file: UploadFile) -> str:
    raw = file.file.read()
    try:
        return re.sub(r"\s+", " ", raw.decode("utf-8", errors="ignore")).strip()
    finally:
        file.file.seek(0)

def parse_helium10_csv(file: UploadFile, max_rows: int = 500) -> List[Dict[str, str]]:
    raw = file.file.read()
    try:
        text = raw.decode("utf-8", errors="ignore")
    finally:
        file.file.seek(0)
    rows: List[Dict[str, str]] = []
    for i, row in enumerate(csv.DictReader(io.StringIO(text))):
        if i >= max_rows:
            break
        rows.append({str(k).strip(): (str(v).strip() if v is not None else "")
                     for k, v in row.items() if k})
    return rows

# ---------------------------------------------------------------------------
# Helium10 analyzer
# ---------------------------------------------------------------------------

def _norm_num(s: Any) -> Optional[float]:
    if s is None:
        return None
    x = str(s).strip()
    if not x or x == "-":
        return None
    if x.startswith(">"):
        x = x[1:].strip()
    x = x.replace(" ", "")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", x):
        x = x.replace(".", "")
    if re.fullmatch(r"\d{1,3}(,\d{3})+", x):
        x = x.replace(",", "")
    if re.fullmatch(r"\d+,\d+", x):
        x = x.replace(",", ".")
    x = re.sub(r"[^0-9.\-]", "", x)
    try:
        return float(x)
    except Exception:
        return None

def _pick(headers: List[str], cands: List[str]) -> Optional[str]:
    lows = [(h, h.lower()) for h in headers]
    for c in cands:
        cl = c.lower()
        for orig, low in lows:
            if cl in low:
                return orig
    return None

def analyze_helium10_rows(rows: List[Dict[str, str]], top_n: int = 15) -> Dict[str, Any]:
    if not rows:
        return {"detected_type": "none", "keyword_column": None, "volume_column": None,
                "iq_column": None, "competition_column": None,
                "top_keywords": [], "warnings": ["No rows provided."]}
    headers = list(rows[0].keys())
    kc = _pick(headers, ["keyword phrase","keyword","phrase","search term","query"])
    vc = _pick(headers, ["search volume","volume","sv"])
    ic = _pick(headers, ["iq score","cerebro iq","magnet iq"])
    cc = _pick(headers, ["competing products","competition","title density","sponsored asins"])
    warn: List[str] = []
    if not kc:
        warn.append("No keyword col; using first.")
        kc = headers[0] if headers else None
    if not vc:
        warn.append("No volume col.")
    if not ic:
        warn.append("No IQ col.")
    scored: List[Tuple[str, float, Dict[str, Any]]] = []
    for idx, r in enumerate(rows):
        term = ((r.get(kc) if kc else "") or "").strip()
        if not term:
            continue
        vol = _norm_num(r.get(vc)) if vc else None
        iq  = _norm_num(r.get(ic)) if ic else None
        comp= _norm_num(r.get(cc)) if cc else None
        sc  = 0.0
        if vol is not None:
            sc += math.log(1.0 + max(vol, 0.0))
        if iq is not None:
            sc += max(0.0, min(iq, 100000.0)) / 10000.0
        if comp is not None:
            sc *= 1.0 / (1.0 + comp / 10000.0)
        if sc == 0.0:
            sc = 1.0 / (1.0 + idx)
        scored.append((term, sc, {"volume": vol, "iq": iq, "competition": comp}))
    best: Dict[str, Tuple[float, Dict[str, Any], str]] = {}
    for term, sc, meta in scored:
        key = term.lower()
        if key not in best or sc > best[key][0]:
            best[key] = (sc, meta, term)
    ranked = sorted([(orig, sc, meta) for (_, (sc, meta, orig)) in best.items()],
                    key=lambda x: (-x[1], x[0].lower()))
    top = [{"term": t, "score": round(sc, 4),
            **{k: v for k, v in meta.items() if v is not None}}
           for t, sc, meta in ranked[:top_n]]
    return {"detected_type": "keyword_list", "keyword_column": kc,
            "volume_column": vc, "iq_column": ic, "competition_column": cc,
            "top_keywords": top, "warnings": warn}

# ---------------------------------------------------------------------------
# Review themes
# ---------------------------------------------------------------------------

_THEMES = [
    {"id":"cleaning","signals":["dishwasher","spül","spul","wash","clean","rinse"],
     "question":"How easy is it to clean?",
     "ans_stated":"The product info states it is dishwasher safe.",
     "ans_not":"Cleaning instructions are not specified in the provided product info."},
    {"id":"durability","signals":["scratch","kratzer","durable","rust","rost","tarnish","chip"],
     "question":"How durable is it over time?","ans_stated":None,
     "ans_not":"Durability details are not specified in the provided product info."},
    {"id":"gift","signals":["gift","geschenk","present","packaging","verpackung","box"],
     "question":"Is this suitable as a gift?","ans_stated":None,
     "ans_not":"Gift packaging details are not specified in the provided product info."},
    {"id":"weight_feel","signals":["heavy","light","schwer","leicht","balanced","feel","grip"],
     "question":"How does it feel to use?","ans_stated":None,
     "ans_not":"Weight and balance details are not specified in the provided product info."},
    {"id":"size_fit","signals":["size","fit","small","large","groß","gross","klein","dimension","length"],
     "question":"What size is it?","ans_stated":None,
     "ans_not":"Dimensions are not specified in the provided product info."},
]

def extract_review_themes(reviews_text: str, product_facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not reviews_text:
        return []
    rl = reviews_text.lower()
    has_dw = bool((product_facts.get("dishwasher_safe_phrase") or {}).get("evidence"))
    out: List[Dict[str, Any]] = []
    for t in _THEMES:
        if any(sig in rl for sig in t["signals"]):
            ans = t["ans_stated"] if t["id"] == "cleaning" and has_dw else t["ans_not"]
            out.append({"theme_id": t["id"], "question": t["question"],
                        "answer": ans, "source": "review_signals"})
    return out

# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

def _src_label(start: int, pdp_len: int) -> str:
    if pdp_len and start < pdp_len:
        return "pdp"
    return "amazon"

def _fact(value: str, src: str, pdp_len: int = 0) -> Dict[str, Any]:
    ev = find_best_span(src, value)
    if ev:
        d = ev.to_dict()
        d["source"] = _src_label(ev.start, pdp_len)
        return {"value": value, "evidence": [d]}
    return {"value": value, "evidence": []}

def _pdp_only(src: str, pdp_len: int) -> str:
    return src[:pdp_len] if pdp_len and pdp_len > 0 else src

def _fact_from_match(m: re.Match, segment: str, merged_src: str, pdp_len: int) -> Dict[str, Any]:
    value = segment[m.start():m.end()]
    return _fact(value, merged_src, pdp_len=pdp_len)

def extract_product_facts(src: str, pdp_len: int = 0) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    pdp_segment = _pdp_only(src, pdp_len)

    def _has_color_context(seg: str, term: str) -> bool:
        """
        Accept color terms only if they appear in a likely product-attribute context.
        This avoids nav/category false positives while still capturing real 'black' variants.
        """
        # Attribute words
        attr = r"(finish|farbe|color|colour|surface|oberfl[aä]che|beschichtung|optik|design|edition|variant|variante|ausf[uü]hrung)"
        # Coating/material words
        coat = r"(pvd|coating|coated|beschichtet|beschichtung|pulverbeschichtet|lackiert|anodized|anodisiert)"
        # Component/part words
        part = r"(handle|griff|grip|stiel|case|etui|box|koffer|stand|base|frame|rahmen|blade|klinge|messer|knives?)"

        ctx = rf"(?:{attr}|{coat}|{part})"

        pat1 = rf"\b{ctx}\b.{0,50}\b{re.escape(term)}\b"
        pat2 = rf"\b{re.escape(term)}\b.{0,50}\b{ctx}\b"
        return bool(re.search(pat1, seg, re.IGNORECASE) or re.search(pat2, seg, re.IGNORECASE))

    head = re.split(
        r"Skip to Content|Zum Inhalt springen|FREE SHIPPING|30 DAYS RIGHT OF RETURN|\|AMAZON_PAGE\|",
        src[:400], maxsplit=1
    )[0]
    name = head.strip()[:140]

    # --- Brand detection ---
    for brand in ["WMF"]:
        if re.search(r"\b" + re.escape(brand) + r"\b", src, re.IGNORECASE):
            facts["brand"] = _fact(brand, src, pdp_len)
            break

    # --- Series detection (optional; safe if none found) ---
    # Add only the series names you actually support / expect.
    SERIES = [
        "Palma",
        "Verona",
        "Boston",
        "Philadelphia",
        "Miami",
        "Nuova",
        "Flame",
    ]
    for s in SERIES:
        if re.search(r"\b" + re.escape(s) + r"\b", src, re.IGNORECASE):
            facts["series"] = _fact(s, src, pdp_len)
            break

    if name:
        facts["product_name_guess"] = {
            "value": name,
            "evidence": [{
                "text": name,
                "start": 0,
                "end": len(name),
                "source": _src_label(0, pdp_len)
            }]
        }
    
   
    m = re.search(r"\b(\d{1,3})\s*[- ]?(?:piece|teil|teilig|pcs?|parts?)\b", src, re.IGNORECASE)
    if m:
        num = m.group(1)
        facts["set_size_number"] = {
            "value": num,
            "evidence": [{"text": src[m.start():m.end()], "start": m.start(),
                          "end": m.end(), "source": _src_label(m.start(), pdp_len)}]
        }
        facts["set_size"] = {
            "value": f"{num}-piece",
            "evidence": [{"text": src[m.start():m.end()], "start": m.start(),
                          "end": m.end(), "source": _src_label(m.start(), pdp_len)}]
        }

    for mat in ["Cromargan","stainless steel","stainless","silicone","ceramic","aluminum",
                "aluminium","copper","titanium","cast iron","bamboo","wood","oak","walnut",
                "plastic","nylon","BPA"]:
        if re.search(r"\b" + re.escape(mat) + r"\b", src, re.IGNORECASE):
            facts[f"material_{mat.lower().replace(' ','_')}"] = _fact(mat, src, pdp_len)

    if re.search(r"\bdishwasher\s+safe\b", src, re.IGNORECASE):
        facts["dishwasher_safe_phrase"] = _fact("dishwasher safe", src, pdp_len)
    elif re.search(r"\bdishwasher\b", src, re.IGNORECASE):
        facts["dishwasher_keyword"] = _fact("dishwasher", src, pdp_len)

    if re.search(r"\bhand\s*wash\b", src, re.IGNORECASE):
        facts["hand_wash"] = _fact("hand wash", src, pdp_len)

    if re.search(r"\boven\s*safe\b|\boven-safe\b", src, re.IGNORECASE):
        facts["oven_safe"] = _fact("oven safe", src, pdp_len)

    if re.search(r"\binduc(?:tion|tionable)\b", src, re.IGNORECASE):
        facts["induction_compatible"] = _fact("induction", src, pdp_len)
    # --- DE dishwasher-safe variants (spülmaschinengeeignet / spülmaschinenfest) ---
    m = re.search(r"\bsp[üu]lmaschinen(?:geeignet|fest)\b", src, re.IGNORECASE)
    if m:
        facts["dishwasher_safe_phrase_de"] = _fact(m.group(0), src, pdp_len)

    # --- Cromargan 18/10 (capture full token) ---
    m = re.search(r"\bCromargan(?:®)?\s*18/10\b", src, re.IGNORECASE)
    if m:
        facts["material_cromargan_18_10"] = _fact(m.group(0), src, pdp_len)

    # --- Monobloc knife mention ---
    m = re.search(r"\bMonobloc[- ]?Messer\b", src, re.IGNORECASE)
    if m:
        facts["knife_monobloc"] = _fact(m.group(0), src, pdp_len)

    # --- “für 12 Personen” / “for 12 people” (only if explicitly stated) ---
    m = re.search(r"\b(?:f[üu]r|for)\s*(\d{1,2})\s*(?:personen|people|persons)\b", src, re.IGNORECASE)
    if m:
        facts["persons_count"] = _fact(m.group(0), src, pdp_len)
        
    # --- fallback: "12 Personen" without "für" ---
    m = re.search(r"\b(\d{1,2})\s*[- ]?(?:personen|people|persons)\b", src, re.IGNORECASE)
    if m and "persons_count" not in facts:
        facts["persons_count"] = _fact(m.group(0), src, pdp_len)

    # Gated to PDP segment if present — robust evidence via find_best_span()
    for pat, key in [
        (r"(\d[\d.,]*)\s*(?:cm|mm|inch|in\b|\")", "dimension_mention"),
        (r"(\d[\d.,]*)\s*(?:kg|g\b|gram|lb|lbs|oz)\b", "weight_mention"),
        (r"(\d[\d.,]*)\s*(?:ml|l\b|liter|litre|fl\.?\s*oz)\b", "capacity_mention")
    ]:
        m = re.search(pat, pdp_segment, re.IGNORECASE)
        if m:
            facts[key] = _fact_from_match(m, pdp_segment, src, pdp_len)

    # Finish/color detection PDP-only when PDP exists
    finish_terms = ["black","white","silver","gold","rose gold","matte","brushed","polished","chrome","copper"]

    for col in finish_terms:
        # For plain colors like "black/white/silver/gold" require finish/color context
        if col in ("black","white","silver","gold","rose gold"):
            if _has_color_context(pdp_segment, col):
                facts[f"finish_{col.replace(' ','_')}"] = _fact(col, src, pdp_len)
        else:
            # For surface descriptors like polished/matte/brushed we allow direct mention
            if re.search(r"\b" + re.escape(col) + r"\b", pdp_segment, re.IGNORECASE):
                facts[f"finish_{col.replace(' ','_')}"] = _fact(col, src, pdp_len)

    m = re.search(r"(\d+)\s*[- ]?(?:year|yr|month)\s*(?:warranty|guarantee)\b", src, re.IGNORECASE)
    if m:
        sp = src[m.start():m.end()]
        facts["warranty"] = {"value": sp, "evidence": [{"text": sp, "start": m.start(),
                             "end": m.end(), "source": _src_label(m.start(), pdp_len)}]}

    for compat in ["non-stick","nonstick","non stick","anti-slip","non-slip","heat resistant",
                   "heat-resistant","microwave safe","freezer safe","food safe","food-grade"]:
        if re.search(re.escape(compat), src, re.IGNORECASE):
            facts[f"compat_{compat.lower().replace(' ','_').replace('-','_')}"] = _fact(compat, src, pdp_len)

    return facts

# ---------------------------------------------------------------------------
# Messaging blueprint
# ---------------------------------------------------------------------------

def build_messaging_blueprint(product_facts, helium_top_keywords, review_themes, guidelines):
    confirmed = {k: rec["value"] for k, rec in product_facts.items()
                 if rec.get("evidence") and rec.get("value")}
    care: List[str] = []
    if "dishwasher_safe_phrase" in product_facts: care.append("dishwasher safe (stated)")
    if "hand_wash"             in product_facts: care.append("hand wash (stated)")
    if "oven_safe"             in product_facts: care.append("oven safe (stated)")
    if "induction_compatible"  in product_facts: care.append("induction compatible (stated)")
    return {
        "confirmed_facts":    confirmed,
        "material_summary":   [v for k, v in confirmed.items() if k.startswith("material_")],
        "care_summary":       care,
        "top_keywords":       [kw["term"] for kw in (helium_top_keywords or [])[:10] if kw.get("term")],
        "review_themes":      [t["theme_id"] for t in (review_themes or [])],
        "guideline_excerpts": [g["text"] for g in (guidelines or []) if g.get("text")][:3],
        "has_warranty":       "warranty" in product_facts,
        "has_set_size":       "set_size" in product_facts,
        "fact_keys":          list(confirmed.keys()),
    }

# ---------------------------------------------------------------------------
# RAG (hash embedding demo)
# ---------------------------------------------------------------------------

def _simple_hash_embedding(text: str, dim: int = 1536) -> List[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out: List[float] = []
    seed = h
    while len(out) < dim:
        seed = hashlib.sha256(seed).digest()
        out.extend([(b - 128) / 128.0 for b in seed])
    return out[:dim]

def retrieve_guidelines(query: str, k: int = 5) -> List[Dict[str, Any]]:
    client = get_qdrant_client()
    if client is None:
        return []
    try:
        v = _simple_hash_embedding(query)
        results = client.query_points(collection_name=QDRANT_COLLECTION, query=v, limit=k)
        return [{"chunk_id": (pt.payload or {}).get("chunk_id"),
                 "section":  (pt.payload or {}).get("section"),
                 "text":     (pt.payload or {}).get("text"),
                 "score":    pt.score}
                for pt in results.points]
    except Exception as e:
        logger.warning("Qdrant retrieval failed: %s", e)
        return []

# ---------------------------------------------------------------------------
# SEO keyword fallback sets
# ---------------------------------------------------------------------------

FALLBACK_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "generic": {
        "de": ["wmf", "küchenzubehör", "küchenhelfer"],
        "en": ["wmf", "kitchen accessories", "kitchenware"],
        "fr": ["wmf", "accessoires de cuisine", "ustensiles de cuisine"],
        "es": ["wmf", "accesorios de cocina", "utensilios de cocina"],
    },

    "cutlery": {
        "de": ["besteckset", "edelstahl besteck", "wmf besteckset"],
        "en": ["cutlery set", "stainless steel cutlery", "cutlery set for 12"],
        "fr": ["ménagère", "couverts inox", "service de couverts"],
        "es": ["cubertería", "cubiertos de acero inoxidable", "juego de cubiertos"],
    },

    "pots": {
        "de": ["topfset", "kochtopf", "edelstahl topf"],
        "en": ["pot set", "cooking pot", "stainless steel pot"],
        "fr": ["set de casseroles", "faitout", "casserole inox"],
        "es": ["juego de ollas", "olla", "olla de acero inoxidable"],
    },

    "pans": {
        "de": ["pfanne", "bratpfanne", "antihaft pfanne"],
        "en": ["frying pan", "nonstick pan", "skillet"],
        "fr": ["poêle", "poêle antiadhésive", "poêle à frire"],
        "es": ["sartén", "sartén antiadherente", "sartén para freír"],
    },

    # ...same idea for coffee_machines, kitchen_appliances, etc.
}

CATEGORY_HINTS = {
    "cutlery": ["besteck", "besteckset", "cutlery", "couverts", "cuberter"],
    "pots": ["topf", "kochtopf", "casserole", "pot", "faitout"],
    "pans": ["pfanne", "bratpfanne", "pan", "poêle", "sartén"],
    "coffee_machines": ["kaffeemaschine", "kaffeevollautomat", "coffee machine", "machine à café"],
    "kitchen_knives": ["messer", "messerset", "knife", "knives", "couteau", "cuchillo"],
    "kitchen_appliances": ["küchengerät", "toaster", "wasserkocher", "kettle", "blender", "food processor"],
    "kitchen_storage": ["aufbewahrung", "vorratsdose", "storage", "container", "boîte"],
    "tableware": ["geschirr", "teller", "schale", "tableware", "assiette", "vajilla"],
    "glasses_carafes": ["glas", "karaffe", "glass", "carafe", "verre", "jarra"],
    "baking_accessories": ["back", "baking", "blech", "moule", "hornear"],
    "bbq_accessories": ["bbq", "grill", "grillzange", "tongs", "barbecue"],
}
def detect_category_from_facts(product_facts: Dict[str, Any]) -> Optional[str]:
    name = ((product_facts.get("product_name_guess") or {}).get("value") or "").lower()
    if not name:
        return None

    for cat, hints in CATEGORY_HINTS.items():
        if any(h in name for h in hints):
            return cat

    return None

def pick_fallback_keywords(category: Optional[str], output_lang: str, k: int = 3) -> List[str]:
    lang = _coerce_output_lang(output_lang)

    pool = []
    if category:
        pool = (FALLBACK_KEYWORDS.get(category) or {}).get(lang) or []

    if not pool:
        pool = (FALLBACK_KEYWORDS.get("generic") or {}).get(lang) or []

    return pool[:max(0, min(k, len(pool)))]

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def _system_prompt_for_lang(output_lang: str) -> str:
    lang = _coerce_output_lang(output_lang)
    locale_guidance = {
        "en": "Write natural Amazon listing English. Avoid filler; be concrete but do not invent specs.",
        "de": "Schreibe natürliches Amazon-Deutsch (DE). Keine wörtlichen Übersetzungen; native Formulierungen.",
        "fr": "Écris un français naturel pour Amazon. Évite la traduction littérale; formulation native.",
        "es": "Escribe un español natural para Amazon. Evita traducciones literales; redacción nativa.",
    }[lang]

    return f"""You are an expert Amazon content writer.

OUTPUT LANGUAGE (mandatory): {lang}
{locale_guidance}

GOVERNANCE RULES (non-negotiable):
1. Only use facts from CONFIRMED FACTS for any factual claim/spec. Never invent specifications.
2. You MAY add scenario-based persuasion that does NOT introduce new specs:
   - Allowed: usage moments, table-setting scenarios, gifting occasion framing, audience framing.
   - Forbidden unless sourced: measurable performance, durability promises, certifications, guarantees, comparisons.
3. Forbidden hype terms (whole words only): best, perfect, guaranteed, #1, number one, world-class, amazing, incredible, unbeatable, revolutionary.
4. Do NOT use ALL CAPS shouting for attributes or headings.
5. Title: Brand/Product | Core Attribute | Set Size | Material — pipe-separated, max 200 chars.

OUTPUT JSON SHAPE (strict):
{{
  "amazon_title": "string",
  "bullets": [{{"text": "string", "citations": ["fact_key"]}}, ...],
  "description": "string",
  "faqs": [{{"q": "string", "a": "string", "citations": ["fact_key"]}}, ...],
  "ai_visibility_snippets": [{{"q": "string", "a": "string"}}]
}}

BULLETS (strict):
- exactly 5 bullets
- each bullet text max 255 chars
- each bullet MUST include citations only for confirmed facts used in that bullet (fact_key list).
- Avoid repeating the same information across bullets; each bullet should be meaningfully distinct.

DESCRIPTION (strict):
- plain text, max 2000 chars
- structure: Quick Facts / Use & Care / Material Details
- may include scenario language, but no invented specs
- avoid ALL CAPS headers

FAQs (strict):
- 6 to 8 pairs
- Each: {{"q":"...","a":"...","citations":["fact_key"]}}
- Answer length: 2–5 sentences per FAQ answer. Sentence 1: direct answer.
- Sentences 2–5: brief supporting context based ONLY on CONFIRMED FACTS (no new specs).
- Unknown answer: a must be exactly "Not specified in the provided product information.", citations = [].

AI VISIBILITY SNIPPETS:
- 4 to 6 micro Q&A, 1 sentence answers
- If not specified, say so (do not invent)
- No citations needed, but still must follow rule #1 (no invented specs)

Respond ONLY with valid JSON. No markdown fences, no preamble.
"""

def _build_prompt(bp, excerpt, output_lang: str, section_to_repair=None, prev_out=None, viol_summary=None):
    confirmed = bp.get("confirmed_facts", {})
    fact_keys = bp.get("fact_keys", list(confirmed.keys()))
    facts_table = "\n".join(f'  "{k}": "{v}"' for k, v in confirmed.items())

    lines = [
        f"=== OUTPUT LANGUAGE ===\n{_coerce_output_lang(output_lang)}\n",
        "\nSEO keyword integration rules:",
        f"- Output language: {_coerce_output_lang(output_lang)}",
        "- If keyword data exists: pick 2–3 phrases matching the output language and integrate naturally in title + bullets.",
        "- If no keyword data exists: use fallback phrases for the detected category and output language.",
        "- Avoid keyword stuffing. If category cannot be detected and no fallback exists: do not force keywords.",
        "=== CONFIRMED FACTS (cite keys in bullets and FAQs) ===",
        "{", facts_table, "}",
        f"\nAvailable fact_keys: {json.dumps(fact_keys)}\n",
        "=== MATERIALS ===",
        ", ".join(bp.get("material_summary", [])) or "(not detected)",
        "\n=== CARE ===",
        ", ".join(bp.get("care_summary", [])) or "(not detected)",
        "=== SEO KEYWORDS ===",
        "Primary keywords (Helium10 if available):",
        ", ".join(bp.get("top_keywords", [])) or "(none)",

        "Fallback keywords:",
        ", ".join(bp.get("fallback_keywords", [])) or "(none)",
        "\n=== GUIDELINES (RAG) ===",
        "\n".join(f"- {g}" for g in bp.get("guideline_excerpts", [])) or "(none)",
        "\n=== SOURCE EXCERPT (context; do not copy verbatim) ===",
        (excerpt or "")[:1500], "",
    ]

    if section_to_repair and prev_out and viol_summary:
        lines += [
            "=== REPAIR INSTRUCTIONS ===",
            f"FAILED SECTIONS: {section_to_repair}",
            f"VIOLATIONS: {viol_summary}",
            "Fix ONLY failing sections. Keep passing sections unchanged.",
            "\n=== PREVIOUS OUTPUT ===",
            json.dumps(prev_out, indent=2), "",
        ]

    lines.append("Generate the Amazon listing now.")
    return "\n".join(lines)

def call_llm_openai(client, bp, excerpt, output_lang: str, section_to_repair=None, prev_out=None, viol_summary=None):
    sys_prompt = _system_prompt_for_lang(output_lang)
    content = _build_prompt(bp, excerpt, output_lang, section_to_repair, prev_out, viol_summary)

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.3,
    )

    raw = (resp.choices[0].message.content or "").strip()
    clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"Bad JSON from LLM: {e}. Raw: {raw[:400]}")

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _btxt(b) -> str:
    return (b.get("text") or "") if isinstance(b, dict) else str(b)

def _faq_texts(faqs: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for x in (faqs or []):
        if not isinstance(x, dict):
            continue
        out.append((x.get("q","") or "") + " " + (x.get("a","") or ""))
    return out

def _snippet_texts(snips: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for x in (snips or []):
        if not isinstance(x, dict):
            continue
        out.append((x.get("q","") or "") + " " + (x.get("a","") or ""))
    return out

def val_hype(t, b, d, f, snips=None):
    patterns = [
        r"\bbest\b",
        r"\bperfect\b",
        r"\bguaranteed\b",
        r"\bno\.?\s*1\b",
        r"\bnumber\s+one\b",
        r"\bworld-?class\b",
        r"\bamazing\b",
        r"\bincredible\b",
        r"\bunbeatable\b",
        r"\brevolutionary\b",
    ]
    viol: List[str] = []
    txts = [t, d] + [_btxt(x) for x in (b or [])] + _faq_texts(f or []) + _snippet_texts(snips or [])
    for txt in txts:
        for p in patterns:
            if re.search(p, txt or "", flags=re.IGNORECASE):
                viol.append(f"Forbidden term matched: {p}")
    return {"status": "FAIL" if viol else "PASS", "violations": viol}

def val_limits(t, b, d, f):
    lim = {"title_max":200,"bullet_max":255,"bullets_exact_count":5,
           "description_max":2000,"faqs_max_count":10,"faq_q_max":200,"faq_a_max":500}
    viol: List[Dict[str, Any]] = []
    if len(t or "") > lim["title_max"]:
        viol.append({"field":"amazon_title","rule":"max_length","max":lim["title_max"],"actual":len(t or "")})
    if len(b or []) != lim["bullets_exact_count"]:
        viol.append({"field":"bullets","rule":"exact_count","expected":lim["bullets_exact_count"],"actual":len(b or [])})
    for i, x in enumerate(b or []):
        txt = _btxt(x)
        if len(txt) > lim["bullet_max"]:
            viol.append({"field":f"bullets[{i}].text","rule":"max_length","max":lim["bullet_max"],"actual":len(txt)})
    if len(d or "") > lim["description_max"]:
        viol.append({"field":"description","rule":"max_length","max":lim["description_max"],"actual":len(d or "")})
    if len(f or []) > lim["faqs_max_count"]:
        viol.append({"field":"faqs","rule":"max_count","max":lim["faqs_max_count"],"actual":len(f or [])})
    for i, x in enumerate(f or []):
        q = (x or {}).get("q","") or ""
        a = (x or {}).get("a","") or ""
        if len(q) > lim["faq_q_max"]:
            viol.append({"field":f"faqs[{i}].q","rule":"max_length","max":lim["faq_q_max"],"actual":len(q)})
        if len(a) > lim["faq_a_max"]:
            viol.append({"field":f"faqs[{i}].a","rule":"max_length","max":lim["faq_a_max"],"actual":len(a)})
    return {"status":"FAIL" if viol else "PASS","limits":lim,"violations":viol}

def val_faq_answer_sentences(faqs):
    viol: List[Dict[str, Any]] = []
    unknown = "Not specified in the provided product information."
    for i, x in enumerate(faqs or []):
        if not isinstance(x, dict):
            continue
        a = (x.get("a","") or "").strip()
        if not a:
            continue
        if a == unknown:
            continue
        sc = _sentence_count(a)
        if sc < 2 or sc > 5:
            viol.append({"field": f"faqs[{i}].a", "rule": "sentence_count_2_to_5", "actual": sc})
    return {"status":"FAIL" if viol else "PASS", "violations": viol}

def enforce_faq_sentence_count(listing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic patch:
    - For any FAQ answer that is not the exact UNKNOWN string,
      ensure 2–5 sentences by:
        - appending a neutral 2nd sentence if <2
        - truncating to first 5 sentences if >5
    This prevents random FAILs due to 1-sentence LLM answers.
    """
    unknown = "Not specified in the provided product information."
    faqs = (listing.get("faqs") or [])

    def _split_sentences(text: str) -> List[str]:
        # Use same protections as _sentence_count to avoid splitting decimals/abbreviations.
        t = (text or "").strip()
        if not t:
            return []
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", t)
        low = t.lower()
        for ab in _ABBR_TOKENS:
            if ab in low:
                pat = re.escape(ab)
                repl = ab.replace(".", "<DOT>")
                t = re.sub(pat, repl, t, flags=re.IGNORECASE)
                low = t.lower()
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
        return [p.replace("<DOT>", ".").strip() for p in parts if p.replace("<DOT>", ".").strip()]

    for i, x in enumerate(faqs):
        if not isinstance(x, dict):
            continue
        a = (x.get("a") or "").strip()
        if not a or a == unknown:
            continue

        sc = _sentence_count(a)

        if sc < 2:
            # Append a neutral sentence that adds no new product claims.
            a2 = a.rstrip()
            if not a2.endswith((".", "!", "?")):
                a2 += "."
            a2 += " Details are based on the provided product information."
            x["a"] = a2

        elif sc > 5:
            parts = _split_sentences(a)
            x["a"] = " ".join(parts[:5]).strip() if parts else a

    listing["faqs"] = faqs
    return listing

def val_citations(b, f, pf):
    valid = set(pf.keys())
    issues: List[Dict[str, Any]] = []
    for i, x in enumerate(b or []):
        if not isinstance(x, dict):
            continue
        for key in (x.get("citations") or []):
            if key not in valid:
                issues.append({"location":f"bullets[{i}]","cited_key":key,"issue":"key not in product_facts"})
            elif not (pf.get(key) or {}).get("evidence"):
                issues.append({"location":f"bullets[{i}]","cited_key":key,"issue":"no evidence spans"})
    for i, x in enumerate(f or []):
        if not isinstance(x, dict):
            continue
        for key in (x.get("citations") or []):
            if key not in valid:
                issues.append({"location":f"faqs[{i}]","cited_key":key,"issue":"key not in product_facts"})
            elif not (pf.get(key) or {}).get("evidence"):
                issues.append({"location":f"faqs[{i}]","cited_key":key,"issue":"no evidence spans"})
    return {"status":"FAIL" if issues else "PASS","citation_issues":issues}

def val_trace(t, b, d, f, pf, src, snips=None):
    snip_txt = " ".join([(x.get("q","")+" "+x.get("a","")) for x in (snips or []) if isinstance(x, dict)])
    out = " ".join([
        t or "",
        " ".join([_btxt(x) for x in (b or [])]),
        d or "",
        " ".join([(x.get("q","")+" "+x.get("a","")) for x in (f or []) if isinstance(x, dict)]),
        snip_txt
    ])
    issues: List[Dict[str, Any]] = []
    used: List[str] = []

    out_norm = _norm_for_match(out)
    src_norm = _norm_for_match(src or "")

    def has_set_context(number: str, text_norm: str) -> bool:
        ctx = r"(piece|pcs|parts|set|teil|teilig|teile|stück|stueck)"
        return bool(
            re.search(rf"(?:\b{re.escape(number)}\b.{{0,12}}\b{ctx}\b|\b{ctx}\b.{{0,12}}\b{re.escape(number)}\b)", text_norm)
        )

    for k, rec in (pf or {}).items():
        val = (rec or {}).get("value")
        evs = (rec or {}).get("evidence") or []
        if not val or not isinstance(val, str):
            continue

        used_here = False
        if k == "set_size":
            dv = _digits(val)
            if dv and has_set_context(dv, out_norm):
                used_here = True
        else:
            used_here = _ci(out, val)

        if used_here:
            used.append(k)

            if not evs:
                issues.append({"type":"UNSUPPORTED_FACT","fact_key":k,"fact_value":val,"message":"No evidence."})
                continue

            if k == "set_size":
                dv = _digits(val)
                ev_ok = False
                if dv:
                    for e in evs:
                        et = (e or {}).get("text","") or ""
                        if dv and has_set_context(dv, _norm_for_match(et)):
                            ev_ok = True
                            break
                    if not ev_ok and dv and has_set_context(dv, src_norm):
                        ev_ok = True
                if not ev_ok:
                    issues.append({"type":"TRACEABILITY_FAIL","fact_key":k,"fact_value":val,
                                   "message":"Set size number not supported by evidence/src with count context."})
                continue

            if not any(_ci((e or {}).get("text",""), val) for e in evs) and not _ci(src or "", val):
                issues.append({"type":"TRACEABILITY_FAIL","fact_key":k,"fact_value":val,"message":"Not in evidence/src."})

    return {"traceability":{"used_fact_keys":used,"issues":issues}}

def _term_in_text(term: str, text: str) -> bool:
    term = (term or "").strip()
    if not term:
        return False
    pat = r"\b" + re.escape(term) + r"\b"
    return bool(re.search(pat, text, flags=re.IGNORECASE))


def val_claim_terms_all_text(t, b, d, f, snips, src):
    out = " ".join([
        t or "",
        " ".join([_btxt(x) for x in (b or [])]),
        d or "",
        " ".join([(x.get("q","")+" "+x.get("a","")) for x in (f or []) if isinstance(x, dict)]),
        " ".join([(x.get("q","")+" "+x.get("a","")) for x in (snips or []) if isinstance(x, dict)])
    ])

    sl = (src or "")

    terms = [
        "premium","high quality","best","guaranteed","perfect","ergonomic",
        "rust proof","rustproof","non-slip","nonslip","warranty","guarantee",
        "gift box","ideal gift","bpa free","safe for","clinically","tested",
        "certified","approved"
    ]

    viol = [
        {"type":"NEW_CLAIM_TERM","term":x,"message":f"'{x}' in output not in source."}
        for x in terms
        if _term_in_text(x, out) and not _term_in_text(x, sl)
    ]

    return {"status":"FAIL" if viol else "PASS","violations":viol,"checked_terms":terms}

def val_snippets_numbers_in_source(snips, src):
    """
    Numeric hallucination check for snippets:
    - Any 2+ digit number in snippet answers must appear somewhere in source text
    - PLUS: any single digit 1–9 is checked only when paired with unit/context tokens
      (to avoid flagging list markers or generic '1')

    IMPORTANT: decimals like "1.83 kg" or "5,5 cm" must NOT trigger the single-digit rule.
    """
    issues: List[Dict[str, Any]] = []

    src_text = src or ""

    # --- 2+ digit numbers: strict presence in source (raw text is fine)
    src_digits_2p = set(re.findall(r"\b\d{2,}\b", src_text))

    # --- single-digit-with-unit rule (run on a "decimal-protected" normalized text)
    unit_alt = (
        r"(?:year|yr|month|pcs|piece|parts|set|teil|teilig|teile|stück|stueck|"
        r"cm|mm|kg|g|ml|l|liter|litre|inch|in\b|cups?)"
    )
    pat = rf"(?:(?P<num1>[1-9])\b.{{0,8}}\b(?P<unit1>{unit_alt})\b|\b(?P<unit2>{unit_alt})\b.{{0,8}}\b(?P<num2>[1-9])\b)"

    def _protect_decimals(t: str) -> str:
        # Replace any decimal-like number with a placeholder, e.g. 1.83, 5,5, 0.9
        # so it cannot be misread as a single digit "1" or "5" paired with a unit.
        return re.sub(r"\b[0-9]\s*[.,]\s*[0-9]+\b", "<DECIMAL>", t or "")

    # Build (digit, unit) pairs observed in the source (after normalization + decimal protection)
    src_single_pairs = set()
    src_norm = _protect_decimals(_norm_for_match(src_text))
    for m in re.finditer(pat, src_norm):
        num = m.group("num1") or m.group("num2")
        u = (m.group("unit1") or m.group("unit2") or "").strip()
        if num and u:
            src_single_pairs.add((num, u))

    for i, x in enumerate(snips or []):
        if not isinstance(x, dict):
            continue

        a = (x.get("a", "") or "")

        # 2+ digit numbers must appear in source anywhere
        for n in re.findall(r"\b\d{2,}\b", a):
            if n not in src_digits_2p:
                issues.append({
                    "field": f"ai_visibility_snippets[{i}].a",
                    "issue": "number_not_in_source",
                    "number": n
                })

        # single digits only when paired with unit/context tokens (on normalized + decimal-protected)
        a_norm = _protect_decimals(_norm_for_match(a))
        for m in re.finditer(pat, a_norm):
            num = m.group("num1") or m.group("num2")
            u = (m.group("unit1") or m.group("unit2") or "").strip()
            if num and u and (num, u) not in src_single_pairs:
                issues.append({
                    "field": f"ai_visibility_snippets[{i}].a",
                    "issue": "single_digit_with_unit_not_in_source",
                    "number": num,
                    "unit": u
                })

    return {"status": "FAIL" if issues else "PASS", "violations": issues}

def val_no_all_caps(t, b, d, f, snips):
    allow = {
    # brand / product / commerce identifiers
    "WMF", "BPA", "EAN", "ASIN", "SKU", "FAQ",
    # standards / compliance / marks
    "ISO", "DIN", "EN", "CE", "GS", "VDE", "ROHS", "EMC",
    # common tech acronyms that might appear even in kitchen context
    "USB", "LED", "LCD", "UV",
    # geo / locale acronyms that appear in listings
    "DE", "EU", "USA", "UK",
    }
    text = " ".join([
        t or "",
        " ".join([_btxt(x) for x in (b or [])]),
        d or "",
        " ".join([(x.get("q","")+" "+x.get("a","")) for x in (f or []) if isinstance(x, dict)]),
        " ".join([(x.get("q","")+" "+x.get("a","")) for x in (snips or []) if isinstance(x, dict)])
    ])
    viol: List[Dict[str, Any]] = []
    for w in re.findall(r"\b[A-ZÄÖÜ]{4,}\b", text):
        if w not in allow:
            viol.append({"rule":"no_all_caps", "word": w})
    return {"status":"FAIL" if viol else "PASS", "violations": viol}

STOPWORDS = {
    # German
    "eine","einer","eines","einem","einen","der","die","das","den","dem","des",
    "und","oder","für","mit","auf","im","in","am","an","aus","als","auch","ist",
    "sind","sein","zum","zur","von","bei","nicht","mehr","sehr","so","wie","dass",
    # English
    "the","and","for","with","from","that","this","your","you","are","is","in","on","to","of","a","an"
}

def val_stuffing(b):
    cnt: Dict[str, int] = {}
    for x in (b or []):
        for w in re.findall(r"\b\w{4,}\b", _btxt(x).lower()):
            if w in STOPWORDS:
                continue
            cnt[w] = cnt.get(w, 0) + 1

    stuffed = [w for w, c in cnt.items() if c >= 3]

    return {
        "status": "FAIL" if stuffed else "PASS",
        "repeated_words": stuffed,
        "violations": [f"'{w}' x{cnt[w]}" for w in stuffed]
    }

def run_validators(listing, pf, src):
    t = listing.get("amazon_title","") or ""
    b = listing.get("bullets",[]) or []
    d = listing.get("description","") or ""
    f = listing.get("faqs",[]) or []
    sn = listing.get("ai_visibility_snippets",[]) or []

    rh  = val_hype(t,b,d,f,snips=sn)
    rl  = val_limits(t,b,d,f)
    rfq = val_faq_answer_sentences(f)
    rc  = val_citations(b,f,pf)
    rt  = val_trace(t,b,d,f,pf,src,snips=sn)
    rcl = val_claim_terms_all_text(t,b,d,f,sn,src)
    rs  = val_stuffing(b)
    rnum= val_snippets_numbers_in_source(sn, src)
    rcap= val_no_all_caps(t,b,d,f,sn)

    merged = {
        "no_hype": rh,
        "char_limits": rl,
        "faq_sentence_count": rfq,
        "citations": rc,
        "no_new_claim_terms": rcl,
        "keyword_stuffing": rs,
        "snippets_numbers": rnum,
        "no_all_caps": rcap,
        **rt
    }

    status = "PASS"
    for ch in [rh, rl, rfq, rc, rcl, rs, rnum, rcap]:
        if ch.get("status") == "FAIL":
            status = "FAIL"
    if (rt.get("traceability") or {}).get("issues"):
        status = "FAIL"
    return merged, status

def _viol_summary(report):
    if not report:
        return "unspecified"
    parts: List[str] = []
    for k, v in report.items():
        if k == "traceability":
            issues = (v or {}).get("issues") or []
            if issues:
                parts.append(f"traceability:{json.dumps(issues[:2])}")
        elif isinstance(v, dict):
            viols = v.get("violations") or v.get("citation_issues") or []
            if viols:
                parts.append(f"{k}:{json.dumps(viols[:2])}")
    return " | ".join(parts) or "unspecified"

def _failing_sections(report):
    if not report:
        return "all sections"
    failing: set = set()
    for k in ["no_hype","no_new_claim_terms","keyword_stuffing","citations","faq_sentence_count","snippets_numbers","no_all_caps"]:
        ch = report.get(k) or {}
        if ch.get("violations") or ch.get("citation_issues"):
            if k == "snippets_numbers":
                failing.update(["ai_visibility_snippets"])
            else:
                failing.update(["bullets","faqs"])
    if (report.get("traceability") or {}).get("issues"):
        failing.update(["bullets","description","faqs","ai_visibility_snippets"])
    for v in ((report.get("char_limits") or {}).get("violations") or []):
        field = (v or {}).get("field","")
        if "title" in field:
            failing.add("amazon_title")
        elif "bullet" in field:
            failing.add("bullets")
        elif "description" in field:
            failing.add("description")
        elif "faq" in field:
            failing.add("faqs")
    return ", ".join(sorted(failing)) or "all sections"

# ---------------------------------------------------------------------------
# Coverage + claim ledger
# ---------------------------------------------------------------------------

def kw_coverage(top_kw, out):
    terms = [k.get("term","") for k in (top_kw or []) if k.get("term")]
    if not terms:
        return {"terms_checked":0,"terms_hit":0,"coverage_pct":0.0,"hits":[],"misses":[]}
    hits   = [t for t in terms if     _ci(out, t)]
    misses = [t for t in terms if not _ci(out, t)]
    return {"terms_checked":len(terms),"terms_hit":len(hits),
            "coverage_pct":round(len(hits)/len(terms)*100.0,2),"hits":hits,"misses":misses}

def claim_ledger(pf, out):
    return [{"fact_key":k,"value":(v or {}).get("value"),
             "evidence_source":((v or {}).get("evidence") or [{}])[0].get("source")}
            for k,v in (pf or {}).items()
            if (v or {}).get("value") and _ci(out,(v or {}).get("value"))]

# ---------------------------------------------------------------------------
# n8n persistence
# ---------------------------------------------------------------------------

def send_to_n8n(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not N8N_WEBHOOK_URL:
        return {"skipped": True, "reason": "N8N_WEBHOOK_URL not set"}
    try:
        r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=(10, 10))
        return {"ok": True, "status_code": r.status_code, "response": (r.text or "")[:500]}
    except requests.exceptions.ReadTimeout:
        # n8n acknowledged the request but is processing — this is expected
        return {"ok": True, "status_code": 200, "response": "accepted (timeout expected)"}
    except Exception as e:
        return {"ok": False, "error": repr(e)}

# ===========================================================================
# LANGGRAPH NODES
# ===========================================================================

def node_ingest(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    run_event(run_id,"archivist","ingest","Inputs ingested",{
        "source_mode":state.get("source_mode"),
        "output_lang":state.get("output_lang"),
        "helium_rows":len(state.get("helium_rows") or []),
        "has_reviews":bool(state.get("reviews_text")),
        "has_amazon_url":bool(state.get("amazon_url")),
        "has_raw_source_text": bool(state.get("raw_source_text")),
    })
    amazon_source_text = ""
    amazon_url = state.get("amazon_url") or ""
    if amazon_url:
        try:
            run_event(run_id,"archivist","amazon_fetch",f"Fetching Amazon listing: {amazon_url}")
            raw_az = fetch_page_text(amazon_url)
            amazon_source_text = clean_source_text(raw_az)
            run_event(run_id,"archivist","amazon_fetch_ok",f"{len(amazon_source_text)} chars")
        except Exception as e:
            run_event(run_id,"archivist","amazon_fetch_error",repr(e))
    return {"amazon_source_text": amazon_source_text}

def node_clean_source(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    run_event(run_id,"archivist","clean_source","Cleaning and merging source text")
    pdp = clean_source_text(state.get("raw_source_text") or "")
    az  = state.get("amazon_source_text") or ""
    merged = (pdp + " |AMAZON_PAGE| " + az).strip() if az else pdp
    merged = merged[:12000]
    run_event(run_id,"archivist","clean_source_ok",
              f"{len(merged)} chars",{"pdp":len(pdp),"amazon":len(az)})
    bp = dict(state.get("blueprint") or {})
    bp["_pdp_len"] = len(pdp)
    return {"source_text": merged, "blueprint": bp, "pdp_clean_text": pdp}

def node_retrieve_guidelines(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    run_event(run_id,"archivist","retrieve_guidelines","Querying Qdrant RAG")
    errors = list(state.get("runtime_errors") or [])
    guidelines: List[Dict[str, Any]] = []
    try:
        guidelines = retrieve_guidelines("Amazon listing rules title bullets description FAQs claims", k=5)
        if not guidelines and not os.getenv("QDRANT_URL"):
            errors.append({"stage":"retrieve_guidelines","error":"QDRANT_URL not set; skipped."})
        run_event(run_id,"archivist","retrieve_guidelines_ok",f"{len(guidelines)} chunks")
    except Exception as e:
        errors.append({"stage":"retrieve_guidelines","error":repr(e)})
        run_event(run_id,"archivist","retrieve_guidelines_error",repr(e))
    return {"guidelines":guidelines,"runtime_errors":errors}

def node_analyze_helium10(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    rows = state.get("helium_rows") or []
    if not rows:
        run_event(run_id,"archivist","helium10_skip","No Helium10 rows")
        return {"helium_analysis":{"detected_type":"none","top_keywords":[],"warnings":[]}}
    run_event(run_id,"archivist","helium10_analyze",f"Analyzing {len(rows)} rows")
    errors = list(state.get("runtime_errors") or [])
    try:
        analysis = analyze_helium10_rows(rows, top_n=15)
        run_event(run_id,"archivist","helium10_analyze_ok",f"{len(analysis.get('top_keywords',[]))} keywords ranked")
        return {"helium_analysis":analysis}
    except Exception as e:
        errors.append({"stage":"analyze_helium10","error":repr(e)})
        run_event(run_id,"archivist","helium10_analyze_error",repr(e))
        return {"helium_analysis":{"detected_type":"error","top_keywords":[],"warnings":[repr(e)]},
                "runtime_errors":errors}

def node_extract_facts(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    src = state.get("source_text") or ""
    errors = list(state.get("runtime_errors") or [])
    if not src:
        errors.append({"stage":"extract_facts","error":"No source text available."})
        run_event(run_id,"archivist","extract_facts_skip","No source text")
        return {"product_facts":{},"runtime_errors":errors}
    pdp_len = int((state.get("blueprint") or {}).get("_pdp_len", 0))
    run_event(run_id,"archivist","extract_facts","Extracting evidence-backed facts")
    try:
        facts = extract_product_facts(src, pdp_len=pdp_len)
        run_event(run_id,"archivist","extract_facts_ok",f"{len(facts)} facts",{"keys":list(facts.keys())})
        return {"product_facts":facts}
    except Exception as e:
        errors.append({"stage":"extract_facts","error":repr(e)})
        run_event(run_id,"archivist","extract_facts_error",repr(e))
        return {"product_facts":{},"runtime_errors":errors}

def node_extract_review_themes(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    rt = state.get("reviews_text") or ""
    if not rt:
        run_event(run_id,"archivist","reviews_skip","No reviews")
        return {"review_themes":[]}
    run_event(run_id,"archivist","reviews_analyze","Scanning reviews")
    themes = extract_review_themes(rt, state.get("product_facts") or {})
    run_event(run_id,"archivist","reviews_ok",f"{len(themes)} themes")
    return {"review_themes":themes}

def node_build_blueprint(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    run_event(run_id,"scribe","build_blueprint","Building messaging blueprint")
    errors = list(state.get("runtime_errors") or [])
    try:
        pdp_len = (state.get("blueprint") or {}).get("_pdp_len", 0)
        bp = build_messaging_blueprint(
            state.get("product_facts") or {},
            (state.get("helium_analysis") or {}).get("top_keywords") or [],
            state.get("review_themes") or [],
            state.get("guidelines") or []
        )
         # --- Fix 2: fallback keywords when Helium10 keywords are missing ---
        category = detect_category_from_facts(state.get("product_facts") or {})
        fallback_keywords = pick_fallback_keywords(category, state.get("output_lang") or "en", k=3)

        # Only use fallback if Helium list is empty
        if not (bp.get("top_keywords") or []):
            bp["fallback_keywords"] = fallback_keywords
        else:
            bp["fallback_keywords"] = []
        bp["_pdp_len"] = pdp_len
        run_event(run_id,"scribe","build_blueprint_ok","Blueprint ready")
        return {"blueprint":bp}
    except Exception as e:
        errors.append({"stage":"build_blueprint","error":repr(e)})
        run_event(run_id,"scribe","build_blueprint_error",repr(e))
        return {"blueprint":{},"runtime_errors":errors}
# --- helper for deterministic title building ---
def _dedupe_title_parts(parts: List[str]) -> List[str]:
    out = []
    seen_norm = set()
    for p in [x.strip() for x in parts if x and x.strip()]:
        n = _norm_for_match(p)
        if not n or n in seen_norm:
             continue
        # skip if already contained
        if any(n in _norm_for_match(prev) for prev in out):
            continue
        # remove shorter previous part
        out = [prev for prev in out if _norm_for_match(prev) not in n]
        out.append(p)
        seen_norm.add(n)
    return out


def node_generate_content(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    errors = list(state.get("runtime_errors") or [])
    gc = int(state.get("generation_count") or 0)
    rc = int(state.get("repair_count") or 0)
    is_repair = (gc > 0) and (state.get("validation_status") == "FAIL") and bool(state.get("validation_report"))
    new_rc = rc + (1 if is_repair else 0)

    run_event(
        run_id,
        "scribe",
        "repair_content" if is_repair else "generate_content",
        f"Repair {new_rc}/{MAX_REPAIR_ATTEMPTS}" if is_repair else "Initial generation",
        {"generation_count": gc + 1, "repair_count": new_rc, "output_lang": state.get("output_lang")}
    )

    empty = {"amazon_title":"","bullets":[],"description":"","faqs":[],"ai_visibility_snippets":[]}
    llm = get_openai_client()
    if not llm:
        errors.append({"stage":"generate_content","error":"OPENAI_API_KEY not set."})
        return {"listing": empty, "runtime_errors": errors, "generation_count": gc + 1, "repair_count": new_rc}

    pf = state.get("product_facts") or {}
    if not pf:
        errors.append({"stage":"generate_content","error":"No product facts; cannot generate safely."})
        return {"listing": empty, "runtime_errors": errors, "generation_count": gc + 1, "repair_count": new_rc}

    try:
        prev_report = state.get("validation_report") if is_repair else None
        parsed = call_llm_openai(
            llm,
            state.get("blueprint") or {},
            (state.get("source_text") or "")[:1500],
            output_lang=state.get("output_lang") or "en",
            section_to_repair=_failing_sections(prev_report) if is_repair else None,
            prev_out=state.get("listing") if is_repair else None,
            viol_summary=_viol_summary(prev_report) if is_repair else None,
        )

        for k in ["amazon_title","bullets","description","faqs","ai_visibility_snippets"]:
            parsed.setdefault(k, [] if k in ["bullets","faqs","ai_visibility_snippets"] else "")

        # --- Fix 3: deterministic Amazon title builder (brand + optional series) ---
        pf = state.get("product_facts") or {}
        if pf:
            brand  = ((pf.get("brand") or {}).get("value") or "").strip()
            series = ((pf.get("series") or {}).get("value") or "").strip()

            name = ((pf.get("product_name_guess") or {}).get("value") or "").strip()
            set_size = ((pf.get("set_size") or {}).get("value") or "").strip()

            material = ""
            # Prefer the specific Cromargan 18/10 fact if available
            if pf.get("material_cromargan_18_10"):
                material = (pf["material_cromargan_18_10"].get("value") or "").strip()
            else:
                for k, v in pf.items():
                    if k.startswith("material_"):
                        material = v.get("value")
                        break

            parts = []
            if brand:
                parts.append(brand)
            # only include series if present
            if series:
                parts.append(series)
            
            # keep the “name guess” if you want; optional:
            # If name already contains brand/series, you can still keep it.
            if name:
                parts.append(name)

            if set_size:
                parts.append(set_size)

            if material:
                parts.append(material)

            if parts:
                parts = _dedupe_title_parts(parts)
                parsed["amazon_title"] = " | ".join(parts)[:200]

        run_event(run_id,"scribe","generate_ok","LLM output received",{
            "bullets": len(parsed.get("bullets") or []),
            "faqs": len(parsed.get("faqs") or []),
            "snippets": len(parsed.get("ai_visibility_snippets") or []),
            "output_lang": state.get("output_lang")
        })
        return {"listing": parsed, "generation_count": gc + 1, "repair_count": new_rc}
    except Exception as e:
        errors.append({"stage":"generate_content","error":repr(e)})
        run_event(run_id,"scribe","generate_error",repr(e))
        return {"listing": empty, "runtime_errors": errors, "generation_count": gc + 1, "repair_count": new_rc}

def node_validate_output(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    run_event(run_id, "guard", "validate_output", "Running validators")

    # Deterministic FAQ sentence-count patch BEFORE validation
    patched_listing = enforce_faq_sentence_count(dict(state.get("listing") or {}))

    report, status = run_validators(
        patched_listing,
        state.get("product_facts") or {},
        state.get("source_text") or ""
    )
    run_event(run_id, "guard", "validate_ok", f"Status: {status}", {"status": status})
    return {"listing": patched_listing, "validation_report": report, "validation_status": status}
    

def node_decide_next_action(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    status = state.get("validation_status") or "FAIL"
    rc     = int(state.get("repair_count") or 0)
    if status == "PASS":
        action, reason = "finalize", "validation_passed"
    elif rc < MAX_REPAIR_ATTEMPTS:
        action, reason = "repair", "failed_validation_budget_available"
    else:
        action, reason = "finalize", "failed_validation_budget_exhausted"
    run_event(run_id,"planner","decide_next_action",f"Next: {action}",{
        "validation_status": status,
        "repair_count": rc,
        "max_repair_attempts": MAX_REPAIR_ATTEMPTS,
        "reason": reason
    })
    return {"next_action": action}

def node_finalize(state: AgentState) -> Dict[str, Any]:
    run_id = state["run_id"]
    run_event(run_id,"guard","finalize","Assembling final output")

    listing = dict(state.get("listing") or {})
    themes  = state.get("review_themes") or []
    errors  = list(state.get("runtime_errors") or [])

    # append review-driven FAQs (evidence-only; cap at 10)
    if themes and listing.get("faqs") is not None:
        listing.setdefault("faqs", [])
        existing = {x.get("q","").lower() for x in listing["faqs"] if isinstance(x, dict)}
        for t in themes:
            if len(listing["faqs"]) >= 10:
                break
            if t["question"].lower() not in existing:
                listing["faqs"].append({"q": t["question"], "a": t["answer"], "citations": []})

    bullets_with_cit = listing.get("bullets") or []
    bullets_flat = [_btxt(x) for x in bullets_with_cit]

    status = state.get("validation_status") or "FAIL"
    report = dict(state.get("validation_report") or {})
    report["runtime_errors"] = errors

    out_text = " ".join([
        listing.get("amazon_title","") or "",
        " ".join(bullets_flat),
        listing.get("description","") or "",
        " ".join([(x.get("q","")+" "+x.get("a","")) for x in (listing.get("faqs") or []) if isinstance(x, dict)]),
        " ".join([(x.get("q","")+" "+x.get("a","")) for x in (listing.get("ai_visibility_snippets") or []) if isinstance(x, dict)]),
    ])

    ha     = state.get("helium_analysis") or {}
    cov    = kw_coverage(ha.get("top_keywords") or [], out_text)
    ledger = claim_ledger(state.get("product_facts") or {}, out_text)

    result = {
        "run_id":       run_id,
        "amazon_url":   state.get("amazon_url") or "",
        "pdp_url":      state.get("pdp_url") or "",
        "source_mode":  state.get("source_mode") or "",
        "output_lang":  state.get("output_lang") or "en",
        "amazon_title": listing.get("amazon_title",""),
        "bullets":      bullets_flat,
        "description":  listing.get("description",""),
        "faqs":         listing.get("faqs", []),
        "ai_visibility_snippets": listing.get("ai_visibility_snippets", []),
        "listing_with_citations": listing,
        "validation_status":  status,
        "validation_report":  report,
        "generation_count":   int(state.get("generation_count") or 0),
        "repair_attempts":    int(state.get("repair_count") or 0),
        "guidelines_used":    state.get("guidelines") or [],
        "product_facts":      state.get("product_facts") or {},
        "messaging_blueprint": {k:v for k,v in (state.get("blueprint") or {}).items() if k != "_pdp_len"},
        "review_themes":      themes,
        "pdp_excerpt":        (state.get("pdp_clean_text") or "")[:500],
        "amazon_excerpt":     (state.get("amazon_source_text") or "")[:500],
        "helium10_rows_preview": (state.get("helium_rows") or [])[:20],
        "helium10_analysis":     ha,
        "keyword_coverage_report": cov,
        "claim_ledger":  ledger,
        "event_log":     event_log_snapshot(run_id),
        "google_doc_url": state.get("google_doc_url") or "",
    }

    # Reduced payload by default — ALWAYS send to n8n (lab requirement)
    run_event(run_id, "archivist", "n8n_push", "Posting to n8n webhook")

    base_payload = {
        "run_id": result["run_id"],
        "created_at": (RUN_STORE.get(run_id) or {}).get("created_at") or _now(),
        "amazon_url": result["amazon_url"],
        "pdp_url": result["pdp_url"],
        "output_lang": result["output_lang"],
        "amazon_title": result["amazon_title"],
        "bullets": result["bullets"],
        "description": result["description"],
        "faqs": result["faqs"],
        "ai_visibility_snippets": result["ai_visibility_snippets"],
        "validation_status": result["validation_status"],  # PASS or FAIL
        "validation_report_summary": {
            "trace_issues": len(((result.get("validation_report") or {}).get("traceability") or {}).get("issues") or []),
            "citation_issues": len(((result.get("validation_report") or {}).get("citations") or {}).get("citation_issues") or []),
            "runtime_errors": len((result.get("validation_report") or {}).get("runtime_errors") or []),
        },
        "event_log_count": len(result.get("event_log") or []),
    }

    if N8N_SEND_FULL_PAYLOAD:
            base_payload["listing_with_citations"] = result["listing_with_citations"]
            base_payload["event_log"] = result["event_log"]
            base_payload["validation_report"] = result["validation_report"]

    n8n_status = send_to_n8n(base_payload)
    run_event(run_id, "archivist", "n8n_push_done", "n8n push finished", n8n_status)
    result["n8n_status"] = n8n_status

    if not n8n_status.get("ok") and not n8n_status.get("skipped"):
        errors = (result.get("validation_report") or {}).get("runtime_errors") or []
        errors.append({"stage": "n8n_push", "error": n8n_status.get("error", "unknown")})
        result["validation_report"]["runtime_errors"] = errors
    run_done(run_id, result)
    return result


# ---------------------------------------------------------------------------
# Conditional edge routing
# ---------------------------------------------------------------------------

def route_after_decision(state: AgentState) -> str:
    return state.get("next_action") or "finalize"

# ---------------------------------------------------------------------------
# Build and compile the LangGraph
# ---------------------------------------------------------------------------

def build_agent_graph():
    g = StateGraph(AgentState)
    g.add_node("ingest",                node_ingest)
    g.add_node("clean_source",          node_clean_source)
    g.add_node("retrieve_guidelines",   node_retrieve_guidelines)
    g.add_node("analyze_helium10",      node_analyze_helium10)
    g.add_node("extract_facts",         node_extract_facts)
    g.add_node("extract_review_themes", node_extract_review_themes)
    g.add_node("build_blueprint",       node_build_blueprint)
    g.add_node("generate_content",      node_generate_content)
    g.add_node("validate_output",       node_validate_output)
    g.add_node("decide_next_action",    node_decide_next_action)
    g.add_node("finalize",              node_finalize)

    g.set_entry_point("ingest")
    g.add_edge("ingest",                "clean_source")
    g.add_edge("clean_source",          "retrieve_guidelines")
    g.add_edge("retrieve_guidelines",   "analyze_helium10")
    g.add_edge("analyze_helium10",      "extract_facts")
    g.add_edge("extract_facts",         "extract_review_themes")
    g.add_edge("extract_review_themes", "build_blueprint")
    g.add_edge("build_blueprint",       "generate_content")
    g.add_edge("generate_content",      "validate_output")
    g.add_edge("validate_output",       "decide_next_action")

    g.add_conditional_edges(
        "decide_next_action", route_after_decision,
        {"repair":"generate_content","finalize":"finalize"}
    )
    g.add_edge("finalize", END)
    return g.compile()

AGENT_GRAPH = build_agent_graph()

# ---------------------------------------------------------------------------
# State builder + runner
# ---------------------------------------------------------------------------

def _empty_state(run_id, amazon_url="", pdp_url="", output_lang="en", source_mode="",
                 raw_source_text="", helium_rows=None, reviews_text="",
                 runtime_errors=None) -> AgentState:
    return AgentState(
        run_id=run_id,
        amazon_url=amazon_url or "",
        pdp_url=pdp_url or "",
        output_lang=_coerce_output_lang(output_lang),
        source_mode=source_mode or "",
        raw_source_text=_safe_str(raw_source_text, max_len=300000),
        helium_rows=helium_rows or [],
        reviews_text=_safe_str(reviews_text, max_len=20000),
        source_text="",
        amazon_source_text="",
        pdp_clean_text="",
        guidelines=[],
        helium_analysis={"detected_type":"none","top_keywords":[],"warnings":[]},
        product_facts={},
        review_themes=[],
        blueprint={},
        listing={},
        validation_report={},
        validation_status="FAIL",
        generation_count=0,
        repair_count=0,
        next_action="finalize",
        keyword_coverage_report={},
        claim_ledger=[],
        google_doc_url="",
        runtime_errors=runtime_errors or []
    )

def run_agent(initial_state: AgentState) -> Dict[str, Any]:
    AGENT_GRAPH.invoke(initial_state)
    with RUN_LOCK:
        return (RUN_STORE.get(initial_state["run_id"]) or {}).get("result") or {}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return FileResponse(str(UI_DIR / "client.html"))

@app.get("/elves")
def elves():
    return FileResponse(str(UI_DIR / "elves.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/runs/{run_id}")
def get_run(run_id: str):
    with RUN_LOCK:
        rec = RUN_STORE.get(run_id)
    if not rec:
        raise HTTPException(status_code=404, detail="run_id not found")
    return rec

@app.post("/generate")
def generate(req: GenerateRequest):
    run_id = str(uuid.uuid4())
    run_init(run_id, req.amazon_url or "", req.pdp_url or "")
    run_event(run_id,"archivist","start","Run started (json/url mode)",{
        "output_lang": _coerce_output_lang(req.output_lang)
    })

    errors: List[Dict[str, Any]] = []
    raw = ""
    mode = ""

    if not req.pdp_url and not req.amazon_url:
        errors.append({"stage":"source","error":"Provide pdp_url and/or amazon_url. For file input use /generate_upload."})
    elif req.pdp_url:
        try:
            run_event(run_id,"archivist","pdp_fetch",f"Fetching PDP: {req.pdp_url}")
            raw = fetch_page_text(req.pdp_url)
            mode = "pdp_url"
            run_event(run_id,"archivist","pdp_fetch_ok",f"{len(raw)} raw chars")
        except Exception as e:
            errors.append({"stage":"fetch_page_text","error":repr(e)})
            run_event(run_id,"archivist","pdp_fetch_error",repr(e))

    return run_agent(_empty_state(
        run_id,
        amazon_url=req.amazon_url or "",
        pdp_url=req.pdp_url or "",
        output_lang=req.output_lang or "en",
        source_mode=mode,
        raw_source_text=raw,
        runtime_errors=errors
    ))

@app.post("/generate_upload")
async def generate_upload(
    amazon_url:           Optional[str]        = Form(None),
    pdp_url:              Optional[str]        = Form(None),
    output_lang:          Optional[str]        = Form("en"),
    product_info_file:    Optional[UploadFile] = File(None),
    pdp_text_file:        Optional[UploadFile] = File(None),
    helium10_csv:         Optional[UploadFile] = File(None),
    reviews_file:         Optional[UploadFile] = File(None),
):
    run_id = str(uuid.uuid4())
    run_init(run_id, amazon_url or "", pdp_url or "")
    run_event(run_id,"archivist","start","Run started (upload mode)",{
        "output_lang": _coerce_output_lang(output_lang)
    })

    errors: List[Dict[str, Any]] = []
    helium_rows: List[Dict[str, str]] = []

    if helium10_csv:
        try:
            run_event(run_id,"archivist","helium10_parse",f"Parsing {helium10_csv.filename}")
            helium_rows = parse_helium10_csv(helium10_csv)
            run_event(run_id,"archivist","helium10_parse_ok",f"{len(helium_rows)} rows")
        except Exception as e:
            errors.append({"stage":"parse_helium10_csv","error":repr(e)})

    reviews_text = ""
    if reviews_file:
        try:
            run_event(run_id,"archivist","reviews_parse",f"Parsing {reviews_file.filename}")
            reviews_text = parse_text_upload(reviews_file)[:8000]
            run_event(run_id,"archivist","reviews_parse_ok",f"{len(reviews_text)} chars")
        except Exception as e:
            errors.append({"stage":"parse_reviews_file","error":repr(e)})

    raw = ""
    mode = ""
    try:
        if pdp_url:
            run_event(run_id,"archivist","pdp_fetch",f"Fetching PDP: {pdp_url}")
            raw = fetch_page_text(pdp_url)
            mode = "pdp_url"
            run_event(run_id,"archivist","pdp_fetch_ok",f"{len(raw)} raw chars")
        elif product_info_file:
            run_event(run_id,"archivist","product_info_upload",f"Reading {product_info_file.filename}")
            raw = parse_text_upload(product_info_file)
            mode = "product_info_file"
        elif pdp_text_file:
            run_event(run_id,"archivist","pdp_text_upload",f"Reading {pdp_text_file.filename}")
            raw = parse_text_upload(pdp_text_file)
            mode = "pdp_text_file"
        elif not amazon_url:
            raise ValueError("No source. Supply pdp_url, product_info_file, pdp_text_file, or amazon_url.")
    except Exception as e:
        errors.append({"stage":"source_text","error":repr(e)})

    return run_agent(_empty_state(
        run_id,
        amazon_url=amazon_url or "",
        pdp_url=pdp_url or "",
        output_lang=output_lang or "en",
        source_mode=mode,
        raw_source_text=raw,
        helium_rows=helium_rows,
        reviews_text=reviews_text,
        runtime_errors=errors
    ))