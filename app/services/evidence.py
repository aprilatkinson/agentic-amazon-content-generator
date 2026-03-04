# amazon_agent_clean/app/services/evidence.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import re


@dataclass(frozen=True)
class EvidenceSpan:
    text: str
    start: int
    end: int
    source: str = "pdp"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def find_best_span(haystack: str, needle: str) -> Optional[EvidenceSpan]:
    """
    Deterministic span finder.
    Priority:
      1) Case-insensitive exact substring match on original text.
      2) Whitespace-normalized substring match; returns span with best-effort indices.
    Returns the FIRST match only (deterministic).
    """
    if not haystack or not needle:
        return None

    needle_clean = needle.strip()
    if not needle_clean:
        return None

    # 1) Direct case-insensitive match on original text
    m = re.search(re.escape(needle_clean), haystack, flags=re.IGNORECASE)
    if m:
        return EvidenceSpan(
            text=haystack[m.start():m.end()],
            start=m.start(),
            end=m.end(),
            source="pdp",
        )

    # 2) Whitespace-normalized fallback
    hs_norm = _normalize_ws(haystack)
    nd_norm = _normalize_ws(needle_clean)
    if not hs_norm or not nd_norm:
        return None

    m2 = re.search(re.escape(nd_norm), hs_norm, flags=re.IGNORECASE)
    if not m2:
        return None

    # Best-effort: return normalized text with sentinel indices (0,0).
    # Validator relies on text containment, not indices.
    return EvidenceSpan(
        text=nd_norm,
        start=0,
        end=0,
        source="pdp",
    )