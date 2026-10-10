"""
Phase 6 — Verification Agent (rule-based, deterministic).

Checks the draft narrative against the actual computed numbers and the actual retrieved headlines. Deliberately does NOT ask the LLM to judge its own consistency — every check here is plain string/keyword matching or numeric comparison, so it is fully unit-testable and its behaviour doesn't drift between runs.
Four rules, matching the design discussed:
  1. trend_word_check      — narrative's trend wording vs. the classified trend
  2. date_window_check     — any headline referenced falls inside the window
  3. keyword_support_check — claims like "earnings"/"merger" have a
                              matching headline as evidence
  4. (aggregate)           — collects every flag with a severity level
"""

import re
from datetime import datetime

from state import FinState

# Words that imply a directional claim, grouped by the direction they imply.
UP_WORDS = ["uptrend", "rally", "rallied", "gained", "gains", "rose", "surged", "climbed", "soared"]
DOWN_WORDS = ["downtrend", "decline", "declined", "fell", "dropped", "slumped", "plunged", "tumbled"]

# Topic keywords that require supporting evidence in the retrieved headlines.
EVIDENCE_TOPICS = {
    "earnings": ["earnings", "eps", "quarterly results", "revenue"],
    "merger/acquisition": ["merger", "acquisition", "acquire", "acquired", "takeover"],
    "lawsuit": ["lawsuit", "litigation", "sued", "settlement"],
    "dividend": ["dividend"],
    "guidance": ["guidance", "forecast", "outlook"],
}


def _find_words(text: str, words: list) -> list:
    text_lower = text.lower()
    return [w for w in words if re.search(rf"\b{re.escape(w)}\b", text_lower)]


def trend_word_check(narrative: str, actual_trend: str) -> list:
    """Flag if the narrative's directional language contradicts the
    classified trend."""
    flags = []
    up_hits = _find_words(narrative, UP_WORDS)
    down_hits = _find_words(narrative, DOWN_WORDS)

    if actual_trend == "downtrend" and up_hits:
        flags.append({
            "rule": "trend_word_check",
            "detail": f"Narrative uses upward language ({up_hits}) but classified trend is 'downtrend'.",
            "severity": "high",
        })
    if actual_trend == "uptrend" and down_hits:
        flags.append({
            "rule": "trend_word_check",
            "detail": f"Narrative uses downward language ({down_hits}) but classified trend is 'uptrend'.",
            "severity": "high",
        })
    if actual_trend == "sideways" and (up_hits or down_hits):
        flags.append({
            "rule": "trend_word_check",
            "detail": f"Narrative uses directional language ({up_hits + down_hits}) but classified trend is 'sideways'.",
            "severity": "medium",
        })
    return flags


def date_window_check(headlines: list, start_date: str, end_date: str) -> list:
    """Flag any retrieved headline whose date falls outside the requested window."""
    flags = []
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return [{
            "rule": "date_window_check",
            "detail": f"Could not parse window dates '{start_date}' / '{end_date}'.",
            "severity": "low",
        }]

    for h in headlines:
        h_date_str = h.get("date", "")
        if not h_date_str:
            continue
        try:
            h_date = datetime.strptime(h_date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if not (start <= h_date <= end):
            flags.append({
                "rule": "date_window_check",
                "detail": f"Headline '{h.get('title', '')[:60]}' dated {h_date_str} falls outside {start_date} to {end_date}.",
                "severity": "low",
            })
    return flags


def keyword_support_check(narrative: str, headlines: list) -> list:
    """Flag topic claims in the narrative that have no supporting headline."""
    flags = []
    narrative_lower = narrative.lower()
    headlines_text = " ".join(h.get("title", "") for h in headlines).lower()

    for topic, keywords in EVIDENCE_TOPICS.items():
        claim_present = any(kw in narrative_lower for kw in keywords)
        if not claim_present:
            continue
        evidence_present = any(kw in headlines_text for kw in keywords)
        if not evidence_present:
            flags.append({
                "rule": "keyword_support_check",
                "detail": f"Narrative references '{topic}' but no retrieved headline mentions it.",
                "severity": "high",
            })
    return flags


def verify(narrative: str, trend: str, headlines: list, start_date: str, end_date: str) -> dict:
    """Pure function: run all rules, return the aggregated result. Fully
    testable without LangGraph or any network dependency."""
    flags = []
    flags += trend_word_check(narrative, trend)
    flags += date_window_check(headlines, start_date, end_date)
    flags += keyword_support_check(narrative, headlines)

    status = "flagged" if flags else "verified"
    return {"status": status, "flags": flags}


def verification_agent_node(state: FinState) -> FinState:
    """LangGraph node wrapper around verify()."""
    result = verify(
        narrative=state.get("draft_narrative", ""),
        trend=state.get("trend", "unknown"),
        headlines=state.get("headlines", []),
        start_date=state["start_date"],
        end_date=state["end_date"],
    )

    if result["flags"]:
        print(f"  [verification_agent] {len(result['flags'])} flag(s) raised:")
        for f in result["flags"]:
            print(f"      - ({f['severity']}) {f['rule']}: {f['detail']}")
    else:
        print("  [verification_agent] no issues found — narrative verified")

    return {
        **state,
        "verification_flags": result["flags"],
        "verification_status": result["status"],
    }


def verification_router(state: FinState) -> str:
    """Conditional edge: revise once if flagged, otherwise go straight to
    finalizing. Capped at one revision pass to guarantee termination."""
    if state.get("verification_status") == "flagged" and state.get("revision_count", 0) == 0:
        return "revise_report"
    return "finalize_report"
