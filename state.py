"""
Phase 1 — Shared state schema for the Financial Research & Analytics Agent.

Every node in the graph reads from and writes to this single TypedDict
"""

from typing import TypedDict, List, Dict, Any, Optional


class FinState(TypedDict, total=False):
    # --- intake ---
    raw_query: str            # traveller-style free text, e.g. "How has AAPL done since Jan?"
    ticker: str               # resolved ticker symbol, e.g. "AAPL"
    start_date: str           # "YYYY-MM-DD"
    end_date: str             # "YYYY-MM-DD"

    # --- Data Agent ---
    price_summary: Dict[str, Any]     # first/last close, high, low, avg volume
    fundamentals: Dict[str, Any]      # sector, market cap, PE ratio
    close_series: List[float]         # daily close prices, oldest -> newest (JSON-safe)
    volume_series: List[float]        # daily volume, same order
    data_status: str                  # "success" | "error"
    data_notes: str

    # --- Market Agent (quantitative analysis) ---
    period_return_pct: float
    volatility_pct: float
    ma_short: Optional[float]         # 50-day moving average (None if insufficient data)
    ma_long: Optional[float]          # 200-day moving average
    avg_volume: float
    trend: str                        # "uptrend" | "downtrend" | "sideways"
    index_comparison: Dict[str, Any]  # same stats for a benchmark index, e.g. ^GSPC

    # --- Research Agent ---
    headlines: List[Dict[str, str]]   # [{title, date, publisher, link}]
    research_summary: str             # LLM narrative built only from headlines

    # --- Report Agent (draft) ---
    draft_narrative: str

    # --- Verification Agent ---
    verification_flags: List[Dict[str, str]]   # [{rule, detail, severity}]
    verification_status: str          # "verified" | "flagged"
    revision_count: int

    # --- Report Agent (final) ---
    final_narrative: str
    final_report: Dict[str, Any]
    report_path: str


def new_state(raw_query: str, ticker: str, start_date: str, end_date: str) -> FinState:
    """Create a fresh state object for one research session."""
    return FinState(
        raw_query=raw_query,
        ticker=ticker.upper().strip(),
        start_date=start_date,
        end_date=end_date,
        price_summary={},
        fundamentals={},
        close_series=[],
        volume_series=[],
        data_status="",
        data_notes="",
        period_return_pct=0.0,
        volatility_pct=0.0,
        ma_short=None,
        ma_long=None,
        avg_volume=0.0,
        trend="",
        index_comparison={},
        headlines=[],
        research_summary="",
        draft_narrative="",
        verification_flags=[],
        verification_status="",
        revision_count=0,
        final_narrative="",
        final_report={},
        report_path="",
    )
