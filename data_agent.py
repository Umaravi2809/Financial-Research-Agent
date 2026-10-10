"""
Phase 2 — Data Agent.

Pulls price history and basic fundamentals for a ticker over a date window using yfinance. This is the only place in the project that talks to Yahoo Finance's public data endpoints — no API key required, but it does need an internet connection.
Guardrails that are to be noted arre:
  - empty/short history (bad ticker, no trading days in the window) -> error status
  - .info can be flaky/incomplete for some tickers -> individual .get() with defaults
"""

import yfinance as yf

from state import FinState


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fetch_price_history(ticker: str, start_date: str, end_date: str):
    """Thin wrapper around yfinance so it's easy to mock in tests."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        multi_level_index=False
    )
    return data

def fetch_fundamentals(ticker: str) -> dict:
    """Pull a handful of fundamental fields, tolerating missing keys."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    return {
        "sector": info.get("sector", "unknown"),
        "industry": info.get("industry", "unknown"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
    }


def data_agent_node(state: FinState) -> FinState:
    """LangGraph node: populate price_summary, fundamentals, data_status."""
    ticker = state["ticker"]
    start_date = state["start_date"]
    end_date = state["end_date"]

    try:
        history = fetch_price_history(ticker, start_date, end_date)
    except Exception as e:
        print(f"  [data_agent] download failed: {e}")
        return {
            **state,
            "data_status": "error",
            "data_notes": f"Could not download price history for '{ticker}': {e}",
        }

    if history is None or history.empty or len(history) < 2:
        print(f"  [data_agent] no usable history for {ticker} in this window")
        return {
            **state,
            "data_status": "error",
            "data_notes": (
                f"No trading data for '{ticker}' between {start_date} and {end_date}. "
                "Check the ticker symbol and that the window contains trading days."
            ),
        }

    close_col = "Close" if "Close" in history.columns else history.columns[0]
    first_close = _safe_float(history[close_col].iloc[0])
    last_close = _safe_float(history[close_col].iloc[-1])
    period_high = _safe_float(history["High"].max()) if "High" in history else None
    period_low = _safe_float(history["Low"].min()) if "Low" in history else None
    avg_volume = _safe_float(history["Volume"].mean()) if "Volume" in history else 0.0

    price_summary = {
        "first_close": round(first_close, 2) if first_close is not None else None,
        "last_close": round(last_close, 2) if last_close is not None else None,
        "period_high": round(period_high, 2) if period_high is not None else None,
        "period_low": round(period_low, 2) if period_low is not None else None,
        "avg_volume": round(avg_volume, 0) if avg_volume is not None else 0,
        "trading_days": len(history),
    }

    fundamentals = fetch_fundamentals(ticker)

    close_series = [round(float(v), 2) for v in history[close_col].tolist()]
    volume_series = (
        [round(float(v), 0) for v in history["Volume"].tolist()]
        if "Volume" in history else []
    )

    print(f"  [data_agent] {ticker}: {len(history)} trading days, "
          f"{price_summary['first_close']} -> {price_summary['last_close']}")

    return {
        **state,
        "price_summary": price_summary,
        "fundamentals": fundamentals,
        "close_series": close_series,
        "volume_series": volume_series,
        "data_status": "success",
        "data_notes": f"Fetched {len(history)} trading days for {ticker}.",
    }


# expose the raw dataframe fetch for the Market Agent to reuse without
# re-downloading; kept as a separate cache-friendly function
def get_history_dataframe(ticker: str, start_date: str, end_date: str):
    return fetch_price_history(ticker, start_date, end_date)
