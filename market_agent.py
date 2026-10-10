"""
Phase 3 — Market Agent (quantitative analysis).

Pure Python over the close_series/volume_series that the Data Agent already fetched — no network calls here, which makes this module fully unit-testable offline. Computes period return, volatility, moving averages, and classifies a trend using documented thresholds.
Documented thresholds [Criterias which are used] :
  period_return_pct > +3%  -> "uptrend"
  period_return_pct < -3%  -> "downtrend"
  otherwise                -> "sideways"

Volatility = standard deviation of daily percentage returns, annualised by multiplying by sqrt(252) (the usual finance-industry convention for trading days per year), expressed as a percentage.
"""

import math

from state import FinState

TREND_UP_THRESHOLD = 3.0
TREND_DOWN_THRESHOLD = -3.0
TRADING_DAYS_PER_YEAR = 252


def _daily_returns(closes):
    """Simple daily percentage returns from a list of closing prices."""
    return [
        (closes[i] - closes[i - 1]) / closes[i - 1] * 100
        for i in range(1, len(closes))
        if closes[i - 1] != 0
    ]


def _stdev(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _moving_average(closes, window):
    """Trailing moving average over the last `window` closes, or None if
    there isn't enough history."""
    if len(closes) < window:
        return None
    return round(sum(closes[-window:]) / window, 2)


def compute_market_stats(closes, volumes=None):
    """Pure function: closes/volumes -> stats dict. Testable without any
    LangGraph or network dependency."""
    if not closes or len(closes) < 2:
        return {
            "status": "error",
            "message": "need at least 2 closing prices to compute market stats",
        }

    period_return_pct = round((closes[-1] - closes[0]) / closes[0] * 100, 2)

    daily_returns = _daily_returns(closes)
    daily_vol = _stdev(daily_returns)
    annualised_vol_pct = round(daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR), 2)

    ma_short = _moving_average(closes, 50)
    ma_long = _moving_average(closes, 200)

    avg_volume = round(sum(volumes) / len(volumes), 0) if volumes else 0.0

    if period_return_pct > TREND_UP_THRESHOLD:
        trend = "uptrend"
    elif period_return_pct < TREND_DOWN_THRESHOLD:
        trend = "downtrend"
    else:
        trend = "sideways"

    return {
        "status": "success",
        "period_return_pct": period_return_pct,
        "volatility_pct": annualised_vol_pct,
        "ma_short": ma_short,
        "ma_long": ma_long,
        "avg_volume": avg_volume,
        "trend": trend,
    }


def market_agent_node(state: FinState) -> FinState:
    """LangGraph node wrapper around compute_market_stats."""
    if state.get("data_status") != "success":
        print("  [market_agent] skipped — no usable price data from Data Agent")
        return {**state, "trend": "unknown"}

    stats = compute_market_stats(state.get("close_series", []), state.get("volume_series", []))

    if stats["status"] == "error":
        print(f"  [market_agent] {stats['message']}")
        return {**state, "trend": "unknown"}

    # optional benchmark comparison against the S&P 500 index
    index_comparison = {}
    try:
        from data_agent import get_history_dataframe
        idx_history = get_history_dataframe("^GSPC", state["start_date"], state["end_date"])
        if idx_history is not None and not idx_history.empty and len(idx_history) >= 2:
            idx_close_col = "Close" if "Close" in idx_history.columns else idx_history.columns[0]
            idx_closes = [round(float(v), 2) for v in idx_history[idx_close_col].tolist()]
            idx_stats = compute_market_stats(idx_closes)
            if idx_stats["status"] == "success":
                index_comparison = {
                    "benchmark": "^GSPC",
                    "benchmark_return_pct": idx_stats["period_return_pct"],
                    "relative_return_pct": round(
                        stats["period_return_pct"] - idx_stats["period_return_pct"], 2
                    ),
                }
    except Exception as e:
        print(f"  [market_agent] benchmark comparison skipped: {e}")

    print(f"  [market_agent] return={stats['period_return_pct']}% "
          f"vol={stats['volatility_pct']}% trend={stats['trend']}")

    return {
        **state,
        "period_return_pct": stats["period_return_pct"],
        "volatility_pct": stats["volatility_pct"],
        "ma_short": stats["ma_short"],
        "ma_long": stats["ma_long"],
        "avg_volume": stats["avg_volume"],
        "trend": stats["trend"],
        "index_comparison": index_comparison,
    }
