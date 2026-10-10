"""
Phase 4 — Research Agent.

Pulls recent news headlines for the ticker via yfinance's built-in `Ticker.news` (no separate API key needed) and asks the LLM to summarise them into a short narrative. The LLM is explicitly told to use only the supplied headlines — it should not invent events that aren't in the list.
Guardrail: if no headlines are returned, the research summary will say so plainly instead of inventing context.
"""

from datetime import datetime, timezone

import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm
from state import FinState


def fetch_headlines(ticker: str, limit: int = 8) -> list:
    """Pull recent headlines for a ticker. Returns a list of dicts with
    title, date (YYYY-MM-DD), publisher, and link."""
    try:
        raw_news = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"  [research_agent] news fetch failed: {e}")
        return []

    headlines = []
    for item in raw_news[:limit]:
        # yfinance has changed its news payload shape across versions;
        # handle both the flat and the nested "content" formats defensively.
        content = item.get("content", item)
        title = content.get("title") or item.get("title", "")
        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else item.get("publisher", "unknown")
        )
        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else item.get("link", "")
        )

        timestamp = item.get("providerPublishTime")
        if timestamp:
            date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            pub_date = content.get("pubDate", "")
            date_str = pub_date[:10] if pub_date else ""

        if title:
            headlines.append({
                "title": title,
                "date": date_str,
                "publisher": publisher or "unknown",
                "link": link or "",
            })

    return headlines


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial research assistant. You will be given a list of "
     "recent news headlines for a stock. Write a short summary (3-4 sentences) "
     "of what these headlines suggest is happening with the company. "
     "IMPORTANT: only mention events that are actually present in the "
     "headlines below. Do not invent earnings results, mergers, or price "
     "targets that are not stated. If the headlines are generic or thin, say so."),
    ("human", "Headlines:\n{headlines_text}"),
])


def research_agent_node(state: FinState) -> FinState:
    """LangGraph node: fetch headlines, then have the LLM summarise them."""
    ticker = state["ticker"]
    headlines = fetch_headlines(ticker)

    if not headlines:
        print(f"  [research_agent] no headlines found for {ticker}")
        return {
            **state,
            "headlines": [],
            "research_summary": (
                f"No recent news headlines were available for {ticker} through "
                "the data source used. This report's narrative relies on price "
                "and volume data only."
            ),
        }

    headlines_text = "\n".join(
        f"- [{h['date']}] {h['title']} ({h['publisher']})" for h in headlines
    )

    chain = SUMMARY_PROMPT | get_llm()
    summary = chain.invoke({"headlines_text": headlines_text}).content.strip()

    print(f"  [research_agent] {len(headlines)} headlines retrieved for {ticker}")

    return {**state, "headlines": headlines, "research_summary": summary}
