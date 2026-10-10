"""
Phase 5 — Report Agent (draft & revise stages).

The draft stage asks the LLM to write one narrative paragraph combining the Market Agent's numbers and the Research Agent's headline summary. The revise stage is only invoked if the Verification Agent flags a problem —it feeds the specific flags back to the LLM and asks for a corrected narrative that resolves them.The final assembly stage (turning the approved narrative into the fullJSON report) lives in report.py, mirroring the Trip Concierge's split between report-writing agents and the report-serialization module.
"""

from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm
from state import FinState


DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial research analyst. Write one short research "
     "paragraph (4-6 sentences) about the stock below, for the stated "
     "period. Base every claim strictly on the numbers and news summary "
     "given to you — do not invent figures. State the trend direction "
     "explicitly using the word 'uptrend', 'downtrend', or 'sideways' "
     "exactly as classified. If you reference a news event, only reference "
     "something present in the research summary."),
    ("human",
     "Ticker: {ticker}\n"
     "Period: {start_date} to {end_date}\n"
     "Period return: {period_return_pct}%\n"
     "Annualised volatility: {volatility_pct}%\n"
     "Classified trend: {trend}\n"
     "50-day moving average: {ma_short}\n"
     "200-day moving average: {ma_long}\n"
     "Average daily volume: {avg_volume}\n"
     "Research summary from recent headlines: {research_summary}"),
])


def draft_report_node(state: FinState) -> FinState:
    """LLM node: combine Market + Research outputs into one draft narrative."""
    chain = DRAFT_PROMPT | get_llm(creative=True)

    narrative = chain.invoke({
        "ticker": state["ticker"],
        "start_date": state["start_date"],
        "end_date": state["end_date"],
        "period_return_pct": state.get("period_return_pct", "unknown"),
        "volatility_pct": state.get("volatility_pct", "unknown"),
        "trend": state.get("trend", "unknown"),
        "ma_short": state.get("ma_short") or "insufficient history",
        "ma_long": state.get("ma_long") or "insufficient history",
        "avg_volume": state.get("avg_volume", "unknown"),
        "research_summary": state.get("research_summary", "no research summary available"),
    }).content.strip()

    print("  [report_agent:draft] narrative drafted")
    return {**state, "draft_narrative": narrative}


REVISE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You previously wrote a research paragraph about a stock, but an "
     "automated verification step found problems with it, listed below. "
     "Rewrite the paragraph to fix every listed problem while keeping "
     "everything that was correct. Base claims strictly on the numbers "
     "and research summary provided. State the trend using exactly the "
     "word given."),
    ("human",
     "Original paragraph:\n{draft_narrative}\n\n"
     "Verification problems found:\n{flags_text}\n\n"
     "Correct data to use:\n"
     "Period return: {period_return_pct}%\n"
     "Classified trend: {trend}\n"
     "Research summary: {research_summary}\n\n"
     "Write the corrected paragraph."),
])


def revise_report_node(state: FinState) -> FinState:
    """LLM node: rewrite the narrative to address the Verification Agent's flags."""
    flags = state.get("verification_flags", [])
    flags_text = "\n".join(f"- ({f['severity']}) {f['rule']}: {f['detail']}" for f in flags)

    chain = REVISE_PROMPT | get_llm()
    revised = chain.invoke({
        "draft_narrative": state.get("draft_narrative", ""),
        "flags_text": flags_text or "none listed",
        "period_return_pct": state.get("period_return_pct", "unknown"),
        "trend": state.get("trend", "unknown"),
        "research_summary": state.get("research_summary", ""),
    }).content.strip()

    print(f"  [report_agent:revise] narrative revised (revision #{state.get('revision_count', 0) + 1})")
    return {
        **state,
        "draft_narrative": revised,
        "revision_count": state.get("revision_count", 0) + 1,
    }
