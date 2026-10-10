"""
Phase 7 — Integration: one compiled LangGraph application.

Flow:

    START
      -> data_agent
      -> market_agent
      -> research_agent
      -> draft_report
      -> verification
      -> (conditional) revise_report | finalize_report
      -> [revise_report -> verification (re-check, capped at 1 pass)]
      -> finalize_report
      -> END

The revise loop is capped at exactly one pass by verification_router checking revision_count, so the graph always terminates.
"""

from langgraph.graph import StateGraph, START, END

from data_agent import data_agent_node
from market_agent import market_agent_node
from research_agent import research_agent_node
from report_agent import draft_report_node, revise_report_node
from verification_agent import verification_agent_node, verification_router
from report import finalize_report_node
from state import FinState


def build_graph():
    g = StateGraph(FinState)

    g.add_node("data_agent", data_agent_node)
    g.add_node("market_agent", market_agent_node)
    g.add_node("research_agent", research_agent_node)
    g.add_node("draft_report", draft_report_node)
    g.add_node("verification", verification_agent_node)
    g.add_node("revise_report", revise_report_node)
    g.add_node("finalize_report", finalize_report_node)

    g.add_edge(START, "data_agent")
    g.add_edge("data_agent", "market_agent")
    g.add_edge("market_agent", "research_agent")
    g.add_edge("research_agent", "draft_report")
    g.add_edge("draft_report", "verification")

    # conditional edge: revise once if flagged, otherwise finalize directly
    g.add_conditional_edges("verification", verification_router)

    # after a revision, re-run verification once more (capped by revision_count)
    g.add_edge("revise_report", "verification")

    g.add_edge("finalize_report", END)

    return g.compile()
