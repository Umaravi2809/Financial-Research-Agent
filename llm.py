"""
Shared LLM connection. Everything runs locally through Ollama — no cloud API keys for the LLM. (yfinance itself talks to Yahoo's public endpoints for market data, which is a separate, unauthenticated data source — not an LLM API key.)
"""

from langchain_ollama import ChatOllama

MODEL_NAME = "llama3.2"

llm = ChatOllama(model=MODEL_NAME, temperature=0)
llm_creative = ChatOllama(model=MODEL_NAME, temperature=0.3)


def get_llm(creative: bool = False):
    return llm_creative if creative else llm
