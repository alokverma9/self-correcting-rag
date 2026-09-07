import logging
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def search_duckduckgo(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Execute search using DuckDuckGo (via ddgs)."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise RuntimeError("ddgs or duckduckgo_search package is required for DuckDuckGo search.") from e

    logger.info(f"Executing DuckDuckGo search for: '{query}' (top_k={top_k})")
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=top_k))
    except Exception as e:
        raise RuntimeError(f"DuckDuckGo search failed for query '{query}': {e}") from e

    if not raw_results:
        raise RuntimeError(f"DuckDuckGo search returned no results for query: '{query}'")

    formatted_results: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_results[:top_k]):
        title = item.get("title", f"Web Result {idx + 1}")
        body = item.get("body") or item.get("snippet") or ""
        url = item.get("href") or item.get("link") or ""

        content = f"Title: {title}\nURL: {url}\nSnippet: {body}"
        formatted_results.append({
            "arxiv_id": f"web:{idx + 1}",
            "title": title,
            "chunk_idx": idx,
            "content": content,
            "url": url,
            "source": "web",
        })

    return formatted_results


def search_tavily(
    query: str,
    top_k: int = 3,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute search using Tavily API if key is provided."""
    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError("Tavily API key is missing. Set TAVILY_API_KEY in .env or pass api_key.")

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        response = client.search(query=query, max_results=top_k)
    except Exception as e:
        raise RuntimeError(f"Tavily search API failure for query '{query}': {e}") from e

    raw_results = response.get("results", [])
    if not raw_results:
        raise RuntimeError(f"Tavily search returned empty results for query: '{query}'")

    formatted_results: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_results[:top_k]):
        title = item.get("title", f"Web Result {idx + 1}")
        content = item.get("content", "")
        url = item.get("url", "")
        formatted_results.append({
            "arxiv_id": f"web:{idx + 1}",
            "title": title,
            "chunk_idx": idx,
            "content": f"Title: {title}\nURL: {url}\nContent: {content}",
            "url": url,
            "source": "web",
        })

    return formatted_results


def web_search(
    query: str,
    top_k: int = 3,
    prefer_tavily: bool = False,
) -> List[Dict[str, Any]]:
    """
    Search the web fallback and format results into internal chunk schema.
    Defaults to DuckDuckGo (or Tavily if preferred and key available).
    Raises an error on API/network failure.
    """
    if prefer_tavily and os.getenv("TAVILY_API_KEY"):
        return search_tavily(query=query, top_k=top_k)
    return search_duckduckgo(query=query, top_k=top_k)
