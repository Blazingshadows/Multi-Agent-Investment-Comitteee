"""Google News RSS search — far more stable/guaranteed-to-exist than a
specific finance site's own feed URL, and it aggregates Moneycontrol/ET
Markets/Business Standard/etc. anyway. Fetched via httpx (not feedparser's
built-in urlopen) since the local Python's default SSL context fails cert
verification against news.google.com on this machine.
"""

import httpx
import feedparser

RSS_URL = "https://news.google.com/rss/search"


def fetch_news(query: str, max_items: int = 8, timeout: float = 10.0) -> list[str]:
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    try:
        response = httpx.get(RSS_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError:
        return []

    feed = feedparser.parse(response.text)
    return [entry.title for entry in feed.entries[:max_items]]


def fetch_stock_news(symbol: str, company_name: str | None = None, max_items: int = 8) -> list[str]:
    query = f"{company_name or symbol} NSE stock"
    return fetch_news(query, max_items=max_items)
