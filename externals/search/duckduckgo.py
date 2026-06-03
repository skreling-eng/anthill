"""DuckDuckGo web search (HTML endpoint + Instant Answer API fallback)."""

from __future__ import annotations

import gzip
import html as html_lib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zlib

_HTML_URL = "https://html.duckduckgo.com/html/"
_INSTANT_URL = "https://api.duckduckgo.com/"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_RESULT_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r".*?"
    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _kl(lang: str, region: str) -> str:
    for key in ("AH_SEARCH_KL", "AH_SERCH_KL"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    region = region.strip().lower()
    lang = lang.strip().lower()
    if region and "-" in region:
        return region
    if region and lang:
        return f"{region}-{lang}"
    if lang and len(lang) == 2:
        return f"wt-{lang}"
    return "wt-wt"


def _build_query(query: str, site: str) -> str:
    query = query.strip()
    site = site.strip()
    if site:
        site_q = site if site.startswith("site:") else f"site:{site}"
        return f"{site_q} {query}".strip()
    return query


def unwrap_url(href: str) -> str:
    href = html_lib.unescape(href.strip())
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        uddg = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        if uddg:
            return urllib.parse.unquote(uddg)
    return href


def _strip_tags(fragment: str) -> str:
    return html_lib.unescape(_TAG_RE.sub("", fragment)).strip()


def parse_html_results(html: str, *, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for href, title_html, snippet_html in _RESULT_RE.findall(html):
        url = unwrap_url(href)
        if not url or url.startswith("https://duckduckgo.com/y.js"):
            continue
        title = _strip_tags(title_html)
        snippet = _strip_tags(snippet_html)
        text = title
        if snippet and snippet != title:
            text = f"{title}\n{snippet}" if title else snippet
        out.append({"url": url, "text": text})
        if len(out) >= limit:
            break
    return out


def normalize_instant_results(data: object) -> list[dict[str, str]]:
    """Map DuckDuckGo Instant Answer API JSON to {url, text}."""
    if not isinstance(data, dict):
        return []
    out: list[dict[str, str]] = []

    abstract = str(data.get("AbstractText") or data.get("Abstract") or "").strip()
    abstract_url = str(data.get("AbstractURL") or "").strip()
    if abstract_url and abstract:
        out.append({"url": abstract_url, "text": abstract})

    for item in data.get("Results") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("FirstURL") or "").strip()
        text = str(item.get("Text") or "").strip()
        if url and text:
            out.append({"url": url, "text": text})

    def walk_topics(topics: object) -> None:
        if not isinstance(topics, list):
            return
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            if "Topics" in topic:
                walk_topics(topic.get("Topics"))
                continue
            url = str(topic.get("FirstURL") or "").strip()
            text = str(topic.get("Text") or "").strip()
            if url and text:
                out.append({"url": url, "text": text})

    walk_topics(data.get("RelatedTopics"))
    return out


def _read_response(resp: object) -> str:
    raw = resp.read()
    enc = (resp.headers.get("Content-Encoding") or "").lower()
    if enc == "gzip":
        raw = gzip.decompress(raw)
    elif enc == "deflate":
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    charset = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def _http_post(url: str, data: dict[str, str], *, timeout: float) -> str:
    body = urllib.parse.urlencode(data).encode("utf-8")
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://html.duckduckgo.com",
        "Referer": "https://html.duckduckgo.com/",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _read_response(resp)


def _http_get(url: str, params: dict[str, str], *, timeout: float) -> object:
    qs = urllib.parse.urlencode(params)
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    req = urllib.request.Request(f"{url}?{qs}", headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = _read_response(resp)
    return json.loads(raw)


def _search_html(query: str, *, kl: str, limit: int, timeout: float) -> list[dict[str, str]]:
    data = {"q": query, "b": "", "kl": kl}
    html = _http_post(_HTML_URL, data, timeout=timeout)
    return parse_html_results(html, limit=limit)


def _search_instant(query: str, *, limit: int, timeout: float) -> list[dict[str, str]]:
    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "skip_disambig": "1",
    }
    data = _http_get(_INSTANT_URL, params, timeout=timeout)
    return normalize_instant_results(data)[:limit]


def search(
    query: str,
    *,
    limit: int = 10,
    lang: str = "",
    region: str = "",
    site: str = "",
    timeout: float = 60.0,
) -> list[dict[str, str]]:
    """Run one DuckDuckGo search; returns [{url, text}, ...]."""
    query = _build_query(query, site)
    if not query:
        return []

    limit = max(1, min(limit, 100))
    kl = _kl(lang, region)

    try:
        rows = _search_html(query, kl=kl, limit=limit, timeout=timeout)
        if rows:
            return rows
        rows = _search_instant(query, limit=limit, timeout=timeout)
        if rows:
            return rows
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"DuckDuckGo HTTP {exc.code}: {body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DuckDuckGo request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("DuckDuckGo Instant Answer API returned invalid JSON") from exc

    return []
