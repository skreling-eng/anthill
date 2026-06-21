"""DuckDuckGo web search (stdlib only, vendored for brain independence)."""

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


def _emulate_enabled() -> bool:
    return os.environ.get("BRAIN_EMULATE_SEARCH", "").lower() in ("1", "true", "yes")


def _unwrap_url(href: str) -> str:
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


def _parse_html_results(html: str, *, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for href, title_html, snippet_html in _RESULT_RE.findall(html):
        url = _unwrap_url(href)
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
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://html.duckduckgo.com",
        "Referer": "https://html.duckduckgo.com/",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _read_response(resp)


def web_search(query: str, *, limit: int = 5, timeout: float = 30.0) -> list[dict[str, str]]:
    """Return [{url, text}, ...] for a search query."""
    query = query.strip()
    if not query:
        return []
    limit = max(1, min(limit, 20))
    if _emulate_enabled() or os.environ.get("BRAIN_EMULATE", "").lower() in ("1", "true", "yes"):
        return [
            {
                "url": f"https://example.com/?q={urllib.parse.quote(query)}",
                "text": f"[emulated search] Information about: {query}",
            }
        ][:limit]
    try:
        html = _http_post(_HTML_URL, {"q": query, "b": "", "kl": "wt-wt"}, timeout=timeout)
        rows = _parse_html_results(html, limit=limit)
        if rows:
            return rows
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return [{"url": "", "text": f"[search failed] {exc}"}]
    return []
