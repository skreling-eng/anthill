"""Download and extract readable text from search result pages."""

from __future__ import annotations

import html as html_lib
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from externals.search.duckduckgo import _USER_AGENT, _read_response

# Tags that wrap content and have a real end tag (not void/self-closing).
_BLOCK_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "object",
        "embed",
        "template",
        "head",
    }
)
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_TAG_RE = re.compile(r"<[^>]+>")
_META_CONTENT_RE = re.compile(
    r'<meta[^>]+(?:name|property)\s*=\s*["\']'
    r"(?:description|og:description|twitter:description)[^>]*"
    r'content\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_DROP_ANCESTORS = frozenset(
    {
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "button",
        "menu",
    }
)
_JUNK_RE = re.compile(
    r"(cookie|gdpr|subscribe|newsletter|sign\s*up|log\s*in|"
    r"accept\s+all|privacy\s+policy|terms\s+of\s+use|"
    r"advertisement|sponsored|related\s+articles|"
    r"share\s+on\s+(facebook|twitter)|all\s+rights\s+reserved)",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SKIP_URL_SCHEMES = frozenset({"javascript", "mailto", "tel", "data"})


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def parse_fetch_pages_arg(value: str) -> bool:
    return _truthy(value)


def should_fetch_url(url: str) -> bool:
    url = url.strip()
    if not url:
        return False
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.scheme in _SKIP_URL_SCHEMES:
        return False
    path = (parsed.path or "").lower()
    if path.endswith((".pdf", ".zip", ".exe", ".dmg", ".mp4", ".mp3", ".jpg", ".png")):
        return False
    return True


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._stack: list[str] = []
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            return
        self._stack.append(tag)
        if tag in _BLOCK_TAGS or tag in _DROP_ANCESTORS:
            self._skip += 1
            return
        if self._skip == 0 and tag in (
            "br",
            "p",
            "div",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        ):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            return
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        if tag in _BLOCK_TAGS or tag in _DROP_ANCESTORS:
            if self._skip:
                self._skip -= 1
        elif self._skip == 0 and tag in (
            "p",
            "div",
            "li",
            "tr",
            "table",
            "section",
            "article",
        ):
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip > 0:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)
            self._chunks.append(" ")

    def get_text(self) -> str:
        return _clean_lines("".join(self._chunks))


def _clean_lines(raw: str) -> str:
    raw = html_lib.unescape(raw)
    lines: list[str] = []
    for line in raw.splitlines():
        line = _WS_RE.sub(" ", line).strip()
        if not line or len(line) < 2:
            continue
        if _JUNK_RE.search(line) and len(line) < 120:
            continue
        if len(line) > 20 and sum(c.isalnum() or c.isspace() for c in line) / len(line) < 0.45:
            continue
        lines.append(line)
    text = "\n".join(lines)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def _fallback_strip_tags(html: str) -> str:
    """Last-resort text when the structured parser yields nothing."""
    body = html
    m = re.search(r"<body\b[^>]*>(.*)</body>", html, re.DOTALL | re.IGNORECASE)
    if m:
        body = m.group(1)
    text = html_lib.unescape(re.sub(r"<[^>]+>", "\n", body))
    return _clean_lines(text)


def _meta_snippets(html: str) -> str:
    parts: list[str] = []
    for m in _TITLE_RE.finditer(html):
        t = _strip_tags(m.group(1))
        if t:
            parts.append(t)
    for m in _META_CONTENT_RE.finditer(html):
        t = html_lib.unescape(m.group(1)).strip()
        if t and t not in parts:
            parts.append(t)
    return "\n".join(parts)


def _strip_tags(fragment: str) -> str:
    return html_lib.unescape(_TAG_RE.sub("", fragment)).strip()


def extract_text_from_html(html: str) -> str:
    """Strip boilerplate tags/sections and return plain text."""
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
    for tag in ("script", "style", "noscript"):
        html = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>",
            " ",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
        text = parser.get_text()
    except Exception:
        text = ""
    if not text:
        text = _fallback_strip_tags(html)
    if not text:
        text = _meta_snippets(html)
    return text


def fetch_page_text(url: str, *, timeout: float = 15.0, max_chars: int = 12000) -> str:
    """Download URL and return cleaned plain text."""
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "html" not in ctype and "text/" not in ctype:
            raise RuntimeError(f"unsupported content type: {ctype or 'unknown'}")
        body = _read_response(resp)
    text = extract_text_from_html(body)
    if not text:
        raise RuntimeError("no readable text extracted")
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n…[truncated]"
    return text


def enrich_results_with_pages(
    rows: list[dict[str, str]],
    *,
    max_pages: int,
    timeout: float,
    max_chars: int,
) -> list[dict[str, str]]:
    """Fetch up to max_pages URLs; add page_text (and extend text) on each row."""
    if max_pages <= 0:
        return rows

    fetched = 0
    out: list[dict[str, str]] = []
    for row in rows:
        row = dict(row)
        url = row.get("url", "")
        if fetched < max_pages and should_fetch_url(url):
            try:
                page_text = fetch_page_text(
                    url, timeout=timeout, max_chars=max_chars
                )
                row["page_text"] = page_text
                row["page_fetch"] = "ok"
                snippet = row.get("text", "")
                if page_text:
                    row["text"] = (
                        f"{snippet}\n\n--- page ---\n{page_text}"
                        if snippet
                        else page_text
                    )
                fetched += 1
            except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, TimeoutError) as exc:
                row["page_fetch"] = f"error: {exc}"
        else:
            if fetched >= max_pages and should_fetch_url(url):
                row["page_fetch"] = "skipped: fetch_max"
        out.append(row)
    return out
