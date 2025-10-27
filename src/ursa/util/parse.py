import json
import os
import re
import shutil
import unicodedata
from typing import Any, Optional, Tuple
from urllib.parse import urljoin, urlparse

import justext
import requests
import trafilatura
from bs4 import BeautifulSoup


def extract_json(text: str) -> list[dict]:
    """
    Extract a JSON object or array from text that might contain markdown or other content.

    The function attempts three strategies:
        1. Extract JSON from a markdown code block labeled as JSON.
        2. Extract JSON from any markdown code block.
        3. Use bracket matching to extract a JSON substring starting with '{' or '['.

    Returns:
        A Python object parsed from the JSON string (dict or list).

    Raises:
        ValueError: If no valid JSON is found.
    """
    # Approach 1: Look for a markdown code block specifically labeled as JSON.
    labeled_block = re.search(
        r"```json\s*([\[{].*?[\]}])\s*```", text, re.DOTALL
    )
    if labeled_block:
        json_str = labeled_block.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fall back to the next approach if parsing fails.
            pass

    # Approach 2: Look for any code block delimited by triple backticks.
    generic_block = re.search(r"```(.*?)```", text, re.DOTALL)
    if generic_block:
        json_str = generic_block.group(1).strip()
        if json_str.startswith("{") or json_str.startswith("["):
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    # Approach 3: Attempt to extract JSON using bracket matching.
    # Find the first occurrence of either '{' or '['.
    first_obj = text.find("{")
    first_arr = text.find("[")
    if first_obj == -1 and first_arr == -1:
        raise ValueError("No JSON object or array found in the text.")

    # Determine which bracket comes first.
    if first_obj == -1:
        start = first_arr
        open_bracket = "["
        close_bracket = "]"
    elif first_arr == -1:
        start = first_obj
        open_bracket = "{"
        close_bracket = "}"
    else:
        if first_obj < first_arr:
            start = first_obj
            open_bracket = "{"
            close_bracket = "}"
        else:
            start = first_arr
            open_bracket = "["
            close_bracket = "]"

    # Bracket matching: find the matching closing bracket.
    depth = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == open_bracket:
            depth += 1
        elif text[i] == close_bracket:
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        raise ValueError(
            "Could not find matching closing bracket for JSON content."
        )

    json_str = text[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError("Extracted content is not valid JSON.") from e


PDF_CT_HINTS = (
    "application/pdf",
    "binary/octet-stream",
)  # some servers mislabel
PDF_EXT_RE = re.compile(r"\.pdf($|\?)", re.IGNORECASE)


def _is_pdf_response(resp: requests.Response) -> bool:
    ct = resp.headers.get("Content-Type", "").lower()
    if any(hint in ct for hint in PDF_CT_HINTS):
        return True
    # Sometimes servers omit CT but set filename
    cd = resp.headers.get("Content-Disposition", "")
    if "filename" in cd and ".pdf" in cd.lower():
        return True
    # Last resort: URL extension
    return bool(PDF_EXT_RE.search(resp.url))


def _derive_filename_from_cd_or_url(
    resp: requests.Response, fallback: str
) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^\";]+)"?', cd, re.IGNORECASE)
    if m:
        name = m.group(1)
        # Some headers include quotes
        name = name.strip("\"'")

        # RFC 5987 may encode UTF-8 in filename*; we’re treating as plain here.
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        return name

    # use URL last path segment if looks like PDF
    parsed = urlparse(resp.url)
    base = os.path.basename(parsed.path) or fallback
    if not base.lower().endswith(".pdf"):
        if PDF_EXT_RE.search(resp.url):
            base = re.sub(
                r"(\.pdf)(?:$|\?).*", r"\1", base, flags=re.IGNORECASE
            )
            if not base.lower().endswith(".pdf"):
                base += ".pdf"
        else:
            base = (
                fallback
                if fallback.lower().endswith(".pdf")
                else fallback + ".pdf"
            )
    return base


def _download_stream_to(path: str, resp: requests.Response) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        shutil.copyfileobj(resp.raw, f)
    return path


def _get_soup(
    url: str, timeout: int = 20, headers: Optional[dict[str, str]] = None
) -> BeautifulSoup:
    r = requests.get(url, timeout=timeout, headers=headers or {})
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def _find_pdf_on_landing(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    # 1) meta citation_pdf_url
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta and meta.get("content"):
        return urljoin(base_url, meta["content"])

    # 2) obvious anchors: text contains 'PDF' or 'Download'
    for a in soup.find_all("a", href=True):
        label = (a.get_text(" ", strip=True) or "").lower()
        href = a["href"]
        if "pdf" in label or "download" in label or PDF_EXT_RE.search(href):
            return urljoin(base_url, href)

    # 3) buttons that wrap an anchor
    for btn in soup.find_all(["button", "a"], href=True):
        label = (btn.get_text(" ", strip=True) or "").lower()
        href = btn.get("href")
        if href and (
            "pdf" in label or "download" in label or PDF_EXT_RE.search(href)
        ):
            return urljoin(base_url, href)

    return None


# def _resolve_pdf_via_unpaywall(doi: str, email: str, timeout: int = 15) -> Optional[str]:
#     # Optional helper: respects publisher OA; returns None if no OA PDF
#     try:
#         url = f"https://api.unpaywall.org/v2/{doi}"
#         r = requests.get(url, params={"email": email}, timeout=timeout)
#         r.raise_for_status()
#         data = r.json()
#         loc = data.get("best_oa_location") or {}
#         pdf = loc.get("url_for_pdf") or loc.get("url")
#         if pdf and PDF_EXT_RE.search(pdf):
#             return pdf
#         # Sometimes url points to landing; try it anyway.
#         return pdf
#     except Exception:
#         return None


def resolve_pdf_from_osti_record(
    rec: dict[str, Any],
    *,
    headers: Optional[dict[str, str]] = None,
    unpaywall_email: Optional[str] = None,
    timeout: int = 25,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns (pdf_url, landing_used, note)
      - pdf_url: direct downloadable PDF URL if found (or a strong candidate)
      - landing_used: landing page URL we parsed (if any)
      - note: brief trace of how we found it
    """
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    note_parts: list[str] = []

    links = rec.get("links", []) or []
    # doi = rec.get("doi")

    # 1) Try 'fulltext' first (OSTI purl)
    fulltext = None
    for link in links:
        if link.get("rel") == "fulltext":
            fulltext = link.get("href")
            break

    if fulltext:
        note_parts.append("Tried links[fulltext] purl")
        try:
            # Follow redirects; stream to peek headers without loading whole body
            r = requests.get(
                fulltext,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            )
            r.raise_for_status()

            if _is_pdf_response(r):
                note_parts.append("fulltext resolved directly to PDF")
                return (r.url, None, " | ".join(note_parts))

            # Not a PDF: parse page HTML for meta or obvious PDF anchors
            # (If server sent binary but CT lied, _is_pdf_response would have caught via CD or ext)
            r.close()
            soup = _get_soup(fulltext, timeout=timeout, headers=headers)
            candidate = _find_pdf_on_landing(soup, fulltext)
            if candidate:
                note_parts.append(
                    "found PDF via meta/anchor on fulltext landing"
                )
                return (candidate, fulltext, " | ".join(note_parts))
        except Exception as e:
            note_parts.append(f"fulltext failed: {e}")

    # 2) Try DOE PAGES landing (citation_doe_pages)
    doe_pages = None
    for link in links:
        if link.get("rel") == "citation_doe_pages":
            doe_pages = link.get("href")
            break

    if doe_pages:
        note_parts.append("Tried links[citation_doe_pages] landing")
        try:
            soup = _get_soup(doe_pages, timeout=timeout, headers=headers)
            candidate = _find_pdf_on_landing(soup, doe_pages)
            if candidate:
                # Candidate may itself be a landing—check if it serves PDF
                try:
                    r2 = requests.get(
                        candidate,
                        headers=headers,
                        timeout=timeout,
                        allow_redirects=True,
                        stream=True,
                    )
                    r2.raise_for_status()
                    if _is_pdf_response(r2):
                        note_parts.append("citation_doe_pages → direct PDF")
                        return (r2.url, doe_pages, " | ".join(note_parts))
                    r2.close()
                except Exception:
                    pass
                # If not clearly PDF, still return as a candidate (agent will fetch & parse)
                note_parts.append(
                    "citation_doe_pages → PDF-like candidate (not confirmed by headers)"
                )
                return (candidate, doe_pages, " | ".join(note_parts))
        except Exception as e:
            note_parts.append(f"citation_doe_pages failed: {e}")

    # # 3) Optional: DOI → Unpaywall OA
    # if doi and unpaywall_email:
    #     note_parts.append("Tried Unpaywall via DOI")
    #     pdf_from_ua = _resolve_pdf_via_unpaywall(doi, unpaywall_email)
    #     if pdf_from_ua:
    #         # May be direct PDF or landing; the caller will validate headers during download
    #         note_parts.append("Unpaywall returned candidate")
    #         return (pdf_from_ua, None, " | ".join(note_parts))

    # 4) Give up
    note_parts.append("No PDF found")
    return (None, None, " | ".join(note_parts))


def _normalize_ws(text: str) -> str:
    # Normalize unicode, collapse whitespace, and strip control chars
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def _dedupe_lines(text: str, min_len: int = 40) -> str:
    seen = set()
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        # Ignore very short or repeated lines (menus, cookie banners, etc.)
        if len(stripped) < min_len:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(stripped)
    return "\n\n".join(out)


def extract_main_text_only(html: str, *, max_chars: int = 250_000) -> str:
    """
    Returns plain text with navigation/ads/scripts removed.
    Prefers trafilatura -> jusText -> BS4 paragraphs.
    """
    # 1) Trafilatura
    # You can tune config: with_metadata, include_comments, include_images, favor_recall, etc.
    cfg = trafilatura.settings.use_config()
    cfg.set("DEFAULT", "include_comments", "false")
    cfg.set("DEFAULT", "include_tables", "false")
    cfg.set("DEFAULT", "favor_recall", "false")  # be stricter; less noise
    try:
        # If you fetched HTML already, use extract() on string; otherwise, fetch_url(url)
        txt = trafilatura.extract(
            html,
            config=cfg,
            include_comments=False,
            include_tables=False,
            favor_recall=False,
        )
        if txt and txt.strip():
            txt = _normalize_ws(txt)
            txt = _dedupe_lines(txt)
            return txt[:max_chars]
    except Exception:
        pass

    # 2) jusText
    try:
        paragraphs = justext.justext(html, justext.get_stoplist("English"))
        body_paras = [p.text for p in paragraphs if not p.is_boilerplate]
        if body_paras:
            txt = _normalize_ws("\n\n".join(body_paras))
            txt = _dedupe_lines(txt)
            return txt[:max_chars]
    except Exception:
        pass

    # 4) last-resort: BS4 paragraphs/headings only
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup([
        "script",
        "style",
        "noscript",
        "header",
        "footer",
        "nav",
        "form",
        "aside",
    ]):
        tag.decompose()
    chunks = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "figcaption"]):
        t = el.get_text(" ", strip=True)
        if t:
            chunks.append(t)
    txt = _normalize_ws("\n\n".join(chunks))
    txt = _dedupe_lines(txt)
    return txt[:max_chars]
