# generic_acquisition_agents.py

import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, Mapping, Optional
from urllib.parse import quote, urlparse

import feedparser

# PDF & Vision extras (match your existing stack)
import pymupdf
import requests
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from PIL import Image
from typing_extensions import List, TypedDict

from ursa.agents.base import BaseAgent
from ursa.agents.rag_agent import RAGAgent
from ursa.util.parse import (
    _derive_filename_from_cd_or_url,
    _download_stream_to,
    _get_soup,
    _is_pdf_response,
    extract_main_text_only,
    resolve_pdf_from_osti_record,
)

try:
    from ddgs import DDGS  # pip install duckduckgo-search
except Exception:
    DDGS = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ---------- Shared State / Types ----------


class ItemMetadata(TypedDict, total=False):
    id: str  # canonical ID (e.g., arxiv_id, sha, OSTI id)
    title: str
    url: str
    local_path: str
    full_text: str
    extra: Dict[str, Any]


class AcquisitionState(TypedDict, total=False):
    query: str
    context: str
    items: List[ItemMetadata]
    summaries: List[str]
    final_summary: str


# ---------- Small Utilities reused across agents ----------


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-_.]+", "_", s)
    return s[:240]


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def remove_surrogates(text: str) -> str:
    return re.sub(r"[\ud800-\udfff]", "", text)


def _looks_like_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path.lower().endswith(".pdf")


def _download(url: str, dest_path: str, timeout: int = 20) -> str:
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    return dest_path


def _load_pdf_text(path: str) -> str:
    loader = PyPDFLoader(path)
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)


# def _basic_readable_text_from_html(html: str) -> str:
#     soup = BeautifulSoup(html, "html.parser")
#     # Drop scripts/styles/navs for a crude readability
#     for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
#         tag.decompose()
#     # Keep title for context
#     title = soup.title.get_text(strip=True) if soup.title else ""
#     # Join paragraphs
#     texts = [
#         p.get_text(" ", strip=True)
#         for p in soup.find_all(["p", "h1", "h2", "h3", "li", "figcaption"])
#     ]
#     body = "\n".join(t for t in texts if t)
#     return (title + "\n\n" + body).strip()


def describe_image(image: Image.Image) -> str:
    if OpenAI is None:
        return ""
    client = OpenAI()
    buf = BytesIO()
    image.save(buf, format="PNG")
    import base64

    img_b64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a scientific assistant who explains plots and scientific diagrams.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this scientific image or plot in detail.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        },
                    },
                ],
            },
        ],
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def extract_and_describe_images(
    pdf_path: str, max_images: int = 5
) -> List[str]:
    descriptions: List[str] = []
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return [f"[Image extraction failed: {e}]"]

    count = 0
    for pi in range(len(doc)):
        if count >= max_images:
            break
        page = doc[pi]
        for ji, img in enumerate(page.get_images(full=True)):
            if count >= max_images:
                break
            xref = img[0]
            base = doc.extract_image(xref)
            image = Image.open(BytesIO(base["image"]))
            try:
                desc = describe_image(image) if OpenAI else ""
            except Exception as e:
                desc = f"[Error: {e}]"
            descriptions.append(f"Page {pi + 1}, Image {ji + 1}: {desc}")
            count += 1
    return descriptions


# ---------- The Parent / Generic Agent ----------


class BaseAcquisitionAgent(BaseAgent):
    """
    A generic "acquire-then-summarize-or-RAG" agent.

    Subclasses must implement:
      - _search(self, query) -> List[dict-like]: lightweight hits
      - _materialize(self, hit) -> ItemMetadata: download or scrape and return populated item
      - _id(self, hit_or_item) -> str: stable id for caching/file naming
      - _citation(self, item) -> str: human-readable citation string

    Optional hooks:
      - _postprocess_text(self, text, local_path) -> str (e.g., image interpretation)
      - _filter_hit(self, hit) -> bool
    """

    def __init__(
        self,
        llm: BaseChatModel = init_chat_model("openai:gpt-5-mini"),
        *,
        summarize: bool = True,
        rag_embedding=None,
        process_images: bool = True,
        max_results: int = 5,
        database_path: str = "acq_db",
        summaries_path: str = "acq_summaries",
        vectorstore_path: str = "acq_vectorstores",
        download: bool = True,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.summarize = summarize
        self.rag_embedding = rag_embedding
        self.process_images = process_images
        self.max_results = max_results
        self.database_path = database_path
        self.summaries_path = summaries_path
        self.vectorstore_path = vectorstore_path
        self.download = download

        os.makedirs(self.database_path, exist_ok=True)
        os.makedirs(self.summaries_path, exist_ok=True)

        self._action = self._build_graph()

    # ---- abstract-ish methods ----
    def _search(self, query: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _materialize(self, hit: Dict[str, Any]) -> ItemMetadata:
        raise NotImplementedError

    def _id(self, hit_or_item: Dict[str, Any]) -> str:
        raise NotImplementedError

    def _citation(self, item: ItemMetadata) -> str:
        # Subclass should format its ideal citation; fallback is ID or URL.
        return item.get("id") or item.get("url", "Unknown Source")

    # ---- optional hooks ----
    def _filter_hit(self, hit: Dict[str, Any]) -> bool:
        return True

    def _postprocess_text(self, text: str, local_path: Optional[str]) -> str:
        # Default: optionally add image descriptions for PDFs
        if (
            self.process_images
            and local_path
            and local_path.lower().endswith(".pdf")
        ):
            try:
                descs = extract_and_describe_images(local_path)
                if any(descs):
                    text += "\n\n[Image Interpretations]\n" + "\n".join(descs)
            except Exception:
                pass
        return text

    # ---- shared nodes ----
    def _fetch_items(self, query: str) -> List[ItemMetadata]:
        hits = self._search(query)[: self.max_results] if self.download else []
        items: List[ItemMetadata] = []

        # If not downloading/scraping, try to load whatever is cached in database_path.
        if not self.download:
            for fname in os.listdir(self.database_path):
                if fname.lower().endswith((".pdf", ".txt", ".html")):
                    item_id = os.path.splitext(fname)[0]
                    local_path = os.path.join(self.database_path, fname)
                    full_text = ""
                    try:
                        if fname.lower().endswith(".pdf"):
                            full_text = _load_pdf_text(local_path)
                        else:
                            with open(
                                local_path,
                                "r",
                                encoding="utf-8",
                                errors="ignore",
                            ) as f:
                                full_text = f.read()
                    except Exception as e:
                        full_text = f"[Error reading cached file: {e}]"
                    full_text = self._postprocess_text(full_text, local_path)
                    items.append({
                        "id": item_id,
                        "local_path": local_path,
                        "full_text": full_text,
                    })
            return items

        # Normal path: search → materialize each
        with ThreadPoolExecutor(max_workers=min(32, max(1, len(hits)))) as ex:
            futures = [
                ex.submit(self._materialize, h)
                for h in hits
                if self._filter_hit(h)
            ]
            for fut in as_completed(futures):
                try:
                    item = fut.result()
                    items.append(item)
                except Exception as e:
                    items.append({
                        "id": _hash(str(time.time())),
                        "full_text": f"[Error: {e}]",
                    })
        return items

    def _fetch_node(self, state: AcquisitionState) -> AcquisitionState:
        items = self._fetch_items(state["query"])
        return {**state, "items": items}

    def _summarize_node(self, state: AcquisitionState) -> AcquisitionState:
        prompt = ChatPromptTemplate.from_template("""
        You are an assistant responsible for summarizing retrieved content in the context of this task: {context}

        Summarize the content below:

        {retrieved_content}
        """)
        chain = prompt | self.llm | StrOutputParser()

        if "items" not in state or not state["items"]:
            return {**state, "summaries": None}

        summaries: List[Optional[str]] = [None] * len(state["items"])

        def process(i: int, item: ItemMetadata):
            item_id = item.get("id", f"item_{i}")
            out_path = os.path.join(
                self.summaries_path, f"{_safe_filename(item_id)}_summary.txt"
            )
            try:
                cleaned = remove_surrogates(item.get("full_text", ""))
                summary = chain.invoke(
                    {"retrieved_content": cleaned, "context": state["context"]},
                    config=self.build_config(tags=["acq", "summarize_each"]),
                )
            except Exception as e:
                summary = f"[Error summarizing item {item_id}: {e}]"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(summary)
            return i, summary

        with ThreadPoolExecutor(max_workers=min(32, len(state["items"]))) as ex:
            futures = [
                ex.submit(process, i, it) for i, it in enumerate(state["items"])
            ]
            for fut in as_completed(futures):
                i, s = fut.result()
                summaries[i] = s

        return {**state, "summaries": summaries}  # type: ignore

    def _rag_node(self, state: AcquisitionState) -> AcquisitionState:
        new_state = state.copy()
        rag_agent = RAGAgent(
            llm=self.llm,
            embedding=self.rag_embedding,
            database_path=self.database_path,
        )
        new_state["final_summary"] = rag_agent.invoke(context=state["context"])[
            "summary"
        ]
        return new_state

    def _aggregate_node(self, state: AcquisitionState) -> AcquisitionState:
        if not state.get("summaries") or not state.get("items"):
            return {**state, "final_summary": None}

        blocks: List[str] = []
        for idx, (item, summ) in enumerate(
            zip(state["items"], state["summaries"])
        ):  # type: ignore
            cite = self._citation(item)
            blocks.append(f"[{idx + 1}] {cite}\n\nSummary:\n{summ}")

        combined = "\n\n" + ("\n\n" + "-" * 40 + "\n\n").join(blocks)
        with open(
            os.path.join(self.summaries_path, "summaries_combined.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(combined)

        prompt = ChatPromptTemplate.from_template("""
        You are a scientific assistant extracting insights from multiple summaries.

        Here are the summaries:

        {Summaries}

        Your task is to read all the summaries and provide a response to this task: {context}
        """)
        chain = prompt | self.llm | StrOutputParser()

        final_summary = chain.invoke(
            {"Summaries": combined, "context": state["context"]},
            config=self.build_config(tags=["acq", "aggregate"]),
        )
        with open(
            os.path.join(self.summaries_path, "final_summary.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(final_summary)

        return {**state, "final_summary": final_summary}

    def _build_graph(self):
        graph = StateGraph(AcquisitionState)
        self.add_node(graph, self._fetch_node)

        if self.summarize:
            if self.rag_embedding:
                self.add_node(graph, self._rag_node)
                graph.set_entry_point("_fetch_node")
                graph.add_edge("_fetch_node", "_rag_node")
                graph.set_finish_point("_rag_node")
            else:
                self.add_node(graph, self._summarize_node)
                self.add_node(graph, self._aggregate_node)

                graph.set_entry_point("_fetch_node")
                graph.add_edge("_fetch_node", "_summarize_node")
                graph.add_edge("_summarize_node", "_aggregate_node")
                graph.set_finish_point("_aggregate_node")
        else:
            graph.set_entry_point("_fetch_node")
            graph.set_finish_point("_fetch_node")

        return graph.compile(checkpointer=self.checkpointer)

    def _invoke(
        self,
        inputs: Mapping[str, Any],
        *,
        summarize: bool | None = None,
        recursion_limit: int = 1000,
        **_,
    ) -> str:
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )

        # alias support like your ArxivAgent
        if "query" not in inputs:
            if "arxiv_search_query" in inputs:
                inputs = dict(inputs)
                inputs["query"] = inputs.pop("arxiv_search_query")
            else:
                raise KeyError(
                    "Missing 'query' in inputs (alias 'arxiv_search_query' also accepted)."
                )

        result = self._action.invoke(inputs, config)
        use_summary = self.summarize if summarize is None else summarize
        return (
            result.get("final_summary", "No summary generated.")
            if use_summary
            else "\n\nFinished fetching items!"
        )


# ---------- Concrete: Web Search via ddgs ----------


class WebSearchAgent(BaseAcquisitionAgent):
    """
    Uses DuckDuckGo Search (ddgs) to find pages, downloads HTML or PDFs,
    extracts text, and then follows the same summarize/RAG path.
    """

    def __init__(self, *args, user_agent: str = "Mozilla/5.0", **kwargs):
        super().__init__(*args, **kwargs)
        self.user_agent = user_agent
        if DDGS is None:
            raise ImportError(
                "duckduckgo-search (DDGS) is required for WebSearchAgentGeneric."
            )

    def _id(self, hit_or_item: Dict[str, Any]) -> str:
        url = hit_or_item.get("href") or hit_or_item.get("url") or ""
        return (
            _hash(url)
            if url
            else hit_or_item.get("id", _hash(json.dumps(hit_or_item)))
        )

    def _citation(self, item: ItemMetadata) -> str:
        t = item.get("title", "") or ""
        u = item.get("url", "") or ""
        return f"{t} ({u})" if t else (u or item.get("id", "Web result"))

    def _search(self, query: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                query, max_results=self.max_results, backend="duckduckgo"
            ):
                # r keys typically: title, href, body
                results.append(r)
        return results

    def _materialize(self, hit: Dict[str, Any]) -> ItemMetadata:
        url = hit.get("href") or hit.get("url")
        title = hit.get("title", "")
        if not url:
            return {"id": self._id(hit), "title": title, "full_text": ""}

        headers = {"User-Agent": self.user_agent}
        local_path = ""
        full_text = ""
        item_id = self._id(hit)

        try:
            if _looks_like_pdf_url(url):
                local_path = os.path.join(
                    self.database_path, _safe_filename(item_id) + ".pdf"
                )
                _download(url, local_path)
                full_text = _load_pdf_text(local_path)
            else:
                r = requests.get(url, headers=headers, timeout=20)
                r.raise_for_status()
                html = r.text
                local_path = os.path.join(
                    self.database_path, _safe_filename(item_id) + ".html"
                )
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(html)
                full_text = extract_main_text_only(html)
                # full_text = _basic_readable_text_from_html(html)
        except Exception as e:
            full_text = f"[Error retrieving {url}: {e}]"

        full_text = self._postprocess_text(full_text, local_path)
        return {
            "id": item_id,
            "title": title,
            "url": url,
            "local_path": local_path,
            "full_text": full_text,
            "extra": {"snippet": hit.get("body", "")},
        }


# ---------- Concrete: OSTI.gov Agent (minimal, adaptable) ----------


class OSTIAgent(BaseAcquisitionAgent):
    """
    Minimal OSTI.gov acquisition agent.

    NOTE:
      - OSTI provides search endpoints that can return metadata including full-text links.
      - Depending on your environment, you may prefer the public API or site scraping.
      - Here we assume a JSON API that yields results with keys like:
            {'osti_id': '12345', 'title': '...', 'pdf_url': 'https://...pdf', 'landing_page': 'https://...'}
        Adapt field names if your OSTI integration differs.

    Customize `_search` and `_materialize` to match your OSTI access path.
    """

    def __init__(
        self,
        *args,
        api_base: str = "https://www.osti.gov/api/v1/records",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.api_base = api_base

    def _id(self, hit_or_item: Dict[str, Any]) -> str:
        if "osti_id" in hit_or_item:
            return str(hit_or_item["osti_id"])
        if "id" in hit_or_item:
            return str(hit_or_item["id"])
        if "landing_page" in hit_or_item:
            return _hash(hit_or_item["landing_page"])
        return _hash(json.dumps(hit_or_item))

    def _citation(self, item: ItemMetadata) -> str:
        t = item.get("title", "") or ""
        oid = item.get("id", "")
        return f"OSTI {oid}: {t}" if t else f"OSTI {oid}"

    def _search(self, query: str) -> List[Dict[str, Any]]:
        """
        Adjust params to your OSTI setup. This call is intentionally simple;
        add paging/auth as needed.
        """
        params = {
            "q": query,
            "size": self.max_results,
        }
        try:
            r = requests.get(self.api_base, params=params, timeout=25)
            r.raise_for_status()
            data = r.json()
            # Normalize to a list of hits; adapt key if your API differs.
            if isinstance(data, dict) and "records" in data:
                hits = data["records"]
            elif isinstance(data, list):
                hits = data
            else:
                hits = []
            return hits[: self.max_results]
        except Exception as e:
            return [
                {
                    "id": _hash(query + str(time.time())),
                    "title": "Search error",
                    "error": str(e),
                }
            ]

    def _materialize(self, hit: Dict[str, Any]) -> ItemMetadata:
        item_id = self._id(hit)
        title = hit.get("title") or hit.get("title_public", "") or ""
        landing = None
        local_path = ""
        full_text = ""

        try:
            pdf_url, landing_used, _ = resolve_pdf_from_osti_record(
                hit,
                headers={"User-Agent": "Mozilla/5.0"},
                unpaywall_email=os.environ.get("UNPAYWALL_EMAIL"),  # optional
            )

            if pdf_url:
                # Try to download as PDF (validate headers)
                with requests.get(
                    pdf_url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=25,
                    allow_redirects=True,
                    stream=True,
                ) as r:
                    r.raise_for_status()
                    if _is_pdf_response(r):
                        fname = _derive_filename_from_cd_or_url(
                            r, f"osti_{item_id}.pdf"
                        )
                        local_path = os.path.join(self.database_path, fname)
                        _download_stream_to(local_path, r)
                        # Extract PDF text
                        try:
                            from langchain_community.document_loaders import (
                                PyPDFLoader,
                            )

                            loader = PyPDFLoader(local_path)
                            pages = loader.load()
                            full_text = "\n".join(p.page_content for p in pages)
                        except Exception as e:
                            full_text = (
                                f"[Downloaded but text extraction failed: {e}]"
                            )
                    else:
                        # Not a PDF; treat as HTML landing and parse text
                        landing = r.url
                        r.close()
            # If we still have no text, try scraping the DOE PAGES landing or citation page
            if not full_text:
                # Prefer DOE PAGES landing if present, else OSTI biblio
                landing = (
                    landing
                    or landing_used
                    or next(
                        (
                            link.get("href")
                            for link in hit.get("links", [])
                            if link.get("rel")
                            in ("citation_doe_pages", "citation")
                        ),
                        None,
                    )
                )
                if landing:
                    soup = _get_soup(
                        landing,
                        timeout=25,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    html_text = soup.get_text(" ", strip=True)
                    full_text = html_text[:1_000_000]  # keep it bounded
                    # Save raw HTML for cache/inspection
                    local_path = os.path.join(
                        self.database_path, f"{item_id}.html"
                    )
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(str(soup))
                else:
                    full_text = "[No PDF or landing page text available.]"

        except Exception as e:
            full_text = f"[Error materializing OSTI {item_id}: {e}]"

        full_text = self._postprocess_text(full_text, local_path)
        return {
            "id": item_id,
            "title": title,
            "url": landing,
            "local_path": local_path,
            "full_text": full_text,
            "extra": {"raw_hit": hit},
        }


# ---------- (Optional) Refactor your ArxivAgent to reuse the parent ----------


class ArxivAgent(BaseAcquisitionAgent):
    """
    Drop-in replacement for your existing ArxivAgent that reuses the generic flow.
    Keeps the same behaviors (download PDFs, image processing, summarization/RAG).
    """

    def __init__(
        self,
        llm: BaseChatModel = init_chat_model("openai:gpt-5-mini"),
        *,
        process_images: bool = True,
        max_results: int = 3,
        download: bool = True,
        rag_embedding=None,
        database_path="arxiv_papers",
        summaries_path="arxiv_generated_summaries",
        vectorstore_path="arxiv_vectorstores",
        **kwargs,
    ):
        super().__init__(
            llm,
            rag_embedding=rag_embedding,
            process_images=process_images,
            max_results=max_results,
            database_path=database_path,
            summaries_path=summaries_path,
            vectorstore_path=vectorstore_path,
            download=download,
            **kwargs,
        )

    def _id(self, hit_or_item: Dict[str, Any]) -> str:
        # hits from arXiv feed have 'id' like ".../abs/XXXX.YYYY"
        arxiv_id = hit_or_item.get("arxiv_id")
        if arxiv_id:
            return arxiv_id
        feed_id = hit_or_item.get("id", "")
        if "/abs/" in feed_id:
            return feed_id.split("/abs/")[-1]
        return _hash(json.dumps(hit_or_item))

    def _citation(self, item: ItemMetadata) -> str:
        return f"ArXiv ID: {item.get('id', '?')}"

    def _search(self, query: str) -> List[Dict[str, Any]]:
        enc = quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{enc}&start=0&max_results={self.max_results}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            entries = feed.entries if hasattr(feed, "entries") else []
            hits = []
            for e in entries:
                full_id = e.id.split("/abs/")[-1]
                hits.append({
                    "id": e.id,
                    "title": e.title.strip(),
                    "arxiv_id": full_id.split("/")[-1],
                })
            return hits
        except Exception as e:
            return [
                {
                    "id": _hash(query + str(time.time())),
                    "title": "Search error",
                    "error": str(e),
                }
            ]

    def _materialize(self, hit: Dict[str, Any]) -> ItemMetadata:
        arxiv_id = self._id(hit)
        title = hit.get("title", "")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        local_path = os.path.join(self.database_path, f"{arxiv_id}.pdf")
        full_text = ""
        try:
            _download(pdf_url, local_path)
            full_text = _load_pdf_text(local_path)
        except Exception as e:
            full_text = f"[Error loading ArXiv {arxiv_id}: {e}]"
        full_text = self._postprocess_text(full_text, local_path)
        return {
            "id": arxiv_id,
            "title": title,
            "url": pdf_url,
            "local_path": local_path,
            "full_text": full_text,
        }
