# generic_acquisition_agents.py

import asyncio
import hashlib
import json
import logging
import operator
import os
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, NotRequired, Optional, TypedDict
from urllib.parse import quote, urlparse

import feedparser

logger = logging.getLogger(__name__)

# PDF & Vision extras (match your existing stack)
import pymupdf
import requests
from langchain.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Overwrite, Send
from PIL import Image

from ursa.agents.base import BaseAgent
from ursa.agents.rag_agent import RAGAgent
from ursa.util.http import build_httpx_client
from ursa.util.parse import (
    _derive_filename_from_cd_or_url,
    _download_stream_to,
    _get_soup,
    _is_pdf_response,
    extract_main_text_only,
    read_pdf,
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
    extra: dict[str, Any]


class AcquisitionState(TypedDict, total=False):
    query: str
    context: str
    items: list[ItemMetadata]
    summaries: list[str]
    final_summary: str | None
    source_tasks: list["SourceTask"]
    processed_sources: Annotated[list["ProcessedSource"], operator.add]


class SourceTask(TypedDict):
    index: int
    context: str
    hit: NotRequired[dict[str, Any]]
    cached_path: NotRequired[str]


class ProcessedSource(TypedDict):
    index: int
    item: ItemMetadata
    summary: str | None


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
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
    return dest_path


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
    client = OpenAI(http_client=build_httpx_client())
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
) -> list[str]:
    descriptions: list[str] = []
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return [f"[Image extraction failed: {e}]"]

    count = 0
    with doc:
        for pi in range(len(doc)):
            if count >= max_images:
                break
            page = doc[pi]
            for ji, img in enumerate(page.get_images(full=True)):
                if count >= max_images:
                    break
                xref = img[0]
                base = doc.extract_image(xref)
                with Image.open(BytesIO(base["image"])) as image:
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

    state_type = AcquisitionState

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        summarize: bool = True,
        rag_embedding=None,
        process_images: bool = True,
        max_results: int = 5,
        database_path: str = "acq_db",
        summaries_path: str = "acq_summaries",
        vectorstore_path: str = "acq_vectorstores",
        num_threads: int = 4,
        max_concurrency: int | None = None,
        download: bool = True,
        **kwargs,
    ):
        self.max_concurrency = max(
            1, num_threads if max_concurrency is None else max_concurrency
        )
        self.num_threads = self.max_concurrency
        super().__init__(llm, **kwargs)
        self.summarize = summarize
        self.rag_embedding = rag_embedding
        self.process_images = process_images
        self.max_results = max_results
        self.database_path = self.den / database_path
        self.summaries_path = self.den / summaries_path
        self.vectorstore_path = self.den / vectorstore_path
        self.download = download

        self.database_path.mkdir(exist_ok=True, parents=True)
        self.summaries_path.mkdir(exist_ok=True, parents=True)

    def build_config(self, **overrides) -> dict:
        """Use LangGraph's executor to bound concurrent source tasks."""
        overrides.setdefault("max_concurrency", self.max_concurrency)
        return super().build_config(**overrides)

    # ---- abstract-ish methods ----
    def _search(self, query: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _materialize(self, hit: dict[str, Any]) -> ItemMetadata:
        raise NotImplementedError

    async def _asearch(self, query: str) -> list[dict[str, Any]]:
        """Run a synchronous source search without blocking the event loop."""
        return await asyncio.to_thread(self._search, query)

    async def _amaterialize(self, hit: dict[str, Any]) -> ItemMetadata:
        """Materialize one source without blocking the event loop."""
        return await asyncio.to_thread(self._materialize, hit)

    def _id(self, hit_or_item: dict[str, Any]) -> str:
        raise NotImplementedError

    def _citation(self, item: ItemMetadata) -> str:
        # Subclass should format its ideal citation; fallback is ID or URL.
        return item.get("id") or item.get("url", "Unknown Source")

    # ---- optional hooks ----
    def _filter_hit(self, hit: dict[str, Any]) -> bool:
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
    async def _load_cached_item(self, path: Path) -> ItemMetadata:
        def load() -> ItemMetadata:
            try:
                if path.suffix.lower() == ".pdf":
                    full_text = read_pdf(str(path))
                else:
                    full_text = path.read_text(
                        encoding="utf-8", errors="ignore"
                    )
            except Exception as exc:  # noqa: BLE001
                full_text = f"[Error reading cached file: {exc}]"
            full_text = self._postprocess_text(full_text, str(path))
            return {
                "id": path.stem,
                "local_path": str(path),
                "full_text": full_text,
            }

        return await asyncio.to_thread(load)

    def _normalize_inputs(self, inputs) -> AcquisitionState:
        if isinstance(inputs, str):
            return AcquisitionState(context=inputs)
        elif isinstance(inputs, dict):
            return inputs
        else:
            raise TypeError(f"Invalid input for {self.__class__.__name__}")

    def format_result(self, state: AcquisitionState) -> str:
        if summary := state.get("final_summary"):
            return summary

        # Fallback to dumping an empty string if `self.summarize=False`.
        # This only happens if a user disables summarization when configuring
        # URSA or if the final_summary is an empty string itself.
        return ""

    async def _search_query(self, state: AcquisitionState) -> AcquisitionState:
        """Generate a search query from the input search task (context)"""
        existing = state.get("query")
        if existing:
            return {}

        context = state["context"]
        query = await self.llm.ainvoke(
            f"The user stated {context}. Generate between 1 and 8 words for a search query to address the users need. Return only the words to search."
        )
        return {"query": query.text or context}

    async def _search_sources(
        self, state: AcquisitionState
    ) -> AcquisitionState:
        """Create source tasks for LangGraph's fan-out router."""
        context = state["context"]
        if not self.download:
            paths = sorted(
                path
                for path in self.database_path.iterdir()
                if path.suffix.lower() in {".pdf", ".txt", ".html"}
            )
            tasks = [
                SourceTask(
                    index=index,
                    context=context,
                    cached_path=str(path),
                )
                for index, path in enumerate(paths)
            ]
        else:
            hits = (await self._asearch(state["query"]))[: self.max_results]
            tasks = [
                SourceTask(index=index, context=context, hit=hit)
                for index, hit in enumerate(hits)
                if self._filter_hit(hit)
            ]
        return {
            "source_tasks": tasks,
            "processed_sources": Overwrite([]),
        }

    def _fan_out_sources(self, state: AcquisitionState) -> list[Send] | str:
        """Dispatch one LangGraph task per source, or reduce an empty set."""
        tasks = state.get("source_tasks", [])
        if not tasks:
            return "_reduce_sources"
        return [Send("_process_source", task) for task in tasks]

    async def _summarize_source(
        self, item: ItemMetadata, index: int, context: str
    ) -> str:
        """Summarize one source inside its LangGraph fan-out task."""
        item_id = item.get("id", f"item_{index}")
        prompt = ChatPromptTemplate.from_template("""
        You are an assistant responsible for summarizing retrieved content in the context of this task: {context}

        Summarize the content below:

        {retrieved_content}
        """)
        chain = prompt | self.llm | StrOutputParser()
        try:
            cleaned = remove_surrogates(item.get("full_text", ""))
            summary = await chain.ainvoke(
                {"retrieved_content": cleaned, "context": context},
                config=self.build_config(tags=["acq", "summarize_each"]),
            )
        except Exception as exc:  # noqa: BLE001
            summary = f"[Error summarizing item {item_id}: {exc}]"

        out_path = self.summaries_path / (
            f"{_safe_filename(item_id)}_summary.txt"
        )
        try:
            await asyncio.to_thread(
                out_path.write_text, summary, encoding="utf-8"
            )
        except (OSError, UnicodeError):
            # Cache persistence is best-effort; the in-memory result is still
            # useful and should be allowed to reach the reducer.
            pass
        return summary

    async def _process_source(self, task: SourceTask) -> AcquisitionState:
        """Materialize and optionally summarize one fanned-out source."""
        index = task["index"]
        hit = task.get("hit")
        if hit is None:
            cached_path = task.get("cached_path")
            if cached_path is None:
                raise ValueError(
                    "A source task must contain either 'hit' or 'cached_path'"
                )
            item = await self._load_cached_item(Path(cached_path))
        else:
            try:
                item = await self._amaterialize(hit)
            except Exception as exc:  # noqa: BLE001
                item: ItemMetadata = {
                    "id": self._id(hit),
                    "title": str(hit.get("title") or ""),
                    "url": str(hit.get("href") or hit.get("url") or ""),
                    "full_text": f"[Error: {exc}]",
                }

        summary = (
            await self._summarize_source(item, index, task["context"])
            if self.summarize and self.rag_embedding is None
            else None
        )
        processed = ProcessedSource(
            index=index,
            item=item,
            summary=summary,
        )
        return {"processed_sources": [processed]}

    def _rag_node(self, state: AcquisitionState) -> AcquisitionState:
        new_state = state.copy()
        rag_agent = RAGAgent(
            llm=self.llm,
            workspace=self.den,
            embedding=self.rag_embedding,
            vectorstore_path="rag_vectorstore",
            database_path=self.database_path.name,
        )
        new_state["final_summary"] = rag_agent.invoke(context=state["context"])[
            "summary"
        ]
        return new_state

    async def _arag_node(self, state: AcquisitionState) -> AcquisitionState:
        return await asyncio.to_thread(self._rag_node, state)

    async def _aggregate_sources(
        self,
        items: list[ItemMetadata],
        summaries: list[str],
        context: str,
    ) -> str | None:
        """Reduce ordered source summaries to the final answer."""
        if not summaries or not items:
            return None

        blocks: list[str] = []
        for idx, (item, summ) in enumerate(zip(items, summaries)):
            cite = self._citation(item)
            blocks.append(f"[{idx + 1}] {cite}\n\nSummary:\n{summ}")

        combined = "\n\n" + ("\n\n" + "-" * 40 + "\n\n").join(blocks)
        combined_path = self.summaries_path / "summaries_combined.txt"
        try:
            await asyncio.to_thread(
                combined_path.write_text, combined, encoding="utf-8"
            )
        except (OSError, UnicodeError):
            pass

        prompt = ChatPromptTemplate.from_template("""
        You are a scientific assistant extracting insights from multiple summaries.

        Here are the summaries:

        {Summaries}

        Your task is to read all the summaries and provide a response to this task: {context}
        """)
        chain = prompt | self.llm | StrOutputParser()

        final_summary = await chain.ainvoke(
            {"Summaries": combined, "context": context},
            config=self.build_config(tags=["acq", "aggregate"]),
        )
        final_path = self.summaries_path / "final_summary.txt"
        try:
            await asyncio.to_thread(
                final_path.write_text, final_summary, encoding="utf-8"
            )
        except (OSError, UnicodeError):
            pass

        return final_summary

    async def _reduce_sources(
        self, state: AcquisitionState
    ) -> AcquisitionState:
        """Collect LangGraph reducer output and produce the final state."""
        processed = sorted(
            state.get("processed_sources", []),
            key=lambda source: source["index"],
        )
        items = [source["item"] for source in processed]
        summaries = [
            source["summary"]
            for source in processed
            if source["summary"] is not None
        ]
        result: AcquisitionState = {
            "items": items,
            "summaries": summaries,
        }
        if self.summarize and self.rag_embedding is not None:
            rag_state = await self._arag_node({**state, "items": items})
            result["final_summary"] = rag_state.get("final_summary")
        elif self.summarize:
            result["final_summary"] = await self._aggregate_sources(
                items, summaries, state["context"]
            )
        return result

    def _invoke(self, input, **config):
        """Run the async-only graph for legacy synchronous callers."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._ainvoke(input, **config))
        raise RuntimeError(
            "Acquisition agents are async-first; use `await agent.ainvoke(...)` "
            "inside an active event loop."
        )

    def _stream(self, input, **config):
        """Reject sync streaming because all acquisition nodes are async."""
        raise RuntimeError(
            "Acquisition agents are async-first and do not support synchronous "
            "`.stream()`. Use `await agent.ainvoke(...)` instead."
        )

    def _build_graph(self):
        self.add_node(self._search_query)
        self.add_node(self._search_sources)
        self.add_node(self._process_source)
        self.add_node(self._reduce_sources)
        self.graph.set_entry_point("_search_query")
        self.graph.add_edge("_search_query", "_search_sources")
        self.graph.add_conditional_edges(
            "_search_sources",
            self._fan_out_sources,
            ["_process_source", "_reduce_sources"],
        )
        self.graph.add_edge("_process_source", "_reduce_sources")
        self.graph.set_finish_point("_reduce_sources")


# ---------- Concrete: Web Search via ddgs ----------


class WebSearchAgent(BaseAcquisitionAgent):
    """
    Uses DuckDuckGo Search (ddgs) to find pages, downloads HTML or PDFs,
    extracts text, and then follows the same summarize/RAG path.

    If the ``SERPBASE_API_KEY`` environment variable is set, search results
    come from the SerpBase Google Search API instead of DuckDuckGo. When the
    key is unset (or the API call fails), the agent falls back to DDGS, so
    existing behavior is unchanged.
    """

    def __init__(self, *args, user_agent: str = "Mozilla/5.0", **kwargs):
        super().__init__(*args, **kwargs)
        self.user_agent = user_agent
        self.serpbase_api_key = os.environ.get("SERPBASE_API_KEY", "")
        if DDGS is None:
            raise ImportError(
                "duckduckgo-search (DDGS) is required for WebSearchAgentGeneric."
            )

    def _serpbase_search(self, query: str) -> list[dict[str, Any]]:
        """Search Google via the SerpBase API. Returns [] on any failure."""
        try:
            resp = requests.get(
                "https://api.serpbase.dev/google/search",
                params={
                    "q": query,
                    "api_key": self.serpbase_api_key,
                    "num": self.max_results,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("SerpBase search failed (%s); falling back to DDGS.", e)
            return []
        results: list[dict[str, Any]] = []
        for r in data.get("organic_results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "href": r.get("link", ""),
                    "body": r.get("snippet", ""),
                    "position": r.get("position"),
                }
            )
        return results

    def _search(self, query: str) -> list[dict[str, Any]]:
        if self.serpbase_api_key:
            results = self._serpbase_search(query)
            if results:
                return results
            logger.info("SerpBase returned no results; falling back to DDGS.")
        results: list[dict[str, Any]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                query, max_results=self.max_results, backend="auto"
            ):
                # r keys typically: title, href, body
                results.append(r)
        return results

    def _materialize(self, hit: dict[str, Any]) -> ItemMetadata:
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
                full_text = read_pdf(local_path)
            else:
                with requests.get(url, headers=headers, timeout=20) as response:
                    response.raise_for_status()
                    html = response.text
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

    def _id(self, hit_or_item: dict[str, Any]) -> str:
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

    def _search(self, query: str) -> list[dict[str, Any]]:
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
                    "id": _hash(query + ":search-error"),
                    "title": "Search error",
                    "error": str(e),
                }
            ]

    def _materialize(self, hit: dict[str, Any]) -> ItemMetadata:
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
                            full_text = read_pdf(local_path)
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
        item: ItemMetadata = {
            "id": item_id,
            "title": title,
            "local_path": local_path,
            "full_text": full_text,
            "extra": {"raw_hit": hit},
        }
        if landing:
            item["url"] = landing
        return item


# ---------- (Optional) Refactor your ArxivAgent to reuse the parent ----------


class ArxivAgent(BaseAcquisitionAgent):
    """
    Drop-in replacement for your existing ArxivAgent that reuses the generic flow.
    Keeps the same behaviors (download PDFs, image processing, summarization/RAG).
    """

    def __init__(
        self,
        llm: BaseChatModel,
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

    def _id(self, hit_or_item: dict[str, Any]) -> str:
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

    def _search(self, query: str) -> list[dict[str, Any]]:
        enc = quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{enc}&start=0&max_results={self.max_results}"
        try:
            with requests.get(url, timeout=15) as response:
                response.raise_for_status()
                feed = feedparser.parse(response.content)
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
                    "id": _hash(query + ":search-error"),
                    "title": "Search error",
                    "error": str(e),
                }
            ]

    def _materialize(self, hit: dict[str, Any]) -> ItemMetadata:
        arxiv_id = self._id(hit)
        title = hit.get("title", "")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        local_path = os.path.join(self.database_path, f"{arxiv_id}.pdf")
        full_text = ""
        try:
            _download(pdf_url, local_path)
            full_text = read_pdf(local_path)
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
