"""Locate and fetch model-card materials.

A *source reference* is any string a user or agent can point at a model:

- ``hf://org/name``, ``hf:org/name``, or a ``https://huggingface.co/...``
  link (README, ``config.json``, and Hub metadata are fetched),
- a local file or directory (Markdown card, URSA-native YAML/JSON card,
  ``config.json``),
- any other ``http(s)`` URL to a Markdown, YAML, or JSON card.

MCP-served models (for example Nomad SciFM tools) are turned into documents
by :func:`mcp_tool_document`. Fetched content is treated as untrusted data:
it is parsed, never executed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from ursa.model_cards.extract import CardDocument
from ursa.model_cards.schema import CardSource

logger = logging.getLogger(__name__)

MAX_CARD_BYTES = 2_000_000
CARD_FILENAMES = (
    "model_card.yaml",
    "model_card.yml",
    "model_card.json",
    "MODEL_CARD.md",
    "model_card.md",
    "README.md",
    "readme.md",
)
_HF_URL = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/(?!datasets/|spaces/)"
    r"(?P<repo>[\w.-]+/[\w.-]+)(?:/(?:tree|blob|resolve|raw)/(?P<rev>[\w.-]+))?"
)
_HF_SCHEME = re.compile(
    r"^hf:(?://)?(?P<repo>[\w.-]+/[\w.-]+)(?:@(?P<rev>[\w.-]+))?$"
)
_BARE_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_GITHUB_BLOB = re.compile(
    r"^https?://github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/blob/(?P<rest>.+)$"
)
_URL_IN_TEXT = re.compile(r"https?://[^\s)>\]\"'`]+")
_HF_IN_TEXT = re.compile(r"\bhf:(?://)?[\w.-]+/[\w.-]+(?:@[\w.-]+)?")
_BACKTICK_TOKEN = re.compile(r"`([^`\s]+)`")
_PATH_TOKEN = re.compile(
    r"(?:^|\s)((?:\.{0,2}/|~/)?[\w.-]+(?:/[\w.-]+)*\.(?:md|markdown|yaml|yml|json))\b"
)
# Slash-containing tokens count as paths only when they exist, so phrases
# such as "input/output" never become sources.
_SLASH_TOKEN = re.compile(
    r"(?:^|\s)((?:\.{0,2}/|~/|/)?[\w.-]+(?:/[\w.-]+)+/?)(?=[\s.,;:!?)]|$)"
)


@dataclass(frozen=True)
class SourceRef:
    kind: Literal["huggingface", "url", "file", "directory"]
    value: str
    revision: str | None = None

    @property
    def label(self) -> str:
        if self.kind == "huggingface":
            return f"hf://{self.value}" + (
                f"@{self.revision}" if self.revision else ""
            )
        return self.value


def parse_source(ref: str, workspace: Path | None = None) -> SourceRef | None:
    """Classify a source reference; return None when it is not usable."""
    ref = (ref or "").strip().strip("`'\"<>").rstrip(".,;")
    if not ref:
        return None
    match = _HF_SCHEME.match(ref)
    if match:
        return SourceRef("huggingface", match.group("repo"), match.group("rev"))
    match = _HF_URL.match(ref)
    if match:
        return SourceRef("huggingface", match.group("repo"), match.group("rev"))
    if ref.startswith(("http://", "https://")):
        return SourceRef("url", ref)
    path = _resolve_path(ref, workspace)
    if path is not None:
        return SourceRef("directory" if path.is_dir() else "file", str(path))
    if _BARE_REPO.match(ref) and not ref.endswith((
        ".md",
        ".json",
        ".yaml",
        ".yml",
    )):
        return SourceRef("huggingface", ref)
    return None


def _resolve_path(ref: str, workspace: Path | None) -> Path | None:
    candidates = [Path(ref).expanduser()]
    if workspace is not None and not Path(ref).is_absolute():
        candidates.insert(0, Path(workspace) / ref)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return None


CARD_SUFFIXES = (".md", ".markdown", ".yaml", ".yml", ".json")


def is_card_source(source: SourceRef) -> bool:
    """Whether a source named in a request can hold a model card.

    A local file can only when it is Markdown, YAML, or JSON, the suffixes
    that plain path tokens and card URLs need. Before, a backticked name or a
    slashed path counted whenever the file existed, so "I have
    `molecules.smi`" read the SMILES data file as a model card. Hugging Face
    ids, URLs, and model directories pass.
    """
    return source.kind != "file" or source.value.lower().endswith(CARD_SUFFIXES)


def find_card_sources(text: str, workspace: Path | None = None) -> list[str]:
    """Find card references in free text.

    Bare ``org/name`` tokens count only inside backticks so phrases such as
    "input/output" are not mistaken for repositories. Data files named in the
    text are not card sources (see ``is_card_source``).
    """
    found: list[str] = []

    def add(ref: str) -> None:
        source = parse_source(ref, workspace)
        if (
            source is not None
            and is_card_source(source)
            and source.label not in found
        ):
            found.append(source.label)

    for match in _HF_IN_TEXT.finditer(text or ""):
        add(match.group(0))
    for match in _URL_IN_TEXT.finditer(text or ""):
        url = match.group(0).rstrip(".,;:")
        if (
            _HF_URL.match(url)
            or re.search(r"\.(?:md|markdown|ya?ml|json)(?:\?|$)", url)
            or _GITHUB_BLOB.match(url)
        ):
            add(url)
    for token in _BACKTICK_TOKEN.findall(text or ""):
        if (
            _BARE_REPO.match(token)
            or _resolve_path(token, workspace) is not None
        ):
            add(token)
    for match in _PATH_TOKEN.finditer(text or ""):
        if _resolve_path(match.group(1), workspace) is not None:
            add(match.group(1))
    for match in _SLASH_TOKEN.finditer(text or ""):
        token = match.group(1).rstrip("/")
        if _resolve_path(token, workspace) is not None:
            add(token)
    return found


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _http_client(timeout: float):
    from ursa import __version__
    from ursa.util.http import build_httpx_client

    return build_httpx_client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": f"ursa/{__version__} (model-cards)"},
    )


def _hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def fetch_hf_document(
    repo_id: str,
    revision: str | None = None,
    *,
    timeout: float = 30.0,
    client: Any = None,
) -> CardDocument:
    """Fetch README, config.json, and Hub metadata for one repository."""
    own = client is None
    client = client or _http_client(timeout)
    rev = revision or "main"
    base = hf_endpoint()
    headers = _hf_headers()
    files: list[str] = []
    access_error = None
    try:
        info = None
        response = client.get(f"{base}/api/models/{repo_id}", headers=headers)
        if response.status_code == 200:
            info = response.json()
        elif response.status_code in {401, 403}:
            access_error = f"Hugging Face returned {response.status_code} for {repo_id}; the repository is private or gated."
        elif response.status_code == 404:
            raise FileNotFoundError(
                f"Hugging Face repository '{repo_id}' was not found."
            )
        siblings = {
            s.get("rfilename")
            for s in (info or {}).get("siblings") or []
            if isinstance(s, dict)
        }
        text = ""
        response = client.get(
            f"{base}/{repo_id}/raw/{rev}/README.md", headers=headers
        )
        if (
            response.status_code == 200
            and len(response.content) <= MAX_CARD_BYTES
        ):
            text = response.text
            files.append("README.md")
        elif response.status_code in {401, 403}:
            access_error = (
                access_error
                or f"The model card for {repo_id} is gated; request access or set HF_TOKEN."
            )
        config = None
        if not siblings or "config.json" in siblings:
            response = client.get(
                f"{base}/{repo_id}/raw/{rev}/config.json", headers=headers
            )
            if (
                response.status_code == 200
                and len(response.content) <= MAX_CARD_BYTES
            ):
                try:
                    loaded = response.json()
                except ValueError:
                    loaded = None
                if isinstance(loaded, dict):
                    config = loaded
                    files.append("config.json")
    finally:
        if own:
            client.close()
    return CardDocument(
        source=CardSource(
            kind="huggingface",
            uri=f"hf://{repo_id}" + (f"@{revision}" if revision else ""),
            retrieved_at=_now(),
            files=files + (["hub_api"] if info else []),
        ),
        text=text,
        config=config,
        hub_info=info,
        repo_id=repo_id,
        access_error=access_error if not text else None,
    )


def search_hub(
    query: str, *, limit: int = 10, timeout: float = 30.0, client: Any = None
) -> list[dict[str, Any]]:
    """Search Hugging Face models by keyword, most downloaded first."""
    own = client is None
    client = client or _http_client(timeout)
    try:
        response = client.get(
            f"{hf_endpoint()}/api/models",
            params={
                "search": query,
                "sort": "downloads",
                "direction": -1,
                "limit": limit,
            },
            headers=_hf_headers(),
        )
        response.raise_for_status()
        data = response.json()
    finally:
        if own:
            client.close()
    return [item for item in data if isinstance(item, dict) and item.get("id")]


def _classify_mapping(data: dict[str, Any]) -> str:
    """Return 'native', 'config', or 'hub' for a JSON/YAML mapping."""
    if {"cardData", "siblings"} & set(data) or "modelId" in data:
        return "hub"
    native_keys = {
        "inputs",
        "outputs",
        "domains",
        "tasks",
        "interfaces",
        "schema_version",
    }
    if "name" in data and native_keys & set(data):
        return "native"
    return "config"


def _parse_structured(text: str, suffix: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def read_local_document(path: Path) -> CardDocument:
    """Read a card from a file or a model directory."""
    path = Path(path)
    files: list[str] = []
    text, config, native, hub = "", None, None, None
    if path.is_dir():
        targets = [
            path / name for name in CARD_FILENAMES if (path / name).is_file()
        ]
        if (path / "config.json").is_file():
            targets.append(path / "config.json")
    else:
        targets = [path]
    for target in targets:
        if target.stat().st_size > MAX_CARD_BYTES:
            logger.warning("Skipping oversized card file %s", target)
            continue
        content = target.read_text(encoding="utf-8", errors="replace")
        suffix = target.suffix.lower()
        if suffix in {".json", ".yaml", ".yml"}:
            data = _parse_structured(content, suffix)
            if data is None:
                continue
            kind = _classify_mapping(data)
            if kind == "native" and native is None:
                native = data
            elif kind == "hub" and hub is None:
                hub = data
            elif kind == "config" and config is None:
                config = data
            else:
                continue
        elif not text:
            text = content
        else:
            continue
        files.append(target.name)
    if not (text or config or native or hub):
        raise FileNotFoundError(f"No model card found at {path}")
    stem = path.name if path.is_dir() else path.stem
    if stem.lower() in {"readme", "model_card", "config"}:
        stem = path.parent.name if not path.is_dir() else stem
    return CardDocument(
        source=CardSource(
            kind="file", uri=str(path), retrieved_at=_now(), files=files
        ),
        text=text,
        config=config,
        native=native,
        hub_info=hub,
        name_hint=stem,
        id_hint=(native or {}).get("id") if native else None,
    )


def fetch_url_document(
    url: str, *, timeout: float = 30.0, client: Any = None
) -> CardDocument:
    """Fetch a Markdown, YAML, or JSON card from a URL."""
    match = _GITHUB_BLOB.match(url)
    fetch_url = (
        f"https://raw.githubusercontent.com/{match.group('owner')}/{match.group('repo')}/{match.group('rest')}"
        if match
        else url
    )
    own = client is None
    client = client or _http_client(timeout)
    try:
        response = client.get(fetch_url)
        response.raise_for_status()
        if len(response.content) > MAX_CARD_BYTES:
            raise ValueError(f"Card at {url} exceeds {MAX_CARD_BYTES} bytes")
        content = response.text
    finally:
        if own:
            client.close()
    path = fetch_url.split("?")[0]
    suffix = Path(path).suffix.lower()
    doc = CardDocument(
        source=CardSource(
            kind="url", uri=url, retrieved_at=_now(), files=[Path(path).name]
        ),
        name_hint=Path(path).stem
        if Path(path).stem.lower() not in {"readme", "model_card"}
        else Path(path).parent.name,
    )
    data = (
        _parse_structured(content, suffix)
        if suffix in {".json", ".yaml", ".yml"}
        else None
    )
    if data is not None:
        kind = _classify_mapping(data)
        if kind == "native":
            doc.native = data
        elif kind == "hub":
            doc.hub_info = data
        else:
            doc.config = data
    else:
        doc.text = content
    return doc


def load_document(
    ref: str | SourceRef,
    *,
    workspace: Path | None = None,
    timeout: float = 30.0,
    client: Any = None,
) -> CardDocument:
    """Load raw card materials for any supported source reference."""
    source = ref if isinstance(ref, SourceRef) else parse_source(ref, workspace)
    if source is None:
        raise ValueError(f"Not a recognizable model-card source: {ref!r}")
    if source.kind == "huggingface":
        return fetch_hf_document(
            source.value, source.revision, timeout=timeout, client=client
        )
    if source.kind == "url":
        return fetch_url_document(source.value, timeout=timeout, client=client)
    return read_local_document(Path(source.value))


def mcp_tool_document(
    *,
    name: str,
    server: str | None,
    description: str | None,
    input_schema: dict[str, Any] | None,
    card_text: str | None = None,
    output_schema: dict[str, Any] | None = None,
    nomad: bool = False,
) -> CardDocument:
    """Card materials for a model exposed as an MCP tool."""
    uri = f"mcp://{server or 'mcp'}/{name}"
    return CardDocument(
        source=CardSource(
            kind="nomad" if nomad else "mcp",
            uri=uri,
            retrieved_at=_now(),
            files=(["get_model_card"] if card_text else []) + ["tool_schema"],
        ),
        text=card_text or "",
        mcp_tool={
            "name": name,
            "server": server,
            "description": description or "",
            "input_schema": input_schema or {},
            "output_schema": output_schema or {},
        },
        id_hint=f"mcp:{server or 'mcp'}:{name}",
        name_hint=name,
    )
