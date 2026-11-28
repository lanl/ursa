"""Markdown structure helpers for model-card text.

These helpers are deliberately conservative. They recover the structure a
card author wrote (front matter, headings, key-value bullets, tables, code
blocks, links) and remove template boilerplate, but they do not interpret
scientific meaning. Interpretation happens in :mod:`ursa.model_cards.extract`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

_FRONT_MATTER = re.compile(
    r"\A﻿?\s*---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S
)
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_ATX = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT_H1 = re.compile(r"^\s{0,3}=+\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>")
_BADGE = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_BARE_URL = re.compile(r"(?<![(\[<])https?://[^\s)>\]\"'`]+")
_ARXIV = re.compile(
    r"(?:arxiv[:\s/]*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", re.I
)
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>(){}\[\],;`]+)", re.I)
_KV_BOLD = re.compile(
    r"^\s*(?:[-*+]\s+)?\*\*(?P<key>[^*\n]{1,80}?)\s*:?\s*\*\*\s*:?\s*"
    r"(?P<value>.*)$"
)
_KV_PLAIN = re.compile(
    r"^\s*[-*+]\s+(?P<key>[A-Za-z][\w /()&'.-]{0,60}?)\s*:\s+(?P<value>\S.*)$"
)
_CODE_SPAN = re.compile(r"(`+)(.+?)\1")
_DATE_KEY = re.compile(r"^\d{4}-\d{2}-\d{2}$|^[\d.,\s]+$")
_TABLE_SEPARATOR = re.compile(
    r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$"
)
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# Template variables look like {{ model_id }}; BibTeX titles such as
# {{Prithvi-EO-2.0: A Versatile ...}} must not match.
_PLACEHOLDER = re.compile(
    r"\{\{\s*[\w.]+\s*\}\}|\$\{[A-Z_]+\}|\{(?:spdx_license_id|license_name|"
    r"license_link|base_model)\}|YYYY-MM-DD",
    re.I,
)

# Instructional text from the Hugging Face and DOE Genesis card templates.
# Lines containing these markers are template residue, not facts.
TEMPLATE_MARKERS = (
    "[more information needed]",
    "more information needed",
    "this modelcard aims to be a base template",
    "this model card has been automatically generated",
    "provide a quick summary of what the model is",
    "use the code below to get started with the model",
    "the model description provides basic details about the model",
    "a person or group that was primarily responsible for the creation",
    "a person or group provided input or support to the model",
    "roberta-based binary classifier for stanford sentiment treebank",
    "roberta finetuned on snli",
    "this model is trained on roberta large",
    "blip-2 consists of 3 models",
    "the authors initialize the weights of the image encoder",
    "the goal for the model is to predict the next text token",
    "this allows the model to be used for tasks like image captioning",
    "the model was trained with 224 x 224 input images",
    "the input images are expected to have color values in the range [0,1]",
    "the output is a batch of feature vectors",
    "the output is the prediction score denoting",
    "example: this model was trained on a slurm cluster",
    "please include a link to the doe resource",
    "example: this model was trained on 4 nvidia",
    "example: the model was evaluated on 4 amd",
    "example: this model was trained with flashattention",
    "code snippets for getting configurations",
    "put output or link to output here",
    "if python: pip freeze > requirements.txt",
    "if spack: provide a spack lock file",
    "if conda: conda list --explicit",
    "if docker: include docker file",
    "else, another software package is used",
    "citations in [bibtex format]",
    "if using a non-standard license",
    "provide one or more corresponding authors with emails",
    "this section describes the use cases the model is intended for",
    "cases/examples/tasks for which the model was intended to be used",
    "example: the model will be used by researchers to understand",
    "this could include tasks linked to doe projects",
    "describe cases not recommended for this model",
    "this section is the most important for reusability of the model",
    "include code for training and inference and running the model",
    "the most powerful ai systems may pose novel national security risks",
    "such as cyberattacks and the development of chemical",
    "explosives (cbrne) weapons",
    "currently leads on ai capabilities",
    "preview for what foreign adversaries will possess",
    "nature of these risks as they emerge is vital",
    "from the ai action plan, please document risks",
    "any additional concerns, or tests/data needed",
    "other relevant cases not covered by the testing data",
    "like other large language models for which the homogeneity",
    "in this section you should describe all the relevant aspects of training",
    "describe all the relevant aspects of training that are useful",
    "description, includes where the data came from",
    "url(s) to dataset card(s) with relevant information",
    "url-link(s) to dataset card(s) with relevant information",
    "see also instructions for training data",
    "hyperparameter values, and whether hyperparameter tuning was performed",
    "random seed used: 42",
    "machine/environment info: gpu type, software versions",
    "link to evaluation or training pipeline: [url or repository path]",
    "this model was trained for 100k gradient steps",
    "the model uses normalfloat4 datatype and lora adapters",
    "provide an indication of how well the model performs",
    "relevant context amongst which the model was evaluated",
    "performance metrics, benchmarks tested, the baseline",
    "describe how  uncertainty/variability is calculated",
    "describe how uncertainty/variability is calculated",
    "evaluation quality metrics: e.g.",
    "evaluation runtime:  e.g.",
    "evaluation runtime: e.g.",
    "if using inference time compute or tools",
    "anything else you think it is important to communicate",
    "if your model is a fine-tune, an adapter, or a quantized version",
    "list of related/parent models (optional)",
    "roberta large, vit-huge-patch14-224-in21k",
    "(text, images, time series, etc.)",
    "examples of inputs and outputs:",
)
# Bare template bullets; matched against the whole normalized line so a real
# value such as "Batch size: 8" is kept.
TEMPLATE_LINES = frozenset({
    "example",
    "examples",
    "model initialization",
    "information about any fine-tuning, chain of thought, and n-shot "
    "learning performed",
    "optimizer information",
    "loss function",
    "stopping criterion",
    "number of training epochs",
    "number of training steps",
    "batch size",
    "training speed: number of steps per second and number of samples "
    "per second",
    "training loss: the average training loss and a specific training loss",
    "what prompting templates/functions were used? were prompts "
    "optimized? prompt examples",
    "model optimization techniques -- distillation/quantization/pruning/"
    "sparsity (learned during training or posthoc)",
})
GENERIC_TITLES = frozenset({
    "model card",
    "model overview",
    "overview",
    "model description",
    "description",
    "introduction",
    "citation",
    "model details",
    "summary",
    "about",
    "readme",
    "usage",
    "abstract",
    "model summary",
})


@dataclass
class Section:
    title: str
    level: int
    own_text: str
    text: str
    path: tuple[str, ...]
    line: int

    @property
    def key(self) -> str:
        return normalize_title(self.title)


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]
    section: str = ""


@dataclass
class CodeBlock:
    language: str
    code: str
    section: str = ""


@dataclass
class CardText:
    """Parsed structure of one model-card document."""

    raw: str
    front_matter: dict[str, Any] = field(default_factory=dict)
    front_matter_error: str | None = None
    body: str = ""
    clean_body: str = ""
    sections: list[Section] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)
    template_hits: int = 0


def split_front_matter(
    text: str,
) -> tuple[dict[str, Any], str, str | None]:
    """Split YAML front matter from a card. Returns (data, body, error)."""
    match = _FRONT_MATTER.match(text or "")
    if not match:
        return {}, text or "", None
    raw = match.group(1)
    body = text[match.end() :]
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        return {}, body, f"Invalid YAML front matter: {exc}".splitlines()[0]
    if not isinstance(data, dict):
        return {}, body, "YAML front matter is not a mapping"
    return data, body, None


def normalize_title(title: str) -> str:
    text = strip_inline_markdown(title).lower()
    text = re.sub(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+)", "", text)
    text = re.sub(r"[^\w\s/&+-]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" :-")


def strip_inline_markdown(text: str) -> str:
    """Remove inline markup while keeping code spans and identifiers."""
    spans: list[str] = []

    def protect(match: re.Match[str]) -> str:
        spans.append(match.group(2).strip())
        return f"\x00{len(spans) - 1}\x00"

    text = _BADGE.sub("", text or "")
    text = _CODE_SPAN.sub(protect, text)
    text = _MD_LINK.sub(r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|\*)(?=\S)(.+?)(?<=\S)\1", r"\2", text)
    # Underscores only mark emphasis at word boundaries (CommonMark), so
    # identifiers such as av_density survive.
    text = re.sub(
        r"(?<![A-Za-z0-9])(__|_)(?=\S)(.+?)(?<=\S)\1(?![A-Za-z0-9])",
        r"\2",
        text,
    )
    text = _HTML_TAG.sub(" ", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)
    text = text.replace("`", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def iter_lines(text: str):
    """Yield (line_no, line, in_code) triples, tracking fenced code."""
    fence: str | None = None
    for number, line in enumerate((text or "").splitlines()):
        match = _FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker[0] * 3
                yield number, line, True
                continue
            if marker.startswith(fence):
                fence = None
                yield number, line, True
                continue
        yield number, line, fence is not None


def evidence_text(text: str) -> str:
    """Card text without HTML comments and template residue.

    The LLM pass reads and quotes this text, so a value that appears only in
    a comment or in unfilled template text cannot enter the card.
    """
    return remove_template_residue(_HTML_COMMENT.sub("", text or ""))[0]


def remove_template_residue(text: str) -> tuple[str, int]:
    """Drop lines that come from card templates; code blocks are kept."""
    hits = 0
    kept = []
    for _, line, in_code in iter_lines(text):
        if in_code:
            kept.append(line)
            continue
        lowered = line.lower()
        bare = re.sub(r"^[\s>*+\-–]+", "", lowered)
        bare = re.sub(r"\s+", " ", bare.replace("*", "")).strip(" :")
        if (
            any(marker in lowered for marker in TEMPLATE_MARKERS)
            or _PLACEHOLDER.search(line)
            or bare in TEMPLATE_LINES
        ):
            hits += 1
            continue
        if "[optional]" in lowered:
            # Hugging Face template keys such as "**Paper [optional]:**" carry
            # real values; drop only the marker.
            hits += 1
            line = re.sub(r"\s*\[optional\]", "", line, flags=re.I)
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)), hits


def logical_lines(text: str) -> list[str]:
    """Join hard-wrapped prose lines; drop code blocks and tables.

    A new logical line starts at a list item, a bold key, a heading, a blank
    line, or a line that ends the previous sentence.
    """
    lines: list[str] = []
    for _, line, in_code in iter_lines(text):
        if in_code:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("|"):
            lines.append("")
            continue
        starts_new = (
            not lines
            or not lines[-1]
            or _LIST_ITEM.match(line)
            or _ATX.match(line)
            or stripped.startswith(("**", ">", "<"))
            or lines[-1].rstrip().endswith(("<br>", ":"))
            or _ATX.match(lines[-1])
        )
        if starts_new:
            lines.append(stripped)
        else:
            lines[-1] = f"{lines[-1]} {stripped}"
    return [line for line in lines if line]


def prose(text: str) -> str:
    """Plain prose of a section: no code, tables, or headings."""
    return "\n".join(
        strip_inline_markdown(_LIST_ITEM.sub("", line, count=1))
        for line in logical_lines(text)
        if not _ATX.match(line)
    )


def parse_sections(body: str) -> list[Section]:
    lines = body.splitlines()
    headings: list[tuple[int, int, str]] = []
    previous = ""
    for number, line, in_code in iter_lines(body):
        if in_code:
            previous = ""
            continue
        match = _ATX.match(line)
        if match:
            title = strip_inline_markdown(match.group(2))
            if title:
                headings.append((number, len(match.group(1)), title))
        elif _SETEXT_H1.match(line) and previous.strip():
            headings.append((number - 1, 1, strip_inline_markdown(previous)))
        previous = line
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    for index, (number, level, title) in enumerate(headings):
        next_any = (
            headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        )
        next_same = len(lines)
        for later_number, later_level, _ in headings[index + 1 :]:
            if later_level <= level:
                next_same = later_number
                break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        start = number + 1
        sections.append(
            Section(
                title=title,
                level=level,
                own_text="\n".join(lines[start:next_any]).strip(),
                text="\n".join(lines[start:next_same]).strip(),
                path=tuple(item[1] for item in stack),
                line=number,
            )
        )
    return sections


def parse_tables(text: str, section: str = "") -> list[Table]:
    tables: list[Table] = []
    lines = [line for _, line, in_code in iter_lines(text) if not in_code]
    index = 0
    while index < len(lines) - 1:
        line, separator = lines[index], lines[index + 1]
        if "|" in line and _TABLE_SEPARATOR.match(separator):
            headers = _split_row(line)
            rows = []
            index += 2
            while index < len(lines) and "|" in lines[index]:
                rows.append(_split_row(lines[index]))
                index += 1
            tables.append(Table(headers=headers, rows=rows, section=section))
            continue
        index += 1
    return tables


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [strip_inline_markdown(cell) for cell in stripped.split("|")]


def parse_code_blocks(text: str, section: str = "") -> list[CodeBlock]:
    blocks: list[CodeBlock] = []
    current: list[str] | None = None
    language = ""
    fence = ""
    for line in (text or "").splitlines():
        match = _FENCE.match(line)
        if current is None:
            if match:
                fence = match.group(1)[0] * 3
                language = line.strip()[len(match.group(1)) :].strip()
                language = language.split()[0].lower() if language else ""
                current = []
            continue
        if match and match.group(1).startswith(fence):
            blocks.append(CodeBlock(language, "\n".join(current), section))
            current = None
            continue
        current.append(line)
    return blocks


def parse_key_values(text: str) -> list[tuple[str, str]]:
    """Collect ``**Key:** value`` and ``- Key: value`` pairs.

    A bold key with an empty value takes the following list items or lines
    as its value, which matches the NVIDIA Model Card++ layout. A value that
    opens a parenthesis continues on following lines until it closes.
    """
    pairs: list[tuple[str, str]] = []
    lines = [line for _, line, in_code in iter_lines(text) if not in_code]
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _KV_BOLD.match(line) or _KV_PLAIN.match(line)
        if not match or _ATX.match(line):
            index += 1
            continue
        key = strip_inline_markdown(match.group("key")).strip(" :")
        raw_value = match.group("value")
        index += 1
        while (
            raw_value.count("(") > raw_value.count(")")
            and index < len(lines)
            and lines[index].strip()
            and not _KV_BOLD.match(lines[index])
        ):
            raw_value = f"{raw_value} {lines[index].strip()}"
            index += 1
        value = strip_inline_markdown(raw_value).strip()
        if not value and match.re is _KV_BOLD:
            collected = []
            while index < len(lines):
                follower = lines[index]
                if _ATX.match(follower) or _KV_BOLD.match(follower):
                    break
                if follower.strip():
                    collected.append(
                        strip_inline_markdown(follower.strip(" -*+"))
                    )
                elif collected:
                    break
                index += 1
            value = "; ".join(item for item in collected if item)
        if key and value and len(key.split()) <= 8 and not _DATE_KEY.match(key):
            pairs.append((key, value))
    return pairs


def parse_links(text: str) -> list[tuple[str | None, str]]:
    links: list[tuple[str | None, str]] = []
    seen: set[str] = set()
    for label, url in _MD_LINK.findall(text or ""):
        url = url.rstrip(".,;")
        if url not in seen:
            seen.add(url)
            links.append((strip_inline_markdown(label) or None, url))
    for url in _BARE_URL.findall(text or ""):
        url = url.rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            links.append((None, url))
    return links


def find_arxiv_ids(text: str) -> list[str]:
    return list(dict.fromkeys(_ARXIV.findall(text or "")))


def find_dois(text: str) -> list[str]:
    found: list[str] = []
    for doi in _DOI.findall(text or ""):
        doi = doi.rstrip(".}:")
        if doi not in found:
            found.append(doi)
    return found


def _balanced(text: str, start: int) -> int:
    """Index just past the brace group that opens at ``text[start]``."""
    depth, index = 0, start
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return len(text)


def _bibtex_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    index = 0
    name = re.compile(r"\s*,?\s*(\w+)\s*=\s*")
    while index < len(body):
        match = name.match(body, index)
        if not match:
            break
        key, index = match.group(1).lower(), match.end()
        if index < len(body) and body[index] == "{":
            end = _balanced(body, index)
            value = body[index + 1 : end - 1]
        elif index < len(body) and body[index] == '"':
            end = body.find('"', index + 1)
            end = len(body) if end < 0 else end + 1
            value = body[index + 1 : end - 1]
        else:
            token = re.match(r"[^,\s}]+", body[index:])
            value = token.group(0) if token else ""
            end = index + len(value)
        # Inner braces only protect capitalization in BibTeX.
        value = value.replace("{", "").replace("}", "")
        fields[key] = re.sub(r"\s+", " ", value).strip()
        index = end
    return fields


def parse_bibtex(code: str) -> list[dict[str, str]]:
    """Parse BibTeX entries into flat dictionaries of fields."""
    entries = []
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", code):
        open_brace = code.index("{", match.start())
        body = code[match.end() : _balanced(code, open_brace) - 1]
        entries.append({"key": match.group(1), **_bibtex_fields(body)})
    return entries


def parse_card_text(text: str) -> CardText:
    """Parse a card into front matter, sections, tables, and code."""
    front_matter, body, error = split_front_matter(text or "")
    cleaned = _HTML_COMMENT.sub("", body)
    without_template, hits = remove_template_residue(cleaned)
    if (
        "for reference on model card metadata, see the spec"
        in (text or "").lower()
    ):
        hits += 1
    card = CardText(
        raw=text or "",
        front_matter=front_matter,
        front_matter_error=error,
        body=body,
        clean_body=without_template,
        template_hits=hits,
    )
    card.sections = parse_sections(without_template)
    for section in card.sections:
        label = " > ".join(section.path)
        card.tables.extend(parse_tables(section.own_text, label))
        card.code_blocks.extend(parse_code_blocks(section.own_text, label))
    preamble = without_template
    if card.sections:
        first = card.sections[0].line
        preamble = "\n".join(without_template.splitlines()[:first])
    card.tables.extend(parse_tables(preamble))
    card.code_blocks.extend(parse_code_blocks(preamble))
    return card


def first_paragraph(text: str, *, min_words: int = 6) -> str | None:
    """First prose paragraph, skipping badges, images, lists, and tables."""
    for line in logical_lines(text):
        if _ATX.match(line) or _LIST_ITEM.match(line) or line.startswith(">"):
            continue
        plain = strip_inline_markdown(line)
        if len(plain.split()) >= min_words and not plain.startswith("@"):
            return plain
    return None
