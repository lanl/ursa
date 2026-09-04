"""MkDocs startup hooks and metadata-driven example pages."""

import os
import posixpath
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from mkdocs.structure.files import File, Files

from ursa.util.http import inject_truststore_into_ssl

# Inventory downloads happen while MkDocs plugins process their configuration,
# before any URSA command-line entry point can initialize TLS.
inject_truststore_into_ssl()


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
DOCS_ROOT = REPOSITORY_ROOT / "docs"
EXAMPLE_METADATA = "example.yaml"
TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"
TEMPLATES = Environment(
    loader=FileSystemLoader(TEMPLATES_ROOT),
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)
MARKDOWN_LINK = re.compile(r"(!?)\[([^]]+)]\(([^)]+)\)")


def _examples() -> list[tuple[Path, dict]]:
    """Return validated example folders and metadata in display order."""
    found: list[tuple[Path, dict]] = []
    for metadata_path in EXAMPLES_ROOT.rglob(EXAMPLE_METADATA):
        folder = metadata_path.parent
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"{metadata_path} must contain a YAML mapping")
        for required in ("title", "summary", "tags"):
            if not metadata.get(required):
                raise ValueError(f"{metadata_path} is missing '{required}'")
        if not isinstance(metadata["tags"], list) or not all(
            isinstance(tag, str) and tag for tag in metadata["tags"]
        ):
            raise ValueError(
                f"{metadata_path} 'tags' must be a list of strings"
            )
        for required_file in ("README.md", "pyproject.toml"):
            if not (folder / required_file).is_file():
                raise ValueError(f"{folder} is missing {required_file}")
        found.append((folder, metadata))
    return sorted(
        found,
        key=lambda item: (
            "source-only" in item[1]["tags"],
            str(item[1]["title"]).casefold(),
        ),
    )


def _example_slug(folder: Path) -> str:
    """Use the folder's repository-relative path as its stable URL slug."""
    return folder.relative_to(EXAMPLES_ROOT).as_posix()


def _github_ref() -> str:
    """Map Mike's docs version to this repository's source branch or tag."""
    version = os.environ.get("MIKE_DOCS_VERSION", "main")
    if version == "main":
        return version
    return version if version.startswith("v") else f"v{version}"


def _published_links(markdown: str, folder: Path) -> str:
    """Link docs to rendered pages and other files to versioned GitHub."""

    def replace(match: re.Match) -> str:
        marker, label, destination = match.groups()
        if destination.startswith("<") and ">" in destination:
            end = destination.index(">")
            target = destination[1:end]
            suffix = destination[end + 1 :]
        else:
            target, separator, remainder = destination.partition(" ")
            suffix = f"{separator}{remainder}" if separator else ""

        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)

        source = (folder / unquote(parsed.path)).resolve()
        if not source.exists() or not source.is_relative_to(REPOSITORY_ROOT):
            return match.group(0)

        if source.is_file() and source.is_relative_to(DOCS_ROOT):
            docs_path = source.relative_to(DOCS_ROOT).as_posix()
            page_path = PurePosixPath("examples", _example_slug(folder))
            published_url = posixpath.relpath(docs_path, page_path.as_posix())
            if parsed.query:
                published_url += f"?{parsed.query}"
            if parsed.fragment:
                published_url += f"#{parsed.fragment}"
            return f"{marker}[{label}]({published_url}{suffix})"

        repository_path = source.relative_to(REPOSITORY_ROOT).as_posix()
        if marker:
            github_url = (
                f"https://raw.githubusercontent.com/lanl/ursa/"
                f"{_github_ref()}/{quote(repository_path, safe='/')}"
            )
        else:
            kind = "blob" if source.is_file() else "tree"
            github_url = (
                f"https://github.com/lanl/ursa/{kind}/{_github_ref()}/"
                f"{quote(repository_path, safe='/')}"
            )
        if parsed.query:
            github_url += f"?{parsed.query}"
        if parsed.fragment:
            github_url += f"#{parsed.fragment}"
        return f"{marker}[{label}]({github_url}{suffix})"

    return MARKDOWN_LINK.sub(replace, markdown)


def _example_page(folder: Path, metadata: dict) -> str:
    """Render an example README with Material metadata and source link."""
    readme = (folder / "README.md").read_text(encoding="utf-8").rstrip()
    heading, separator, body = readme.partition("\n")
    if not separator or not heading.startswith("# "):
        raise ValueError(
            f"{folder / 'README.md'} must start with a level-one heading"
        )
    return TEMPLATES.get_template("example-page.md.jinja").render(
        title=heading.removeprefix("# "),
        body=_published_links(body.lstrip(), folder),
        tags=metadata["tags"],
        github_url=(
            f"https://github.com/lanl/ursa/tree/{_github_ref()}/examples/"
            f"{_example_slug(folder)}"
        ),
    )


def _examples_index(examples: list[tuple[Path, dict]]) -> str:
    """Insert the template-rendered card catalog into the root README."""
    readme = (EXAMPLES_ROOT / "README.md").read_text(encoding="utf-8").rstrip()
    marker = "<!-- example-catalog -->"
    if marker not in readme:
        raise ValueError(f"{EXAMPLES_ROOT / 'README.md'} is missing {marker}")

    cards = [
        {
            **metadata,
            "url": f"{_example_slug(folder)}/index.md",
        }
        for folder, metadata in examples
    ]
    all_tags = sorted(
        {tag for card in cards for tag in card["tags"]},
        key=str.casefold,
    )
    catalog = (
        TEMPLATES.get_template("example-catalog.md.jinja")
        .render(
            examples=cards,
            all_tags=all_tags,
        )
        .rstrip()
    )
    return readme.replace(marker, catalog) + "\n"


def on_files(files: Files, config) -> Files:
    """Expose example READMEs as virtual documentation pages."""
    examples = _examples()
    files.append(
        File.generated(
            config,
            "examples/index.md",
            content=_examples_index(examples),
        )
    )
    for folder, metadata in examples:
        slug = _example_slug(folder)
        files.append(
            File.generated(
                config,
                f"examples/{slug}/index.md",
                content=_example_page(folder, metadata),
            )
        )
    return files
