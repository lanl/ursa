"""Persistent catalog of normalized model cards.

Each card is stored as one JSON file, so catalogs are easy to inspect, diff,
share, and edit by hand. By default the catalog lives in the URSA group
cache (``~/.cache/ursa/<group>/model_cards``) and is shared by every agent in
that group.
"""

from __future__ import annotations

import difflib
import logging
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from ursa.model_cards.schema import ModelCard
from ursa.model_cards.taxonomy import DOMAINS, TASKS

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its model models of on "
    "or that the this to with we our use using can will into per via".split()
)


def default_catalog_dir(group: str | None = None) -> Path:
    from ursa.security import group_root_dir

    return group_root_dir(group or "default") / "model_cards"


def card_slug(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", model_id).strip("_.")
    return slug.lower()[:180] or "card"


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall((text or "").lower().replace("_", " ")):
        if token not in _STOPWORDS and len(token) > 1:
            tokens.append(token)
    return tokens


def card_text(card: ModelCard) -> str:
    """Searchable text: names, science, I/O, datasets, and uses."""
    parts = [card.name, *card.aliases, card.summary or "", *card.keywords]
    for concept_id in card.domains:
        concept = DOMAINS.get(concept_id)
        parts.append(concept.label if concept else concept_id)
    for concept_id in card.tasks:
        concept = TASKS.get(concept_id)
        parts.append(concept.label if concept else concept_id)
    for spec in [*card.inputs, *card.outputs]:
        parts += [spec.name, spec.description or "", spec.representation or ""]
        parts += [channel.name for channel in spec.channels]
    parts += [dataset.name for dataset in card.training_data]
    parts += card.intended_uses
    parts.append(card.architecture.family or "")
    parts.append(card.architecture.name or "")
    return " ".join(part for part in parts if part)


class ModelCatalog:
    """A directory of normalized model cards."""

    def __init__(self, directory: Path | str | None = None):
        self.directory = Path(directory).expanduser() if directory else None
        self._cards: dict[str, ModelCard] = {}
        # Card text, kept so answers can quote the exact line a fact came from.
        self._texts: dict[str, str] = {}
        if self.directory is not None and self.directory.is_dir():
            for path in sorted(self.directory.glob("*.json")):
                try:
                    card = ModelCard.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                except (ValidationError, ValueError, OSError) as exc:
                    logger.warning(
                        "Skipping unreadable model card %s: %s", path, exc
                    )
                    continue
                self._cards[card.id] = card

    @classmethod
    def for_group(cls, group: str | None = None) -> ModelCatalog:
        return cls(default_catalog_dir(group))

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[ModelCard]:
        return iter(self.all())

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._cards

    def all(self) -> list[ModelCard]:
        return sorted(self._cards.values(), key=lambda card: card.name.lower())

    def add(
        self,
        card: ModelCard,
        *,
        persist: bool = True,
        text: str | None = None,
    ) -> ModelCard:
        self._cards[card.id] = card
        if text is not None:
            self._texts[card.id] = text
        if persist and self.directory is not None:
            self._write(card)
            if text is not None:
                self._write_text(card.id, text)
        return card

    def text(self, ref: str | None) -> str | None:
        """The stored card text of a model, or None when none was kept."""
        card = self.get(ref)
        if card is None:
            return None
        if card.id not in self._texts and self.directory is not None:
            path = self.directory / f"{card_slug(card.id)}.md"
            if path.is_file():
                self._texts[card.id] = path.read_text(encoding="utf-8")
        return self._texts.get(card.id)

    def add_all(
        self, cards: Iterable[ModelCard], *, persist: bool = True
    ) -> None:
        for card in cards:
            self.add(card, persist=persist)

    def _write(self, card: ModelCard) -> None:
        self._atomic_write(
            f"{card_slug(card.id)}.json",
            card.model_dump_json(indent=2, exclude_none=True),
        )

    def _write_text(self, model_id: str, text: str) -> None:
        self._atomic_write(f"{card_slug(model_id)}.md", text)

    def _atomic_write(self, name: str, content: str) -> None:
        assert self.directory is not None
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(tmp, self.directory / name)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def remove(self, ref: str) -> bool:
        card = self.get(ref)
        if card is None:
            return False
        del self._cards[card.id]
        self._texts.pop(card.id, None)
        if self.directory is not None:
            for suffix in (".json", ".md"):
                (self.directory / f"{card_slug(card.id)}{suffix}").unlink(
                    missing_ok=True
                )
        return True

    def get(self, ref: str | None) -> ModelCard | None:
        """Resolve an id, name, alias, or repository id; fuzzy as last resort."""
        if not ref:
            return None
        ref = ref.strip()
        if ref in self._cards:
            return self._cards[ref]
        lowered = ref.lower().removeprefix("hf://").removeprefix("hf:")
        for card in self._cards.values():
            names = {
                card.id.lower(),
                card.name.lower(),
                *(a.lower() for a in card.aliases),
            }
            if card.id.startswith("hf:"):
                names.add(card.id[3:].lower())
            if card.source.uri:
                names.add(card.source.uri.lower().removeprefix("hf://"))
            if lowered in names:
                return card
        keys = {}
        for card in self._cards.values():
            for name in (card.name, *card.aliases):
                keys[name.lower()] = card
        close = difflib.get_close_matches(lowered, list(keys), n=1, cutoff=0.85)
        return keys[close[0]] if close else None

    def find_source(
        self, uri: str, sha256: str | None = None
    ) -> ModelCard | None:
        for card in self._cards.values():
            if card.source.uri == uri and (
                sha256 is None or card.source.sha256 == sha256
            ):
                return card
        return None

    def search(self, query: str, k: int = 10) -> list[tuple[ModelCard, float]]:
        """Rank cards by TF-IDF cosine similarity to the query."""
        scores = lexical_scores(query, self.all())
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        return [
            (self._cards[model_id], score)
            for model_id, score in ranked[:k]
            if score > 0
        ]


def lexical_scores(query: str, cards: list[ModelCard]) -> dict[str, float]:
    """TF-IDF cosine similarity between a query and each card's text."""
    documents = {card.id: Counter(tokenize(card_text(card))) for card in cards}
    query_terms = Counter(tokenize(query))
    if not documents or not query_terms:
        return {card.id: 0.0 for card in cards}
    n_docs = len(documents)
    document_frequency: Counter[str] = Counter()
    for terms in documents.values():
        document_frequency.update(set(terms))

    def idf(term: str) -> float:
        return (
            math.log((1 + n_docs) / (1 + document_frequency.get(term, 0))) + 1.0
        )

    def weights(terms: Counter[str]) -> dict[str, float]:
        return {
            term: (1 + math.log(count)) * idf(term)
            for term, count in terms.items()
        }

    q = weights(query_terms)
    q_norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
    scores = {}
    for model_id, terms in documents.items():
        d = weights(terms)
        d_norm = math.sqrt(sum(v * v for v in d.values())) or 1.0
        dot = sum(q[t] * d.get(t, 0.0) for t in q)
        scores[model_id] = dot / (q_norm * d_norm)
    return scores
