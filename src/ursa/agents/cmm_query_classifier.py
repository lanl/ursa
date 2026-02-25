from __future__ import annotations

import re
from dataclasses import dataclass

from ursa.agents.cmm_taxonomy import (
    detect_commodity_tags,
    detect_subdomain_tags,
    extract_temporal_indicators,
)

_FACTOID_PREFIXES = {
    "what",
    "which",
    "who",
    "when",
    "where",
    "how",
    "is",
    "are",
    "does",
    "do",
    "can",
}
_COMPARE_MARKERS = {"compare", "versus", "vs", "difference between", "relative to"}
_CAUSAL_MARKERS = {"affect", "impact", "consequence", "lead to", "because"}


@dataclass
class QueryProfile:
    query_type: str
    commodity_hints: list[str]
    subdomain_hints: list[str]
    temporal_hints: list[str]
    retrieval_k: int
    return_k: int
    alpha: float
    filters: dict


class CMMQueryClassifier:
    def classify(self, query: str) -> QueryProfile:
        q = query.strip()
        lower_q = q.lower()
        tokens = q.split()

        commodity_hints = detect_commodity_tags(q)
        subdomain_hints = detect_subdomain_tags(q)
        temporal_hints = extract_temporal_indicators(q)

        is_factoid = (
            len(tokens) < 15
            and bool(tokens)
            and tokens[0].lower().strip("?:!,.") in _FACTOID_PREFIXES
        )
        is_comparative = any(marker in lower_q for marker in _COMPARE_MARKERS)
        is_multi_hop = (
            any(marker in lower_q for marker in _CAUSAL_MARKERS)
            or len(subdomain_hints) >= 2
        )
        is_temporal = bool(
            temporal_hints
            or re.search(r"\b(recent|current|trend|historical)\b", lower_q)
        )

        query_type = "general"
        retrieval_k = 20
        return_k = 5
        alpha = 0.7

        if is_comparative:
            query_type = "comparative"
            retrieval_k = 30
            return_k = 8
            alpha = 0.7
        elif is_multi_hop:
            query_type = "multi_hop"
            retrieval_k = 30
            return_k = 10
            alpha = 0.8
        elif is_temporal:
            query_type = "temporal"
            retrieval_k = 24
            return_k = 6
            alpha = 0.8
        elif is_factoid:
            query_type = "factoid"
            retrieval_k = 10
            return_k = 3
            alpha = 0.5

        filters: dict = {}
        if commodity_hints:
            filters["commodity_tags"] = commodity_hints
        if subdomain_hints:
            filters["subdomain_tags"] = subdomain_hints
        if temporal_hints:
            years = sorted(
                {hint[:4] if "-Q" in hint else hint for hint in temporal_hints}
            )
            filters["temporal_indicator_gte"] = years[0]
            filters["temporal_indicator_lte"] = years[-1]

        return QueryProfile(
            query_type=query_type,
            commodity_hints=commodity_hints,
            subdomain_hints=subdomain_hints,
            temporal_hints=temporal_hints,
            retrieval_k=retrieval_k,
            return_k=return_k,
            alpha=alpha,
            filters=filters,
        )
