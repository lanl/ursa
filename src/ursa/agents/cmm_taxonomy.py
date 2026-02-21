from __future__ import annotations

import re
from typing import Iterable

COMMODITY_KEYWORDS: dict[str, set[str]] = {
    "HREE": {
        "dysprosium",
        "terbium",
        "heavy rare earth",
        "hree",
        "dy",
        "tb",
        "yttrium",
        "y",
    },
    "LREE": {
        "neodymium",
        "praseodymium",
        "lanthanum",
        "cerium",
        "light rare earth",
        "lree",
        "nd",
        "pr",
        "la",
        "ce",
        "bastnasite",
        "monazite",
    },
    "CO": {"cobalt", "co", "li-ion cathode", "nmc", "lco"},
    "LI": {
        "lithium",
        "li",
        "spodumene",
        "lithium carbonate",
        "li2co3",
        "lithium hydroxide",
        "brine",
    },
    "GA": {"gallium", "ga", "gaas", "gallium arsenide", "ga2o3"},
    "GR": {"graphite", "gr", "anode", "natural graphite", "synthetic graphite"},
    "NI": {"nickel", "ni", "class i nickel", "sulfide nickel", "laterite"},
    "CU": {"copper", "cu", "porphyry", "cathode copper"},
    "GE": {"germanium", "ge", "optical fiber", "infrared optics"},
    "OTH": {"critical mineral", "critical materials", "cmm", "strategic mineral"},
}

SUBDOMAIN_KEYWORDS: dict[str, set[str]] = {
    "T-EC": {
        "extraction",
        "leaching",
        "solvent extraction",
        "sx",
        "ion exchange",
        "separation",
        "acid digestion",
        "flotation",
        "beneficiation",
    },
    "T-PM": {
        "processing",
        "refining",
        "purification",
        "calcination",
        "roasting",
        "hydrometallurgy",
        "pyrometallurgy",
    },
    "T-GO": {
        "grade",
        "ore",
        "resource estimate",
        "reserve",
        "geometallurgy",
        "deposit",
    },
    "Q-PS": {
        "price",
        "spot price",
        "index",
        "market price",
        "premium",
        "discount",
    },
    "Q-TF": {
        "import",
        "export",
        "trade flow",
        "comtrade",
        "hs code",
        "tariff",
        "trade balance",
        "shipment",
    },
    "Q-EP": {
        "elasticity",
        "supply shock",
        "demand shock",
        "substitution",
        "cross-price",
    },
    "G-PR": {
        "policy",
        "regulation",
        "executive order",
        "ira",
        "inflation reduction act",
        "chips",
        "export control",
        "dpa",
        "sanction",
    },
    "G-BM": {
        "benchmark",
        "best practice",
        "governance",
        "standards",
        "traceability",
    },
    "S-CC": {
        "carbon",
        "co2",
        "emissions",
        "scope 1",
        "scope 2",
        "decarbonization",
    },
    "S-ST": {
        "sustainability",
        "esg",
        "water usage",
        "tailings",
        "waste",
        "environmental impact",
    },
}

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_QUARTER_RE = re.compile(r"\b((19|20)\d{2})[-\s]?Q([1-4])\b", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"\d")


def _normalize(text: str) -> str:
    return text.lower()


def _matches_keyword(text: str, keyword: str) -> bool:
    k = keyword.strip().lower()
    if not k:
        return False
    if " " in k or "-" in k:
        return k in text
    return re.search(rf"\b{re.escape(k)}\b", text) is not None


def detect_tags(
    text: str, taxonomy: dict[str, set[str]], fallback: Iterable[str] | None = None
) -> list[str]:
    normalized = _normalize(text)
    tags: list[str] = []
    for code, words in taxonomy.items():
        if any(_matches_keyword(normalized, word) for word in words):
            tags.append(code)
    if tags:
        return sorted(set(tags))
    if fallback:
        return sorted(set(fallback))
    return []


def detect_commodity_tags(
    text: str, fallback: Iterable[str] | None = None
) -> list[str]:
    return detect_tags(text, COMMODITY_KEYWORDS, fallback=fallback)


def detect_subdomain_tags(
    text: str, fallback: Iterable[str] | None = None
) -> list[str]:
    return detect_tags(text, SUBDOMAIN_KEYWORDS, fallback=fallback)


def extract_temporal_indicators(text: str) -> list[str]:
    years = {match.group(0) for match in _YEAR_RE.finditer(text)}
    quarters = {
        f"{match.group(1)}-Q{match.group(3)}" for match in _QUARTER_RE.finditer(text)
    }
    return sorted(quarters) + sorted(years)


def first_temporal_indicator(text: str) -> str:
    indicators = extract_temporal_indicators(text)
    return indicators[0] if indicators else ""


def has_numerical_data(text: str) -> bool:
    return _NUMERIC_RE.search(text) is not None
