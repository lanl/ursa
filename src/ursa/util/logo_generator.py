# ursa/util/logo_generator.py
from __future__ import annotations

import base64
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from openai import OpenAI  # pip install openai
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Reuse a small thread pool so callers can "fire-and-continue" with one line.
_EXEC = ThreadPoolExecutor(max_workers=2, thread_name_prefix="logo-gen")

# Optional: a default console if one isn’t passed in
_DEFAULT_CONSOLE = Console()

# Color for slug label per style (fallback used if missing)
_STYLE_COLORS = {
    "horror": "red",
    "sci-fi": "cyan",
    "cyberpunk": "magenta",
    "comic": "yellow",
    "fantasy": "green",
    "anime": "deep_sky_blue1",
    "mecha": "steel_blue",
    "steampunk": "dark_goldenrod",
    "ukiyoe": "light_sky_blue1",
    "surreal": "plum1",
    "noir": "grey70",
    "synthwave": "hot_pink",
    "mascot": "bright_magenta",
    "sticker": "bright_magenta",
    "random": "bright_yellow",
}

_WORKSPACE_COLOR = "bright_green"  # distinct from slug, border, and prompt
_BORDER_COLOR = "bright_cyan"  # distinct from slug & workspace

# ---------------------------
# Variety-driven prompt pools
# ---------------------------

META_TEMPLATE = (
    "Design a logo-ready symbol for '{workspace}'. Favor a strong silhouette and clear negative space. "
    "Use {render_mode} and a {composition} composition with a {palette_rule} palette. "
    "Nod subtly to {style} via {style_cues} (choose a couple, avoid clichés). "
    "Optional mood only: {problem_text}. {glyphs_rule} {wildcard} {system_hint}"
)

RENDER_MODES = [
    "flat vector shapes",
    "paper cutout",
    "linocut print",
    "stippled ink",
    "ceramic glaze texture",
    "folded origami",
    "wireframe mesh",
    "brushed metal",
    "neon tubing",
    "knitted yarn",
    "mosaic tiles",
    "laser-cut plywood",
    "light painting",
]

COMPOSITIONS = [
    "strict symmetry",
    "radial burst",
    "off-center tension",
    "stacked vertical",
    "nested negative space",
    "interlocking shapes",
    "spiral growth",
    "tilted diagonal",
]

PALETTE_RULES = [
    "monochrome",
    "duotone",
    "triadic accent",
    "black/white with a single shock color",
    "muted naturals",
    "warm metallics",
    "cool grayscale with neon accent",
]

STYLE_CUES = {
    "horror": [
        "elongated shadows",
        "organic asymmetry",
        "eroded edges",
        "subtle unease",
    ],
    "sci-fi": [
        "modular geometry",
        "specular highlights",
        "gridded logic",
        "soft glow",
    ],
    "cyberpunk": [
        "dense layering",
        "wet sheen",
        "electric micro-accents",
        "overlapping signage shapes",
    ],
    "comic": [
        "bold contour",
        "snap motion shapes",
        "halftone texture",
        "exaggerated foreshortening",
    ],
    "fantasy": [
        "ornamental motifs",
        "heroic scale cues",
        "mythic symmetry",
        "carved relief feel",
    ],
    "anime": [
        "silhouette clarity",
        "clean cel edges",
        "dramatic framing",
        "speed lines",
    ],
    "mecha": [
        "panel seams",
        "industrial joints",
        "mechanical symmetry",
        "maintenance patina",
    ],
    "steampunk": [
        "valve/gear hints",
        "brass/oxidized contrast",
        "pressure-gauge arcs",
        "Victorian filigree shapes",
    ],
    "ukiyoe": [
        "flat planes",
        "patterned waves/sky",
        "bold contour rhythm",
        "asymmetric balance",
    ],
    "surreal": [
        "scale paradox",
        "unexpected juxtapositions",
        "floating forms",
        "uncanny calm",
    ],
    "noir": [
        "hard light cuts",
        "oblique lattice angles",
        "rain sheen",
        "deep shadow masses",
    ],
    "synthwave": [
        "retro gradients",
        "sunset discs",
        "hazy horizon",
        "wire grid hint",
    ],
}

# Optional “don’t do this cliché this time” guardrails per style
CLICHE_BANS = {
    "horror": ["Do not use dripping blood."],
    "sci-fi": ["Avoid starfields and rocket silhouettes."],
    "cyberpunk": ["Skip city skylines and katakana signage."],
    "noir": ["No rain or venetian blinds this time."],
    "synthwave": ["Avoid palm trees and 80s grid sunsets."],
}

WILDCARDS = [
    "Introduce an unexpected but relevant metaphor or material.",
    "Consider how the mark could tessellate into a repeatable pattern.",
    "Hide a secondary icon in negative space.",
    "Let exactly one edge break the expected geometry.",
    "Constrain yourself to three primitive shapes total.",
    "Make the letterform (if any) only visible on second glance.",
]

SYSTEM_HINTS = [
    "Prefer forms that can be redrawn in ≤12 vector paths.",
    "Design for recognizability at 16×16 and at poster scale.",
    "Bias toward bold silhouette over surface detail.",
]


def _glyphs_rule(allow_text: bool) -> str:
    if allow_text:
        return "Letterforms are allowed; keep any words minimal and integrated into the symbol."
    # Softer than “No text” → allows abstract glyphs/monograms
    return "Avoid readable words; abstract glyphs/monograms are allowed if they strengthen the mark."


def _choose_style_slug(style: str | None) -> str:
    """
    Resolve a requested style to a known slug; if unknown or generic (e.g., 'sticker'),
    choose a random one from STYLE_CUES.
    """
    if not style:
        return random.choice(list(STYLE_CUES.keys()))
    s = style.strip().lower()
    if s in {"random"}:
        return random.choice(list(STYLE_CUES.keys()))
    return s if s in STYLE_CUES else random.choice(list(STYLE_CUES.keys()))


def _build_logo_prompt(
    *,
    style_slug: str,
    workspace: str,
    gist: str,
    allow_text: bool,
    palette: str | None,
    n_directions: int,
) -> str:
    render = random.choice(RENDER_MODES)
    comp = random.choice(COMPOSITIONS)
    palette_rule = palette if palette else random.choice(PALETTE_RULES)
    cues = ", ".join(random.sample(STYLE_CUES[style_slug], k=2))
    wildcard = random.choice(WILDCARDS)
    system_hint = random.choice(SYSTEM_HINTS)
    glyphs_rule = _glyphs_rule(allow_text)

    ban = ""
    if style_slug in CLICHE_BANS and random.random() < 0.6:
        ban = CLICHE_BANS[style_slug][0]  # single succinct ban

    meta = META_TEMPLATE.format(
        workspace=workspace,
        render_mode=render,
        composition=comp,
        palette_rule=palette_rule,
        style=style_slug,
        style_cues=cues,
        problem_text=gist,
        glyphs_rule=glyphs_rule,
        wildcard=wildcard,
        system_hint=system_hint,
    )

    # Encourage divergent outputs when requesting multiple images
    diversity_note = ""
    if n_directions and n_directions > 1:
        diversity_note = f" Produce {n_directions} distinct symbol directions that would never be confused with one another."

    return f"{meta} {ban} {diversity_note}".strip()


def _render_prompt_panel(
    *,
    console: Optional[Console],
    style_slug: str,
    workspace: str,
    prompt: str,
):
    c = console or _DEFAULT_CONSOLE
    slug_color = _STYLE_COLORS.get(style_slug, "bright_yellow")

    title = (
        f"[bold {slug_color}]style: {style_slug}[/bold {slug_color}] "
        f"[dim]•[/dim] "
        f"[bold {_WORKSPACE_COLOR}]workspace: {workspace}[/bold {_WORKSPACE_COLOR}]"
    )

    body = Text()
    body.append(prompt, style="bright_white")  # prompt text color

    panel = Panel.fit(
        body,
        title=title,
        border_style=_BORDER_COLOR,
        padding=(1, 2),
    )
    c.print(panel)


def _craft_logo_prompt(
    problem_text: str,
    workspace: str,
    *,
    style: str = "sticker",
    allow_text: bool = False,
    palette: str | None = None,
    n_directions: int = 1,
) -> tuple[str, str]:
    """
    Builds a less-prescriptive, variety-heavy prompt.
    Retains special handling for sticker/mascot.
    Returns (prompt, style_slug)
    """
    gist = " ".join(
        line.strip()
        for line in problem_text.strip().splitlines()
        if line.strip()
    )

    # Special path: sticker/mascot request
    if style in {"sticker", "mascot"}:
        prompt = (
            f"Create a die-cut sticker with a solid white background, a strong black border surrounding the white "
            f"die-cut border, and no shadow. The sticker image should be a 'mascot' related to the "
            f"topic: `{workspace}`."
        ).strip()
        return prompt, "sticker"

    # Resolve style and build meta-template prompt with randomized knobs
    style_slug = _choose_style_slug(style)
    prompt = _build_logo_prompt(
        style_slug=style_slug,
        workspace=workspace,
        gist=gist,
        allow_text=allow_text,
        palette=palette,
        n_directions=n_directions,
    )
    return prompt, style_slug


def _slugify(s: str) -> str:
    s = s.lower().strip().replace(" ", "-")
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-")


def _compose_filenames(
    out_dir: Path, style_slug: str, filename: str | None, n: int
):
    out_dir = Path(out_dir)
    if filename:
        stem = Path(filename).stem
        suffix = Path(filename).suffix or ".png"
        main = out_dir / f"{stem}{suffix}"
        alts = [out_dir / f"{stem}_{i}{suffix}" for i in range(2, n + 1)]
    else:
        suffix = ".png"
        base = f"{_slugify(style_slug)}_logo"
        main = out_dir / f"{base}{suffix}"
        alts = [out_dir / f"{base}_{i}{suffix}" for i in range(2, n + 1)]
    return main, alts


def generate_logo_sync(
    *,
    problem_text: str,
    workspace: str,
    out_dir: str | Path,
    filename: str | None = None,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    background: str = "opaque",
    quality: str = "high",
    n: int = 1,
    overwrite: bool = False,
    style: str = "sticker",
    allow_text: bool = False,
    palette: str | None = None,
    console: Optional[Console] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt, style_slug = _craft_logo_prompt(
        problem_text,
        workspace,
        style=style,
        allow_text=allow_text,
        palette=palette,
        n_directions=n,  # <-- tell the prompt how many distinct directions are requested
    )

    # Pretty console output
    _render_prompt_panel(
        console=console,
        style_slug=style_slug,
        workspace=workspace,
        prompt=prompt,
    )

    main_path, alt_paths = _compose_filenames(out_dir, style_slug, filename, n)
    if main_path.exists() and not overwrite:
        return main_path

    client = OpenAI()
    kwargs = dict(
        model=model,
        prompt=prompt,
        size=size,
        n=n,
        quality=quality,
        background=background,
    )
    try:
        resp = client.images.generate(**kwargs)
    except Exception:
        # Some models ignore/forbid background=; retry without it
        kwargs.pop("background", None)
        resp = client.images.generate(**kwargs)

    main_path.write_bytes(base64.b64decode(resp.data[0].b64_json))
    for i, item in enumerate(resp.data[1:], start=0):
        if i < len(alt_paths):
            alt_paths[i].write_bytes(base64.b64decode(item.b64_json))
    return main_path


def kickoff_logo(
    *,
    problem_text: str,
    workspace: str,
    out_dir: str | Path,
    filename: str | None = None,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    background: str = "opaque",
    quality: str = "high",
    n: int = 4,
    overwrite: bool = False,
    on_done=None,
    on_error=None,
    style: str = "sticker",
    allow_text: bool = False,
    palette: str | None = None,
    console: Optional[Console] = None,
):
    def _job() -> Path:
        return generate_logo_sync(
            problem_text=problem_text,
            workspace=workspace,
            out_dir=out_dir,
            filename=filename,
            model=model,
            size=size,
            background=background,
            quality=quality,
            n=n,
            overwrite=overwrite,
            style=style,
            allow_text=allow_text,
            palette=palette,
            console=console,
        )

    fut = _EXEC.submit(_job)
    if on_done or on_error:

        def _cb(f):
            try:
                p = f.result()
                on_done and on_done(p)
            except BaseException as e:
                on_error and on_error(e)

        fut.add_done_callback(_cb)
    return fut
