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

# ---------------------------
# Top-level tuning knobs
# ---------------------------

# If you want "web UI vibes" (more novelty), increase APERTURE (0..1).
# Higher APERTURE => fewer hard constraints, more "art-direction" phrasing.
DEFAULT_APERTURE = 0.75

# Default image model (can still be overridden by callers passing model= / image_model=)
DEFAULT_IMAGE_MODEL = "gpt-image-1.5"

# When generating scenes with style="random" and n>1, pick distinct styles.
SCENE_MULTI_STYLE_DEFAULT = True

# ---------------------------
# Console colors
# ---------------------------

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
# Variety-driven prompt pools (logos)
# ---------------------------

# "Strict" logo framing (kept, but we now apply a constraint budget)
STRICT_LOGO_TEMPLATE = (
    "Design a logo-ready symbol for '{workspace}'. "
    "{constraints} "
    "{render_bits} "
    "Nod to {style} via {style_cues}. "
    "Optional mood only: {problem_text}. {glyphs_rule} {wildcard} {system_hint}"
)

# "Loose" logo framing (opens the aperture)
LOOSE_LOGO_TEMPLATE = (
    "Create an original emblem for '{workspace}'. "
    "Pick an unusual but relevant metaphor and an unexpected material or medium. "
    "{constraints} "
    "{render_bits} "
    "Let {style} be felt through: {style_cues}. "
    "Mood (optional): {problem_text}. {glyphs_rule} {wildcard}"
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

# A small set of constraints; we pick a subset each time (constraint budget).
LOGO_CONSTRAINTS = [
    "Favor a strong silhouette.",
    "Use clear negative space.",
    "Keep the shape simple enough to redraw from memory.",
    "Avoid overly intricate linework; prioritize clear geometry.",
    "Design for recognizability at 16×16 and at poster scale.",
    "Prefer forms that can be redrawn in ≤12 vector paths.",
]

CLICHE_AVOIDANCE = "Avoid common logo tropes: shields, hexagons, circuits, brains, infinity loops, generic rockets."

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

# ---------------------------
# Scene-first prompt pools (scenes)
# ---------------------------

# Concrete, unmistakable set-pieces per style (expanded to reduce samey scenes).
STYLE_OBJECTS = {
    "horror": [
        "flickering corridor lights and long shadows",
        "peeling paint, damp walls, and subtle mold textures",
        "old warning placards with torn edges",
        "foggy air catching thin light beams",
        "an unsettling, half-seen shape at the edge of frame",
    ],
    "fantasy": [
        "carved stone arches with runic inlays",
        "glowing moss and bioluminescent flora",
        "ancient tapestries and ornate metalwork",
        "crystal lanterns casting colored light",
        "a relic or sigil with mythic significance",
    ],
    "sci-fi": [
        "clean modular panels with soft emissive seams",
        "holographic UI elements floating in space",
        "transparent conduits and fiber-optic glows",
        "precision machinery with minimal fasteners",
        "a distant megastructure or orbital ring",
    ],
    "cyberpunk": [
        "neon signage reflected in wet pavement",
        "layered cables, vents, and grime-streaked glass",
        "dense street clutter—stalls, tarps, and ductwork",
        "electric accent lighting and chromatic aberration vibes",
        "a glowing billboard haze in the background",
    ],
    "comic": [
        "bold ink contours and graphic shadow shapes",
        "halftone textures and punchy visual icons",
        "exaggerated perspective lines and motion cues",
        "speech-bubble-like signage shapes (no readable text)",
        "stylized impact bursts as compositional accents",
    ],
    "anime": [
        "clean cel-style edges with soft gradient shading",
        "dramatic rim lighting and cinematic framing",
        "iconic props with crisp silhouettes",
        "speed-line energy implied through composition",
        "a bright sky or night city backdrop with simple shapes",
    ],
    "mecha": [
        "panel seams and industrial joints",
        "maintenance markings and warning stripes (no text)",
        "hydraulic pistons and articulated plating",
        "coolant vapor and heat shimmer",
        "a hangar bay or test range environment",
    ],
    "steampunk": [
        "polished brass and riveted steel",
        "visible interlocking gears with sharp teeth",
        "pressure gauges with needles and glass reflections",
        "pistons and exposed pipework venting steam",
        "Victorian filigree panels and leather straps",
    ],
    "ukiyoe": [
        "flat planes of color with woodblock grain",
        "patterned waves, clouds, or wind lines",
        "bold contour rhythm and asymmetrical balance",
        "paper texture and restrained gradients",
        "stylized nature motifs as framing elements",
    ],
    "surreal": [
        "scale paradox—tiny objects looming large",
        "floating forms casting impossible shadows",
        "a calm scene with one uncanny impossibility",
        "unexpected material swaps (stone like fabric, etc.)",
        "a horizon that bends subtly out of reality",
    ],
    "noir": [
        "hard-edged light cuts and deep shadow masses",
        "rain sheen on streets or window glass",
        "venetian-blind slashes of light",
        "cigarette-smoke haze and high contrast",
        "silhouetted figures and oblique angles",
    ],
    "synthwave": [
        "retro sunset discs and hazy horizons",
        "neon magenta/cyan accents with deep blacks",
        "wire-grid ground plane receding to the horizon",
        "glowing edge lighting and soft bloom",
        "geometric mountains or abstract city silhouettes",
    ],
}

CAMERAS = [
    "low-angle hero shot",
    "wide-angle 24mm",
    "telephoto compression",
    "over-the-shoulder cinematic framing",
    "bird's-eye parallax",
    "macro close-up with shallow depth of field",
    "isometric cutaway",
]

COMPOSITIONS_SCENE = [
    "rule-of-thirds focal point with leading lines",
    "layered foreground/midground/background for depth",
    "silhouette against a bright atmospheric backdrop",
    "quiet negative space with a small, crisp focal subject",
    "dynamic diagonal energy with one strong vanishing point",
    "centered symmetry broken by a single asymmetrical detail",
    "tight close-up detail that implies a larger world",
]

PALETTES_SCENE = [
    "cold moonlight blues with warm tungsten highlights",
    "neon magenta and cyan with deep blacks",
    "misty greens and desaturated grays",
    "sunset orange with violet shadows",
    "ink black with ivory highlights and one accent color",
    "dusty pastels with soft film grain",
    "muted earth tones with a single vivid glow",
]

SCENE_WILDCARDS = [
    "Use volumetric light rays and haze for depth.",
    "Introduce one surprising prop that hints at the project name (no readable text).",
    "Add subtle surface wear—scratches, fingerprints, dust—where appropriate.",
    "Include a repeating motif in the environment that echoes the subject’s shape language.",
    "Add one gentle surreal element that still feels coherent.",
    "Let a single color accent guide the viewer’s eye to the focal point.",
]

SCENE_TEMPLATE = (
    "Create a full-frame illustration (not a logo) for '{workspace}'. "
    "Use a {composition} composition and {camera} camera angle with cinematic lighting and atmospheric depth. "
    "{style_strength} {style} unmistakable at first glance through: {style_objects}. "
    "Include a coherent background environment; avoid borders or sticker outlines. "
    "Palette: {palette_rule}. Mood (optional): {problem_text}. {wildcard}"
)


STICKER_ART_STYLES = [
    "cute chibi mascot",
    "bold 90s skate-sticker style",
    "minimal flat-vector mascot",
    "inked comic mascot with clean fills",
    "retro patch-style embroidery look (still sticker)",
]

STICKER_POSES = [
    "hero pose",
    "mid-action leap",
    "waving / friendly greeting",
    "leaning forward with curiosity",
    "arms crossed, confident stance",
]

STICKER_PROPS = [
    "holding a tiny tool",
    "wearing goggles",
    "with a small companion critter",
    "with a floating icon-shaped charm",
    "with a subtle geometric motif behind them",
]


# ---------------------------
# Helpers
# ---------------------------


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return DEFAULT_APERTURE


def _glyphs_rule(allow_text: bool) -> str:
    if allow_text:
        return "Letterforms are allowed; keep any words minimal and integrated into the symbol."
    # Softer than “No text” → allows abstract glyphs/monograms
    return "Avoid readable words; abstract glyphs/monograms are allowed if they strengthen the mark."


def _choose_style_slug(style: str | None) -> str:
    """
    Resolve a requested style to a known slug; if unknown or generic, choose a random one.
    """
    if not style:
        return random.choice(list(STYLE_CUES.keys()))
    s = style.strip().lower()
    if s in {"random"}:
        return random.choice(list(STYLE_CUES.keys()))
    return s if s in STYLE_CUES else random.choice(list(STYLE_CUES.keys()))


def _choose_n_distinct_styles(n: int) -> list[str]:
    pool = list(STYLE_CUES.keys())
    if n <= 1:
        return [random.choice(pool)]
    # If n > len(pool), allow repeats (but we’ll try to distinct first)
    if n <= len(pool):
        return random.sample(pool, k=n)
    out = random.sample(pool, k=len(pool))
    while len(out) < n:
        out.append(random.choice(pool))
    return out


def _style_strength_phrase(level: str) -> str:
    return {
        "subtle": "Include gentle references to",
        "clear": "Make the influence of",
        "overt": "Make the aesthetic of",
    }.get(level, "Make the aesthetic of")


SUPPORTED_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}


def _size_for(aspect: str, mode: str) -> str:
    """
    Prefer cinematic rectangles for scenes; keep square for logos.
    Returns only API-supported sizes.
    """
    if aspect == "wide":
        return "1536x1024"
    if aspect == "tall":
        return "1024x1536"
    return "1024x1024"


def _normalize_size(size: Optional[str], aspect: str, mode: str) -> str:
    """
    If size is None or invalid, pick a sensible API-supported default based on aspect/mode.
    """
    if not size or size not in SUPPORTED_SIZES:
        return _size_for(aspect, mode)
    return size


def _pick_logo_constraints(aperture: float) -> str:
    """
    "Constraint budget":
      - open aperture => fewer constraints
      - closed aperture => more constraints
    """
    a = _clamp01(aperture)
    # 0.0 -> 5 constraints, 1.0 -> 2 constraints
    k = int(round(5 - 3 * a))
    k = max(2, min(5, k))
    chosen = random.sample(LOGO_CONSTRAINTS, k=k)
    # Occasionally include cliché avoidance when aperture is open (pushes novelty)
    if random.random() < (0.35 + 0.35 * a):
        chosen.append(CLICHE_AVOIDANCE)
    return " ".join(chosen).strip()


# ---------------------------
# Prompt builders
# ---------------------------


def _build_logo_prompt(
    *,
    style_slug: str,
    workspace: str,
    gist: str,
    allow_text: bool,
    palette: str | None,
    aperture: float,
) -> str:
    a = _clamp01(aperture)

    render = random.choice(RENDER_MODES)
    comp = random.choice(COMPOSITIONS)
    palette_rule = palette if palette else random.choice(PALETTE_RULES)
    cues = ", ".join(random.sample(STYLE_CUES[style_slug], k=2))
    wildcard = random.choice(WILDCARDS)
    system_hint = random.choice(SYSTEM_HINTS)
    glyphs_rule = _glyphs_rule(allow_text)

    # Open aperture => sometimes omit some rigid “bits”
    bits = []
    if random.random() < (0.85 - 0.35 * a):
        bits.append(f"Use {render}")
    if random.random() < (0.80 - 0.30 * a):
        bits.append(f"a {comp} composition")
    if random.random() < (0.90 - 0.25 * a):
        bits.append(f"with a {palette_rule} palette")
    render_bits = ""
    if bits:
        # Make a nice sentence fragment
        if len(bits) == 1:
            render_bits = f"{bits[0]}."
        elif len(bits) == 2:
            render_bits = f"{bits[0]} and {bits[1]}."
        else:
            render_bits = f"{', '.join(bits[:-1])}, and {bits[-1]}."

    constraints = _pick_logo_constraints(a)
    template = (
        LOOSE_LOGO_TEMPLATE if random.random() < a else STRICT_LOGO_TEMPLATE
    )

    return template.format(
        workspace=workspace,
        constraints=constraints,
        render_bits=render_bits,
        style=style_slug,
        style_cues=cues,
        problem_text=gist,
        glyphs_rule=glyphs_rule,
        wildcard=wildcard,
        system_hint=system_hint,
    ).strip()


def _build_scene_prompt(
    *,
    style_slug: str,
    workspace: str,
    gist: str,
    palette: Optional[str],
    style_intensity: str = "overt",
    aperture: float = DEFAULT_APERTURE,
) -> str:
    a = _clamp01(aperture)

    comp = random.choice(COMPOSITIONS_SCENE)
    camera = random.choice(CAMERAS)

    # Open aperture => slightly broader palette choices (still guided)
    palette_rule = palette or random.choice(PALETTES_SCENE)

    style_strength = _style_strength_phrase(style_intensity)

    objects_pool = (
        STYLE_OBJECTS.get(style_slug)
        or STYLE_CUES.get(style_slug)
        or ["signature motifs"]
    )

    # Open aperture => fewer “must-include” objects (less checklist-y),
    # Closed aperture => more objects for unmistakable style.
    k_max = min(4, len(objects_pool))
    k_min = 2 if k_max >= 2 else 1
    k = k_min if a >= 0.7 else k_max
    k = max(1, min(k_max, k))
    style_objects = ", ".join(random.sample(objects_pool, k=k))

    wildcard = random.choice(SCENE_WILDCARDS)

    return SCENE_TEMPLATE.format(
        workspace=workspace,
        composition=comp,
        camera=camera,
        style_strength=style_strength,
        style=style_slug,
        style_objects=style_objects,
        palette_rule=palette_rule,
        problem_text=gist,
        wildcard=wildcard,
    ).strip()


def _render_prompt_panel(
    *,
    console: Optional[Console],
    style_slug: str,
    workspace: str,
    prompt: str,
    extra_title: str | None = None,
):
    c = console or _DEFAULT_CONSOLE
    slug_color = _STYLE_COLORS.get(style_slug, "bright_yellow")

    title_bits = [
        f"[bold {slug_color}]style: {style_slug}[/bold {slug_color}]",
        f"[dim]•[/dim] [bold {_WORKSPACE_COLOR}]workspace: {workspace}[/bold {_WORKSPACE_COLOR}]",
    ]
    if extra_title:
        title_bits.append(f"[dim]•[/dim] {extra_title}")

    title = " ".join(title_bits)

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
    mode: str = "logo",  # "logo" | "scene"
    style_intensity: str = "overt",  # "subtle" | "clear" | "overt"
    aperture: float = DEFAULT_APERTURE,
) -> tuple[str, str]:
    """
    Builds either a logo-style prompt (legacy) or a scene-style prompt (new).
    Retains special handling for sticker/mascot.
    Returns (prompt, style_slug)
    """
    gist = " ".join(
        line.strip()
        for line in problem_text.strip().splitlines()
        if line.strip()
    )

    # Special path: sticker/mascot request (unchanged)
    if style in {"sticker", "mascot"}:
        art = random.choice(STICKER_ART_STYLES)
        pose = random.choice(STICKER_POSES)
        prop = random.choice(STICKER_PROPS)
        prompt = (
            "Create a die-cut sticker with a solid white background, a strong black border surrounding the white "
            "die-cut border, and no shadow. "
            f"The sticker image should be a {art} related to the topic: `{workspace}`. "
            f"Pose: {pose}. Detail: {prop}. "
            "Avoid readable words; simple emblematic shapes only."
        ).strip()
        return prompt, "sticker"

    style_slug = _choose_style_slug(style)

    if mode == "scene":
        prompt = _build_scene_prompt(
            style_slug=style_slug,
            workspace=workspace,
            gist=gist,
            palette=palette,
            style_intensity=style_intensity,
            aperture=aperture,
        )
        return prompt, style_slug

    prompt = _build_logo_prompt(
        style_slug=style_slug,
        workspace=workspace,
        gist=gist,
        allow_text=allow_text,
        palette=palette,
        aperture=aperture,
    )
    return prompt, style_slug


def _slugify(s: str) -> str:
    s = s.lower().strip().replace(" ", "-")
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-")


def _compose_filenames(
    out_dir: Path, style_slug: str, filename: str | None, n: int, *, mode: str
):
    """
    For logos/stickers (single-style), keep backward-ish naming.
    For scenes, caller may use multi-style naming elsewhere.
    """
    out_dir = Path(out_dir)
    suffix = ".png"

    if filename:
        stem = Path(filename).stem
        suffix = Path(filename).suffix or suffix
        main = out_dir / f"{stem}{suffix}"
        alts = [out_dir / f"{stem}_{i}{suffix}" for i in range(2, n + 1)]
        return main, alts

    # default: style_slug_mode_{i}.png
    base = f"{_slugify(style_slug)}_{mode}"
    main = out_dir / f"{base}{suffix}"
    alts = [out_dir / f"{base}_{i}{suffix}" for i in range(2, n + 1)]
    return main, alts


def _compose_multi_scene_paths(
    out_dir: Path, workspace: str, style_slugs: list[str]
) -> list[Path]:
    """
    Names are explicit per-style so 4 random scenes don't overwrite each other.
    Example: <workspace>_noir_scene_01.png, <workspace>_fantasy_scene_02.png, ...
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w = _slugify(workspace) or "project"
    paths: list[Path] = []
    for i, slug in enumerate(style_slugs, start=1):
        paths.append(out_dir / f"{w}_{_slugify(slug)}_scene_{i:02d}.png")
    return paths


# ---------------------------
# Public API
# ---------------------------


def generate_logo_sync(
    *,
    problem_text: str,
    workspace: str,
    out_dir: str | Path,
    filename: str | None = None,
    model: str = DEFAULT_IMAGE_MODEL,
    size: str | None = None,
    background: str = "opaque",
    quality: str = "high",
    n: int = 1,
    overwrite: bool = False,
    style: str = "sticker",
    allow_text: bool = False,
    palette: str | None = None,
    mode: str = "logo",
    aspect: str = "square",
    style_intensity: str = "overt",
    aperture: float = DEFAULT_APERTURE,
    console: Optional[Console] = None,
    image_model_provider: str = "openai",
    image_provider_kwargs: Optional[dict] = None,
) -> Path:
    """
    Generate images.

    Key change (diversity):
      - We no longer rely on a single prompt with n>1 siblings for scenes.
      - If mode='scene' and style='random' and n>1, we pick n distinct scene styles
        (horror/fantasy/etc) and generate 1 image per style/prompt.

    Return value:
      - Returns the "main" path (first generated image). Additional variants are saved alongside it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # this is how we'll pass through a vision model and provider/url/endpoint
    client_kwargs = {}
    if image_provider_kwargs:
        # Only pass through safe/known kwargs
        for k in ("api_key", "base_url", "organization"):
            if k in image_provider_kwargs and image_provider_kwargs[k]:
                client_kwargs[k] = image_provider_kwargs[k]
    client = OpenAI(**client_kwargs)

    final_size = _normalize_size(size, aspect, mode)
    # Scenes tend to look odd with transparent backgrounds; force opaque.
    final_background = "opaque" if mode == "scene" else background

    # -------------------------
    # Multi-style scene generation (requested change)
    # -------------------------
    if (
        mode == "scene"
        and n > 1
        and SCENE_MULTI_STYLE_DEFAULT
        and (style is None or style.strip().lower() == "random")
        and filename
        is None  # filename implies "single series"; keep single-style naming
    ):
        style_slugs = _choose_n_distinct_styles(n)
        out_paths = _compose_multi_scene_paths(out_dir, workspace, style_slugs)

        # If everything already exists and overwrite is False, skip regeneration.
        if not overwrite and all(p.exists() for p in out_paths):
            return out_paths[0]

        for idx, (style_slug, path) in enumerate(
            zip(style_slugs, out_paths), start=1
        ):
            prompt, _ = _craft_logo_prompt(
                problem_text,
                workspace,
                style=style_slug,
                allow_text=allow_text,
                palette=palette,
                mode="scene",
                style_intensity=style_intensity,
                aperture=aperture,
            )

            extra_title = (
                f"[bold magenta]mode: scene[/bold magenta] [dim]•[/dim] "
                f"aspect: {aspect} [dim]•[/dim] variant {idx}/{n}"
            )
            _render_prompt_panel(
                console=console,
                style_slug=style_slug,
                workspace=workspace,
                prompt=prompt,
                extra_title=extra_title,
            )

            if path.exists() and not overwrite:
                continue

            kwargs = dict(
                model=model,
                prompt=prompt,
                size=final_size,
                n=1,
                quality=quality,
                background=final_background,
            )
            try:
                resp = client.images.generate(**kwargs)
            except Exception:
                # Some models ignore/forbid background=; retry without it
                kwargs.pop("background", None)
                resp = client.images.generate(**kwargs)

            path.write_bytes(base64.b64decode(resp.data[0].b64_json))

        return out_paths[0]

    # -------------------------
    # Default behavior (single-style series)
    #   - Also improved diversity: when n>1, we do n separate prompts (not siblings).
    # -------------------------
    # Build filenames for this series
    prompt0, style_slug0 = _craft_logo_prompt(
        problem_text,
        workspace,
        style=style,
        allow_text=allow_text,
        palette=palette,
        mode=mode,
        style_intensity=style_intensity,
        aperture=aperture,
    )

    main_path, alt_paths = _compose_filenames(
        out_dir, style_slug0, filename, n, mode=mode
    )

    # If everything exists and overwrite is False, return main
    if (
        not overwrite
        and main_path.exists()
        and all(p.exists() for p in alt_paths)
    ):
        return main_path

    # Generate 1 image per prompt (more divergence than n>1 siblings)
    paths = [main_path] + alt_paths
    for idx, path in enumerate(paths, start=1):
        # For n>1, rebuild prompt each time so pools + aperture actually matter
        prompt_i, style_slug_i = (
            (prompt0, style_slug0)
            if idx == 1
            else _craft_logo_prompt(
                problem_text,
                workspace,
                style=style,
                allow_text=allow_text,
                palette=palette,
                mode=mode,
                style_intensity=style_intensity,
                aperture=aperture,
            )
        )

        extra_title = (
            f"[bold magenta]mode: {mode}[/bold magenta] [dim]•[/dim] "
            f"aspect: {aspect} [dim]•[/dim] variant {idx}/{len(paths)}"
        )
        _render_prompt_panel(
            console=console,
            style_slug=style_slug_i,
            workspace=workspace,
            prompt=prompt_i,
            extra_title=extra_title,
        )

        if path.exists() and not overwrite:
            continue

        kwargs = dict(
            model=model,
            prompt=prompt_i,
            size=final_size,
            n=1,
            quality=quality,
            background=final_background,
        )
        try:
            resp = client.images.generate(**kwargs)
        except Exception:
            kwargs.pop("background", None)
            resp = client.images.generate(**kwargs)

        path.write_bytes(base64.b64decode(resp.data[0].b64_json))

    return main_path


def kickoff_logo(
    *,
    problem_text: str,
    workspace: str,
    out_dir: str | Path,
    filename: str | None = None,
    model: str = DEFAULT_IMAGE_MODEL,
    size: str | None = None,  # allow None → computed from aspect/mode
    background: str = "opaque",
    quality: str = "high",
    n: int = 4,
    overwrite: bool = False,
    on_done=None,
    on_error=None,
    style: str = "sticker",
    allow_text: bool = False,
    palette: str | None = None,
    mode: str = "logo",
    aspect: str = "square",
    style_intensity: str = "overt",
    aperture: float = DEFAULT_APERTURE,
    console: Optional[Console] = None,
    image_model: Optional[str] = None,
    image_model_provider: str = "openai",
    image_provider_kwargs: Optional[dict] = None,
):
    _final_model = image_model or model

    def _job() -> Path:
        return generate_logo_sync(
            problem_text=problem_text,
            workspace=workspace,
            out_dir=out_dir,
            filename=filename,
            model=_final_model,
            size=size,
            background=background,
            quality=quality,
            n=n,
            overwrite=overwrite,
            style=style,
            allow_text=allow_text,
            palette=palette,
            mode=mode,
            aspect=aspect,
            style_intensity=style_intensity,
            aperture=aperture,
            console=console,
            image_model_provider=image_model_provider,
            image_provider_kwargs=image_provider_kwargs,
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
