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

# Style presets focused on interesting visuals (not “corporate logo” vibes)
_STYLE_PRESETS = {
    "sigil": (
        "Create an abstract, geometric tech sigil with a strong silhouette. "
        "Flat/vector look (no photorealism), bold shapes, striking negative space, "
        "centered composition, crisp edges, subtle shading only."
    ),
    "patch": (
        "Create a circular mission-patch style emblem. Clean iconography, thick outline, "
        "limited retro palette, slight grain for texture, centered composition."
    ),
    "mascot": (
        "Create a cute minimalist mascot sticker. Simple shapes, flat shading, high saturation, "
        "bold outline, front-facing, playful expression, slight drop shadow."
    ),
    "gradient-glyph": (
        "Create a minimalist glyph centered on a vibrant synthetic gradient background. "
        "High contrast, soft glow, glassy highlights, simple geometry."
    ),
    "brutalist": (
        "Create a stark brutalist composition: heavy blocks, asymmetric layout, coarse texture/grain, "
        "limited palette, experimental shapes, poster-tile look."
    ),
}


# Distinct, bombastic styles; slug drives the filename stem.
_SCENE_TEMPLATES = [
    (
        "horror",
        "Create a dramatic horror-movie scene that is FIRST AND FOREMOST a visual interpretation of '{workspace}'. Cinematic lighting, fog, high contrast. Do not render text. Secondary mood only: {problem_text}.",
    ),
    (
        "sci-fi",
        "Create an epic sci-fi space-opera vista centered on the essence of '{workspace}'. Grand scale, atmospheric perspective, plausible materials. No text. Secondary cues: {problem_text}.",
    ),
    (
        "cyberpunk",
        "Create a neon-soaked cyberpunk night scene whose core concept is '{workspace}'. Wet streets, reflections, dense atmosphere. No words or logos. Secondary vibe: {problem_text}.",
    ),
    (
        "comic",
        "Create a bold comic-book splash composition that visually interprets '{workspace}'. Dynamic angles, inked linework, halftone shading, punchy colors. No lettering. Mood only: {problem_text}.",
    ),
    (
        "fantasy",
        "Create a high-fantasy illustration that channels the spirit of '{workspace}'. Heroic scale, dramatic lighting, painterly detail. No text. Secondary tone: {problem_text}.",
    ),
    (
        "renaissance",
        "Create a Renaissance-style oil painting where the allegorical subject embodies '{workspace}'. Rich chiaroscuro, period palette. No text. Secondary context: {problem_text}.",
    ),
    # ("baroque",     "Create a Baroque dramatic scene focused on the essence of '{workspace}'. Sweeping motion, strong chiaroscuro, theatrical staging. No text. Secondary mood: {problem_text}."),
    (
        "anime",
        "Create an anime-style key visual that interprets '{workspace}'. Cel-shaded clarity, striking silhouette, cinematic framing. No typography. Secondary vibe: {problem_text}.",
    ),
    (
        "mecha",
        "Create a large-scale mecha/industrial hangar scene expressing '{workspace}'. Massive machinery, moody backlight, realistic wear. No text. Secondary mood: {problem_text}.",
    ),
    (
        "steampunk",
        "Create a steampunk laboratory tableau inspired by '{workspace}'. Brass, valves, gauges, warm/cool interplay. No labels or text. Secondary tone: {problem_text}.",
    ),
    # ("art-deco",    "Create an Art-Deco poster-like image whose geometry and ornamentation embody '{workspace}'. Symmetry, sunbursts, stepped forms, elegant palette. No text. Secondary mood: {problem_text}."),
    (
        "ukiyoe",
        "Create a Ukiyo-e woodblock-print style scene that interprets '{workspace}'. Flat shapes, bold contour lines, patterned waves/sky. No text. Secondary vibe: {problem_text}.",
    ),
    (
        "surreal",
        "Create a surrealist composition that makes '{workspace}' feel dreamlike and uncanny. Naturalistic rendering of unreal scenes. No text. Secondary cues: {problem_text}.",
    ),
    (
        "noir",
        "Create a film-noir scene centered on '{workspace}'. High-contrast lighting, rain-slick streets, venetian-blind shadows. No words. Secondary vibe: {problem_text}.",
    ),
    (
        "synthwave",
        "Create a vaporwave/synthwave scene where the energy of '{workspace}' leads. Retro-futurist gradients, nostalgic glow. No text. Secondary mood: {problem_text}.",
    ),
    # ("watercolor",  "Create a watercolor concept-art piece that captures the spirit of '{workspace}'. Soft washes, visible brush texture. No text. Secondary tone: {problem_text}."),
]


# Optional: a default console if one isn’t passed in
_DEFAULT_CONSOLE = Console()

# Color for slug label per style (fallback used if missing)
_STYLE_COLORS = {
    "horror": "red",
    "sci-fi": "cyan",
    "cyberpunk": "magenta",
    "comic": "yellow",
    "fantasy": "green",
    "renaissance": "gold3",
    "baroque": "orange3",
    "anime": "deep_sky_blue1",
    "mecha": "steel_blue",
    "steampunk": "dark_goldenrod",
    "art-deco": "bright_cyan",
    "ukiyoe": "light_sky_blue1",
    "surreal": "plum1",
    "noir": "grey70",
    "synthwave": "hot_pink",
    "watercolor": "aquamarine1",
    # logo-like fallbacks
    "sigil": "white",
    "patch": "bright_white",
    "mascot": "bright_magenta",
    "gradient-glyph": "bright_blue",
    "brutalist": "bright_red",
}

_WORKSPACE_COLOR = "bright_green"  # distinct from slug, border, and prompt
_BORDER_COLOR = "bright_cyan"  # distinct from slug & workspace


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
    style: str = "sigil",
    allow_text: bool = False,
    palette: str | None = None,
) -> tuple[str, str]:
    gist = " ".join(
        line.strip()
        for line in problem_text.strip().splitlines()
        if line.strip()
    )
    txt_rule = (
        "Do not render any text, letters, numbers, logos, or watermarks."
        if not allow_text
        else "If text appears, limit it to a subtle single-letter monogram; avoid words/logos."
    )
    palette_rule = f" Palette hint: {palette}." if palette else ""

    # Random cinematic/genre scene → pick a (slug, template)
    if style in {"random", "random-scene"}:
        style_slug, tpl = random.choice(_SCENE_TEMPLATES)
        prompt = (
            f"Create an image that is FIRST AND FOREMOST a visual interpretation of '{workspace}'. "
            f"{tpl.format(workspace=workspace, problem_text=gist)} "
            f"Treat the following as SECONDARY mood cues only (do not depict literally): {gist}. "
            f"The final image should clearly read as an evocative take on '{workspace}'. {txt_rule}{palette_rule}"
        ).strip()

        return prompt, style_slug

    # Logo-like presets → use the given style key as the slug
    style_slug = style
    base = _STYLE_PRESETS.get(style, _STYLE_PRESETS["sigil"])
    prompt = (
        f"{base} {txt_rule} "
        f"PRIMARY CONCEPT: the essence/energy of '{workspace}' (use as inspiration only; do not render text). "
        f"SECONDARY CONTEXT for tone only: {gist}. "
        f"The piece should feel like a visual interpretation of '{workspace}'.{palette_rule}"
    ).strip()
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
    style: str = "sigil",
    allow_text: bool = False,
    palette: str | None = None,
    console: Optional[Console] = None,  # <-- NEW
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt, style_slug = _craft_logo_prompt(
        problem_text,
        workspace,
        style=style,
        allow_text=allow_text,
        palette=palette,
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
    style: str = "sigil",
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
            console=console,  # pass it down
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
