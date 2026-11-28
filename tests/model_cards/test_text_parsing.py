import pytest

from ursa.model_cards import patterns as P
from ursa.model_cards.markdown import (
    logical_lines,
    parse_bibtex,
    parse_card_text,
    parse_key_values,
    remove_template_residue,
    split_front_matter,
    strip_inline_markdown,
)
from ursa.model_cards.taxonomy import DOMAINS, TASKS


def test_front_matter_and_invalid_yaml():
    data, body, error = split_front_matter(
        "---\nlicense: mit\ntags: [a]\n---\n# Title\n"
    )
    assert data == {"license": "mit", "tags": ["a"]}
    assert body == "# Title\n"
    assert error is None
    _, _, error = split_front_matter("---\nlicense: [mit\n---\nBody")
    assert error and error.startswith("Invalid YAML")


def test_headings_inside_code_fences_are_not_sections():
    card = parse_card_text(
        "# Model\n\n```python\n# not a heading\nx = 1\n```\n\n## Inputs\n\n- a\n"
    )
    assert [s.title for s in card.sections] == ["Model", "Inputs"]
    assert card.code_blocks[0].code == "# not a heading\nx = 1"


def test_inline_markdown_keeps_identifiers_and_code_spans():
    text = "Field `av_density`, a _NumPy_ array with **bold** text and pli240420_fp16_full"
    assert strip_inline_markdown(text) == (
        "Field av_density, a NumPy array with bold text and pli240420_fp16_full"
    )


def test_key_values_follow_nvidia_layout():
    text = (
        "**Input Type:** Tensor (72 variables) <br>\n"
        "**Input Parameters:** Six Dimensional (6D) (batch, time, variable,\n"
        "latitude, longitude) <br>\n"
        "**Other Properties Related to Input:**\n"
        "- Input grid: 0.25 degree 721 x 1440\n"
        "- Variables: `u10m`, `v10m`\n"
        "+ **2026-06-24** initial release\n"
    )
    pairs = dict(parse_key_values(text))
    assert pairs["Input Type"] == "Tensor (72 variables)"
    assert pairs["Input Parameters"].endswith(
        "(batch, time, variable, latitude, longitude)"
    )
    assert pairs["Other Properties Related to Input"].startswith(
        "Input grid: 0.25 degree"
    )
    assert "2026-06-24" not in pairs


def test_logical_lines_join_wrapped_prose():
    lines = logical_lines(
        "First line of a\nwrapped sentence.\n\n- item one\n- item two\n"
    )
    assert lines == [
        "First line of a wrapped sentence.",
        "- item one",
        "- item two",
    ]


def test_template_residue_removal_keeps_code_and_real_text():
    text = (
        "Examples:\n"
        "1. Input: The model was trained with 224 x 224 input images and text.\n"
        "Real fact about the model.\n"
        "```bibtex\ntitle = {{A Title}}\n```\n"
    )
    cleaned, hits = remove_template_residue(text)
    assert hits == 2
    assert "Real fact about the model." in cleaned
    assert "{{A Title}}" in cleaned


def test_bibtex_parsing():
    entries = parse_bibtex(
        "@article{Key2026,\n  title = {{Nested {Braces} Title}},\n"
        "  doi = {10.1000/xyz},\n  year = {2026},\n}"
    )
    assert entries[0]["title"] == "Nested Braces Title"
    assert entries[0]["doi"] == "10.1000/xyz"


@pytest.mark.parametrize(
    "number, unit, expected",
    [
        ("710,867,670", None, 710_867_670),
        ("1.3", "B", 1_300_000_000),
        ("26.9", "M", 26_900_000),
        ("158", "million", 158_000_000),
        ("12", None, None),
    ],
)
def test_parse_count(number, unit, expected):
    assert P.parse_count(number, unit) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("shape `(H, W)`", ["H", "W"]),
        ("a grid of 128 x 128", ["128", "128"]),
        (
            "(batch, time, lead time, variable)",
            ["batch", "time", "lead time", "variable"],
        ),
        ("64x64x64 cells", ["64", "64", "64"]),
        ("no shape here (optional)", []),
    ],
)
def test_parse_shape(text, expected):
    assert P.parse_shape(text) == expected


@pytest.mark.parametrize(
    "sentence, step, horizon",
    [
        (
            "The model is trained to produce 5-day forecasts and supports rollouts up to 365 days.",
            "5 days",
            "365 days",
        ),
        (
            "A 60-day global forecast at 0.25 degrees, 6-hourly resolution.",
            "6 hours",
            "60 days",
        ),
        (
            "Time step: 10 microseconds per rollout step.",
            "10 microseconds",
            None,
        ),
        ("Use dt = 0.01 s for stability.", "0.01 s", None),
        (
            "The lead time was chosen from [0, 6, 12, 24] hours.",
            None,
            "[0, 6, 12, 24] hours",
        ),
    ],
)
def test_time_step_and_horizon_are_separate(sentence, step, horizon):
    found = P.find_time_step(sentence)
    assert (found[0] if found else None) == step
    found_horizon = P.find_horizon(sentence, exclude=step)
    assert (found_horizon[0] if found_horizon else None) == horizon


def test_context_steps():
    assert (
        P.find_context_steps(
            "The model takes data from two timestamps as input."
        )[0]
        == 2
    )
    assert (
        P.find_context_steps("In time, the context length was set to 4.")[0]
        == 4
    )


def test_license_normalization_and_commercial_use():
    assert P.normalize_license("Apache 2.0") == "apache-2.0"
    assert P.normalize_license("CC-BY-NC-4.0") == "cc-by-nc-4.0"
    assert P.commercial_use_for("cc-by-nc-4.0") is False
    assert P.commercial_use_for("mit") is True
    assert P.commercial_use_for("other") is None
    assert P.BOTH_USES.search(
        "This model is ready for commercial/non-commercial use."
    )


def test_cpu_statements_ignore_comparisons():
    assert P.CPU_OK.search("Inference can run on CPU or a single GPU.")
    assert P.CPU_OK.search("CPU can run the current small configuration.")
    assert P.COMPARATIVE.search(
        "Faster inference compared to CPU-only solutions."
    )


def test_spatial_dims_ignore_architecture_components():
    assert P.SPATIAL_DIMS.search("3D turbulence simulations")
    assert P.SPATIAL_DIMS.search("two-dimensional flow fields")
    assert not P.SPATIAL_DIMS.search(
        "Our 3D patch embeddings use a 3D convolutional layer."
    )


@pytest.mark.parametrize(
    "text, domain",
    [
        ("Navier-Stokes turbulence surrogate", "fluid_dynamics"),
        ("full-waveform inversion of seismic data", "exploration_geophysics"),
        ("tokamak edge plasma", "plasma_physics"),
        ("SMILES property prediction on QM9", "molecular_chemistry"),
        (
            "machine learning interatomic potentials for alloys",
            "materials_atomistic",
        ),
        ("medium-range weather forecasting with ERA5", "weather"),
        ("supernova explosion simulations", "astrophysics"),
        ("Helmholtz equation and acoustic scattering", "wave_physics"),
        ("Sentinel-2 land cover mapping", "earth_observation"),
    ],
)
def test_domain_vocabulary_covers_many_physics_branches(text, domain):
    assert domain in {hit.id for hit in DOMAINS.match(text)}


def test_vocabulary_avoids_common_false_positives():
    noisy = "feature fusion, stellar performance, temperature inversion, data migration, patch embedding, the well-known"
    assert DOMAINS.match(noisy) == []
    assert TASKS.match(noisy) == []
    assert [h.id for h in DOMAINS.match("Kelvin-Helmholtz instability")] == [
        "fluid_dynamics"
    ]


def test_vocabulary_similarity_and_extension():
    assert DOMAINS.similarity("weather", "weather") == 1.0
    assert DOMAINS.similarity("weather", "climate") == 0.7
    assert DOMAINS.similarity("seismology", "subsurface_flow") == 0.5
    assert DOMAINS.similarity("weather", "molecular_chemistry") == 0.0
    extended = DOMAINS.extended({
        "glaciology": {"synonyms": ["ice sheet"], "parent": "earth_system"}
    })
    assert "glaciology" in {
        h.id for h in extended.match("Antarctic ice sheet dynamics")
    }
    assert extended.similarity("glaciology", "climate") == 0.5
    assert DOMAINS.get("glaciology") is None


@pytest.mark.parametrize(
    ("text", "unit"),
    [
        ("Temperature samples in kelvin on a periodic grid.", "K"),
        ("Thermal diffusivity in m^2/s.", "m^2/s"),
        ("Elapsed time in seconds.", "s"),
        ("temperature in degrees Celsius", "°C"),
        ("precipitation in mm/day", "mm/day"),
        ("wind in m s-1", "m s-1"),
        ("units: K", "K"),
        # Several units describe several quantities: no single unit.
        ("u in m/s and T in K", None),
        # Slashes and letters that are not units.
        ("positions in x/y coordinates", None),
        ("split in train/test sets", None),
        ("inputs in K-band imagery", None),
        ("trained in 3 days", None),
    ],
)
def test_find_units(text, unit):
    assert P.find_units(text) == unit


def test_physics_fields_group_without_broad_credit():
    children = {c.id for c in DOMAINS.concepts if c.parent == "physics"}
    assert children == {
        "plasma_physics",
        "astrophysics",
        "space_physics",
        "condensed_matter",
        "particle_physics",
        "nuclear_engineering",
    }
    # A broad umbrella gives its children little credit for each other, close
    # pairs keep their own links, and a broad request still reaches each child.
    assert DOMAINS.similarity("particle_physics", "condensed_matter") == 0.2
    assert DOMAINS.similarity("space_physics", "astrophysics") == 0.6
    assert DOMAINS.similarity("physics", "nuclear_engineering") == 0.7
    assert DOMAINS.normalize("physics") == "physics"
    assert not DOMAINS.match("A physics-informed physics foundation model.")
