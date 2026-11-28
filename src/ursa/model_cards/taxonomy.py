"""Physics-domain and task vocabularies for model cards and requirements.

Cards and user requests describe the same science with different words:
"FWI", "full-waveform inversion", and "seismic inversion" should land on
the same concept. Each :class:`Concept` lists curated synonyms that are
matched as whole phrases, so short acronyms such as "PDE" or "MHD" do not
fire inside longer words.

Domains form a two-level hierarchy (``group`` then ``domain``). Matching
uses the hierarchy and a small table of cross-domain links to give partial
credit, for example between wave physics and seismology.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    synonyms: tuple[str, ...] = ()
    parent: str | None = None
    description: str = ""
    # Credit between two children of this concept. A broad umbrella such as
    # "physics" groups fields that are far apart, so its children earn less.
    sibling_similarity: float = 0.5


@dataclass(frozen=True)
class ConceptMatch:
    id: str
    phrase: str
    start: int


@dataclass
class Vocabulary:
    concepts: tuple[Concept, ...]
    related: dict[frozenset[str], float] = field(default_factory=dict)

    def __post_init__(self):
        self._by_id = {concept.id: concept for concept in self.concepts}

    @property
    def ids(self) -> list[str]:
        return list(self._by_id)

    def get(self, concept_id: str) -> Concept | None:
        return self._by_id.get(concept_id)

    def parent(self, concept_id: str) -> str | None:
        concept = self._by_id.get(concept_id)
        return concept.parent if concept else None

    def match(self, text: str) -> list[ConceptMatch]:
        """Return every synonym hit in ``text``, in reading order."""
        if not text:
            return []
        hits: list[ConceptMatch] = []
        for concept in self.concepts:
            pattern = _concept_pattern(self, concept.id)
            if pattern is None:
                continue
            for found in pattern.finditer(text):
                hits.append(
                    ConceptMatch(concept.id, found.group(0), found.start())
                )
        # The longest phrase wins: "weather" inside "space weather" is not a
        # separate weather mention.
        spans = [(h.start, h.start + len(h.phrase), h) for h in hits]
        kept = [
            hit
            for start, end, hit in spans
            if not any(
                s <= start
                and end <= e
                and e - s > end - start
                and other.id != hit.id
                for s, e, other in spans
            )
        ]
        return sorted(kept, key=lambda hit: hit.start)

    def normalize(self, value: str) -> str | None:
        """Map an id, label, or synonym to a concept id."""
        if not value:
            return None
        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        if key in self._by_id:
            return key
        hits = self.match(value)
        if hits:
            return hits[0].id
        for concept in self.concepts:
            if concept.label.lower() == value.strip().lower():
                return concept.id
        return None

    def similarity(self, a: str, b: str) -> float:
        """Similarity in [0, 1] between two concept ids."""
        if a == b:
            return 1.0
        explicit = self.related.get(frozenset((a, b)))
        if explicit is not None:
            return explicit
        parent_a, parent_b = self.parent(a), self.parent(b)
        if parent_a is not None and parent_a == parent_b:
            group = self._by_id.get(parent_a)
            return group.sibling_similarity if group else 0.5
        if a in (parent_b,) or b in (parent_a,):
            return 0.7
        return 0.0

    def extended(self, extra: dict | None) -> Vocabulary:
        """Return a copy with additional or overriding concepts.

        ``extra`` maps an id to a synonym list or to a mapping with
        ``label``, ``synonyms``, ``parent``, ``description``, and
        ``sibling_similarity`` keys, so a team can teach URSA a new field
        from configuration::

            {"magnetic_confinement": {"synonyms": ["tokamak equilibrium"],
                                      "parent": "plasma_physics"}}
        """
        if not extra:
            return self
        concepts = {concept.id: concept for concept in self.concepts}
        for concept_id, spec in extra.items():
            if isinstance(spec, str):
                spec = [spec]
            if isinstance(spec, (list, tuple)):
                spec = {"synonyms": list(spec)}
            name = str(concept_id).strip().lower().replace(" ", "_")
            synonyms = [str(s).lower() for s in spec.get("synonyms", [])]
            synonyms.append(name.replace("_", " "))
            concepts[name] = Concept(
                id=name,
                label=spec.get("label") or name.replace("_", " ").capitalize(),
                synonyms=tuple(dict.fromkeys(synonyms)),
                parent=spec.get("parent"),
                description=spec.get("description", ""),
                sibling_similarity=float(spec.get("sibling_similarity", 0.5)),
            )
        return Vocabulary(tuple(concepts.values()), dict(self.related))

    def describe(self) -> str:
        """Bullet list of ids and labels for prompts."""
        lines = []
        for concept in self.concepts:
            prefix = "  - " if concept.parent else "- "
            note = f": {concept.description}" if concept.description else ""
            lines.append(f"{prefix}{concept.id} ({concept.label}){note}")
        return "\n".join(lines)


def _phrase_regex(phrase: str) -> str:
    parts = [re.escape(part) for part in re.split(r"[\s_-]+", phrase) if part]
    body = r"[\s_\-]*".join(parts)
    # Hyphen-aware boundaries keep "the well" out of "the well-known" and
    # "helmholtz" out of "kelvin-helmholtz".
    # An optional plural ending lets "HOMO-LUMO gap" match "HOMO-LUMO gaps".
    return (
        rf"(?<![A-Za-z0-9])(?<![A-Za-z0-9]-){body}(?:e?s)?"
        rf"(?![A-Za-z0-9])(?!-[A-Za-z0-9])"
    )


@cache
def _compiled(phrases: tuple[str, ...]) -> re.Pattern[str] | None:
    if not phrases:
        return None
    ordered = sorted(set(phrases), key=len, reverse=True)
    return re.compile("|".join(_phrase_regex(p) for p in ordered), re.I)


def _concept_pattern(vocab: Vocabulary, concept_id: str):
    concept = vocab.get(concept_id)
    if concept is None:
        return None
    return _compiled(concept.synonyms)


# Physics domains ---------------------------------------------------------

_DOMAIN_CONCEPTS = (
    Concept(
        "earth_system",
        "Earth system science",
        ("earth system", "earth science", "geoscience"),
    ),
    Concept(
        "weather",
        "Weather",
        (
            "weather",
            "numerical weather prediction",
            "nwp",
            "medium-range forecast",
            "medium-range forecasting",
            "nowcasting",
            "tropical cyclone",
            "tropical cyclones",
            "hurricane",
            "atmospheric dynamics",
            "era5",
            "reanalysis",
            "precipitation forecast",
        ),
        parent="earth_system",
        description="Weather forecasting and atmospheric dynamics",
    ),
    Concept(
        "climate",
        "Climate",
        (
            "climate",
            "climate model",
            "climate emulator",
            "climate emulation",
            "cmip",
            "cmip6",
            "merra-2",
            "merra2",
            "earth system model",
        ),
        parent="earth_system",
        description="Climate emulation and projection",
    ),
    Concept(
        "atmospheric_chemistry",
        "Atmospheric chemistry",
        (
            "atmospheric chemistry",
            "air quality",
            "aerosol",
            "aerosols",
            "air pollution",
        ),
        parent="earth_system",
    ),
    Concept(
        "ocean",
        "Ocean",
        (
            "ocean",
            "oceanography",
            "ocean waves",
            "ocean wave",
            "sea surface temperature",
            "ocean circulation",
            "sea ice",
        ),
        parent="earth_system",
    ),
    Concept(
        "earth_observation",
        "Earth observation",
        (
            "earth observation",
            "remote sensing",
            "satellite imagery",
            "satellite images",
            "multispectral",
            "hyperspectral",
            "sentinel-1",
            "sentinel-2",
            "landsat",
            "harmonized landsat sentinel",
            "land cover",
            "flood mapping",
            "burn scar",
            "geospatial",
        ),
        parent="earth_system",
    ),
    Concept(
        "cryosphere",
        "Cryosphere",
        (
            "cryosphere",
            "glaciology",
            "glacier",
            "glaciers",
            "ice sheet",
            "ice sheets",
            "permafrost",
            "snow cover",
        ),
        parent="earth_system",
    ),
    Concept(
        "hydrology",
        "Hydrology",
        (
            "hydrology",
            "hydrological",
            "streamflow",
            "rainfall-runoff",
            "river discharge",
            "flood forecasting",
        ),
        parent="earth_system",
    ),
    Concept(
        "geophysics",
        "Geophysics",
        ("geophysics", "geophysical", "solid earth"),
    ),
    Concept(
        "seismology",
        "Seismology",
        (
            "seismology",
            "seismological",
            "seismic",
            "seismogram",
            "seismograms",
            "seismometer",
            "earthquake",
            "earthquakes",
            "phase picking",
            "p-wave",
            "s-wave",
            "p wave",
            "s wave",
            "microseismic",
            "focal mechanism",
            "seisbench",
        ),
        parent="geophysics",
        description="Earthquake and passive seismic waveform analysis",
    ),
    Concept(
        "exploration_geophysics",
        "Exploration geophysics",
        (
            "full waveform inversion",
            "full-waveform inversion",
            "fwi",
            "seismic imaging",
            "seismic inversion",
            "reverse time migration",
            "seismic migration",
            "velocity model building",
            "velocity model",
            "velocity models",
            "seismic interpretation",
            "fault detection",
            "salt body",
            "seismic facies",
            "reflection seismic",
            "prestack",
            "poststack",
            "seg-y",
            "segy",
            "openfwi",
            "marmousi",
            "impedance inversion",
        ),
        parent="geophysics",
        description="Seismic imaging, inversion, and interpretation",
    ),
    Concept(
        "subsurface_flow",
        "Subsurface flow",
        (
            "reservoir simulation",
            "porous media",
            "darcy flow",
            "groundwater",
            "hydrogeology",
            "co2 storage",
            "carbon storage",
            "carbon sequestration",
            "multiphase flow",
            "geothermal",
        ),
        parent="geophysics",
    ),
    Concept(
        "electromagnetic_geophysics",
        "Electromagnetic geophysics",
        (
            "magnetotelluric",
            "magnetotellurics",
            "ground penetrating radar",
            "csem",
            "electrical resistivity tomography",
        ),
        parent="geophysics",
    ),
    Concept(
        "continuum_physics",
        "Continuum physics and PDEs",
        ("continuum mechanics", "continuum dynamics"),
    ),
    Concept(
        "pde_general",
        "General PDE surrogates",
        (
            "pde",
            "pdes",
            "partial differential equation",
            "partial differential equations",
            "neural operator",
            "neural operators",
            "operator learning",
            "multiphysics",
            "multi-physics",
            "physics simulation",
            "physics simulations",
            "physical simulation",
            "physics-simulation",
            "scientific machine learning",
            "sciml",
            "the well",
        ),
        parent="continuum_physics",
        description="Surrogates for generic or multiple PDE systems",
    ),
    Concept(
        "fluid_dynamics",
        "Fluid dynamics",
        (
            "fluid dynamics",
            "fluid flow",
            "fluids",
            "cfd",
            "navier-stokes",
            "navier stokes",
            "turbulence",
            "turbulent",
            "incompressible flow",
            "compressible flow",
            "compressible euler",
            "euler equations",
            "shear flow",
            "rayleigh-benard",
            "rayleigh-bénard",
            "kelvin-helmholtz",
            "rayleigh-taylor",
            "shallow water",
            "aerodynamics",
            "airfoil",
            "vortex shedding",
            "flowbench",
            "convection",
        ),
        parent="continuum_physics",
    ),
    Concept(
        "wave_physics",
        "Wave physics and acoustics",
        (
            "acoustic",
            "acoustics",
            "acoustic wave",
            "acoustic waves",
            "wave equation",
            "wave propagation",
            "helmholtz",
            "acoustic scattering",
            "elastic wave",
            "elastic waves",
            "elastodynamic",
            "wavefield",
            "wavefields",
            "ultrasound",
        ),
        parent="continuum_physics",
        description="Acoustic and elastic wave propagation",
    ),
    Concept(
        "heat_transfer",
        "Heat transfer",
        (
            "heat transfer",
            "heat equation",
            "heat conduction",
            "thermal conduction",
            "diffusion equation",
        ),
        parent="continuum_physics",
    ),
    Concept(
        "solid_mechanics",
        "Solid mechanics",
        (
            "solid mechanics",
            "linear elasticity",
            "structural mechanics",
            "fracture",
            "plasticity",
            "finite element",
            "stress field",
            "strain field",
        ),
        parent="continuum_physics",
    ),
    Concept(
        "reaction_diffusion",
        "Reaction-diffusion and pattern formation",
        (
            "reaction-diffusion",
            "reaction diffusion",
            "gray-scott",
            "cahn-hilliard",
            "allen-cahn",
            "phase field",
            "phase-field",
            "pattern formation",
            "active matter",
        ),
        parent="continuum_physics",
    ),
    Concept(
        "electromagnetics",
        "Electromagnetics and optics",
        (
            "maxwell's equations",
            "maxwell equations",
            "electromagnetic",
            "electromagnetics",
            "photonics",
            "optics",
            "antenna",
            "metasurface",
        ),
        parent="continuum_physics",
    ),
    # A grouping without phrases: generic words such as "physics-informed"
    # never tag a card with it, but a broad request may name it.
    Concept(
        "physics",
        "Physics",
        (),
        description="Any physics field; prefer a sub-domain when one fits",
        sibling_similarity=0.2,
    ),
    Concept(
        "plasma_physics",
        "Plasma physics and fusion",
        (
            "plasma",
            "plasma physics",
            "mhd",
            "magnetohydrodynamic",
            "magnetohydrodynamics",
            "nuclear fusion",
            "fusion energy",
            "fusion plasma",
            "fusion plasmas",
            "tokamak",
            "stellarator",
            "gyrokinetic",
        ),
        parent="physics",
    ),
    Concept(
        "astrophysics",
        "Astrophysics",
        (
            "astrophysics",
            "astrophysical",
            "astronomy",
            "supernova",
            "neutron star",
            "neutron stars",
            "galaxy",
            "galaxies",
            "cosmology",
            "cosmological",
            "stellar convection",
            "stellar evolution",
            "stellar atmosphere",
            "stellar atmospheres",
            "black hole",
            "gravitational wave",
            "gravitational waves",
            "exoplanet",
            "sky survey",
        ),
        parent="physics",
    ),
    Concept(
        "space_physics",
        "Space physics and heliophysics",
        (
            "heliophysics",
            "space physics",
            "space weather",
            "solar physics",
            "solar wind",
            "solar flare",
            "solar flares",
            "solar dynamics",
            "solar dynamics observatory",
            "coronal mass ejection",
            "magnetosphere",
            "magnetospheric",
            "ionosphere",
            "ionospheric",
            "sunspot",
        ),
        description="The Sun, solar wind, and near-Earth space environment",
        parent="physics",
    ),
    Concept(
        "condensed_matter",
        "Condensed matter physics",
        (
            "condensed matter",
            "many-body",
            "quantum many-body",
            "spin system",
            "spin systems",
            "ising model",
            "hubbard model",
            "lattice model",
            "superconductivity",
            "superconductor",
            "quantum materials",
        ),
        parent="physics",
    ),
    Concept(
        "particle_physics",
        "Particle physics",
        (
            "particle physics",
            "high energy physics",
            "high-energy physics",
            "collider",
            "jet tagging",
            "calorimeter",
            "lhc",
        ),
        parent="physics",
    ),
    Concept(
        "nuclear_engineering",
        "Nuclear engineering",
        (
            "nuclear reactor",
            "neutron transport",
            "neutronics",
            "radiation transport",
        ),
        parent="physics",
    ),
    Concept(
        "materials",
        "Materials science",
        ("materials science", "materials", "material science"),
    ),
    Concept(
        "materials_atomistic",
        "Atomistic materials modeling",
        (
            "interatomic potential",
            "interatomic potentials",
            "machine learning potential",
            "machine-learning potential",
            "mlip",
            "mlff",
            "force field",
            "molecular dynamics",
            "density functional theory",
            "dft",
            "crystal",
            "crystals",
            "crystalline",
            "materials project",
            "omat24",
            "oc20",
            "oc22",
            "catalysis",
            "phonon",
            "phonons",
            "atomistic",
            "matbench",
            "inorganic materials",
            "alloy",
            "alloys",
        ),
        parent="materials",
    ),
    Concept(
        "light_source",
        "Light-source and X-ray science",
        (
            "light source",
            "lightsource",
            "synchrotron",
            "x-ray diffraction",
            "x-ray",
            "ptychography",
            "beamline",
            "x-ray crystallography",
            "serial crystallography",
            "diffraction pattern",
            "tomography reconstruction",
        ),
        parent="materials",
    ),
    Concept(
        "chemistry",
        "Chemistry",
        ("chemistry", "chemical"),
    ),
    Concept(
        "molecular_chemistry",
        "Molecular chemistry",
        (
            "molecule",
            "molecules",
            "molecular",
            "smiles",
            "selfies",
            "qm9",
            "moleculenet",
            "quantum chemistry",
            "drug discovery",
            "electrolyte",
            "electrolytes",
            "retrosynthesis",
        ),
        parent="chemistry",
    ),
    Concept(
        "life_science",
        "Life science",
        ("biology", "biological", "life science"),
    ),
    Concept(
        "biomolecular",
        "Biomolecules",
        (
            "protein",
            "proteins",
            "protein structure",
            "amino acid",
            "genomics",
            "dna",
            "rna",
        ),
        parent="life_science",
    ),
)

_DOMAIN_RELATED = {
    frozenset(("wave_physics", "seismology")): 0.6,
    frozenset(("wave_physics", "exploration_geophysics")): 0.7,
    frozenset(("seismology", "exploration_geophysics")): 0.6,
    frozenset(("fluid_dynamics", "pde_general")): 0.6,
    frozenset(("wave_physics", "pde_general")): 0.5,
    frozenset(("heat_transfer", "pde_general")): 0.5,
    frozenset(("reaction_diffusion", "pde_general")): 0.5,
    frozenset(("solid_mechanics", "pde_general")): 0.4,
    frozenset(("electromagnetics", "pde_general")): 0.4,
    frozenset(("weather", "climate")): 0.7,
    frozenset(("weather", "ocean")): 0.4,
    frozenset(("climate", "ocean")): 0.5,
    frozenset(("weather", "atmospheric_chemistry")): 0.5,
    frozenset(("weather", "hydrology")): 0.4,
    frozenset(("earth_observation", "climate")): 0.3,
    frozenset(("earth_observation", "weather")): 0.3,
    frozenset(("plasma_physics", "fluid_dynamics")): 0.5,
    frozenset(("astrophysics", "fluid_dynamics")): 0.4,
    frozenset(("astrophysics", "plasma_physics")): 0.4,
    frozenset(("materials_atomistic", "molecular_chemistry")): 0.6,
    frozenset(("light_source", "materials_atomistic")): 0.4,
    frozenset(("subsurface_flow", "fluid_dynamics")): 0.5,
    frozenset(("subsurface_flow", "exploration_geophysics")): 0.4,
    frozenset(("pde_general", "materials")): 0.3,
    frozenset(("space_physics", "astrophysics")): 0.6,
    frozenset(("space_physics", "plasma_physics")): 0.5,
    frozenset(("condensed_matter", "materials_atomistic")): 0.5,
    frozenset(("cryosphere", "climate")): 0.6,
    frozenset(("cryosphere", "ocean")): 0.4,
}

DOMAINS = Vocabulary(_DOMAIN_CONCEPTS, _DOMAIN_RELATED)


# Tasks -------------------------------------------------------------------

_TASK_CONCEPTS = (
    Concept(
        "forecasting",
        "Forecasting",
        (
            "forecast",
            "forecasts",
            "forecasting",
            "weather prediction",
            "nowcasting",
            "time-series forecasting",
            "time series forecasting",
            "lead time",
            "lead times",
            "time-series-forecasting",
        ),
        description="Predict future states from current observations",
    ),
    Concept(
        "surrogate_simulation",
        "Surrogate simulation",
        (
            "surrogate",
            "surrogates",
            "surrogate model",
            "surrogate modeling",
            "emulator",
            "emulation",
            "autoregressive rollout",
            "autoregressive prediction",
            "autoregressive forecasting",
            "autoregressive emulator",
            "rollout",
            "rollouts",
            "time stepping",
            "next-step prediction",
            "forward modeling",
            "forward modelling",
            "forward simulation",
            "solve pdes",
            "pde solving",
            "temporal evolution",
            "first-to-last prediction",
        ),
        description="Replace or accelerate a numerical simulation",
    ),
    Concept(
        "inversion",
        "Inversion",
        (
            "seismic inversion",
            "waveform inversion",
            "geophysical inversion",
            "joint inversion",
            "inverse problem",
            "inverse problems",
            "inverse modeling",
            "full waveform inversion",
            "full-waveform inversion",
            "fwi",
            "parameter estimation",
            "velocity model building",
        ),
        description="Infer model parameters from observations",
    ),
    Concept(
        "imaging",
        "Imaging and reconstruction",
        (
            "seismic imaging",
            "seismic migration",
            "depth migration",
            "reverse time migration",
            "kirchhoff migration",
            "image reconstruction",
            "tomographic reconstruction",
            "ptychography",
        ),
    ),
    Concept(
        "property_prediction",
        "Property prediction",
        (
            "property prediction",
            "properties prediction",
            "molecular property",
            "molecular properties",
            "predict properties",
            "material properties",
            "band gap",
            "formation energy",
            "homo-lumo gap",
            "tabular-regression",
        ),
        description="Regress physical or chemical properties",
    ),
    Concept(
        "interatomic_potential",
        "Interatomic potential",
        (
            "interatomic potential",
            "interatomic potentials",
            "energies and forces",
            "energy and forces",
            "forces and stresses",
            "machine learning potential",
            "mlip",
            "structure relaxation",
            "universal potential",
            "universal interatomic potential",
            "neural network potential",
            "potential energy surface",
        ),
    ),
    Concept(
        "classification",
        "Classification",
        (
            "classification",
            "classifier",
            "classify",
            "image-classification",
        ),
    ),
    Concept(
        "segmentation",
        "Segmentation",
        (
            "segmentation",
            "semantic segmentation",
            "image-segmentation",
            "pixel-wise classification",
            "fault segmentation",
            "land cover mapping",
            "flood mapping",
        ),
    ),
    Concept(
        "detection",
        "Detection",
        (
            "detection",
            "event detection",
            "earthquake detection",
            "anomaly detection",
            "object detection",
            "object-detection",
            "cyclone tracking",
            "tropical-cyclone-tracking",
        ),
    ),
    Concept(
        "phase_picking",
        "Phase picking",
        (
            "phase picking",
            "phase picker",
            "arrival picking",
            "first-break picking",
            "first break picking",
            "first arrival",
            "phasenet",
        ),
    ),
    Concept(
        "denoising",
        "Denoising",
        (
            "denoising",
            "denoise",
            "noise attenuation",
            "noise suppression",
            "deblending",
        ),
    ),
    Concept(
        "super_resolution",
        "Super-resolution and downscaling",
        (
            "super-resolution",
            "super resolution",
            "downscaling",
            "resolution enhancement",
            "coarse-to-fine",
            "reconstruct fine-scale",
        ),
    ),
    Concept(
        "interpolation",
        "Interpolation and completion",
        (
            "interpolation",
            "inpainting",
            "gap filling",
            "data completion",
            "trace interpolation",
            "missing data reconstruction",
        ),
    ),
    Concept(
        "generation",
        "Generative modeling",
        (
            "generative",
            "generative model",
            "molecule generation",
            "de novo design",
            "conditional generation",
            "probabilistic generation",
            "structure generation",
            "crystal generation",
            "materials generation",
            "motion generation",
            "inverse design",
        ),
    ),
    Concept(
        "embedding",
        "Representation learning",
        (
            "embedding vector",
            "embedding vectors",
            "produce embeddings",
            "molecular embeddings",
            "representation learning",
            "feature extraction",
            "feature-extraction",
            "image-feature-extraction",
            "pretrained encoder",
            "pretrained backbone",
            "foundation backbone",
            "masked autoencoder",
            "latent representations",
            "suitable for fine-tuning",
        ),
        description="Produce features or embeddings for downstream use",
    ),
    Concept(
        "data_assimilation",
        "Data assimilation",
        ("data assimilation", "4d-var", "ensemble kalman"),
    ),
    Concept(
        "uncertainty_quantification",
        "Uncertainty quantification",
        (
            "uncertainty quantification",
            "uncertainty estimates",
            "uncertainty sampling",
            "ensemble forecast",
            "probabilistic forecast",
            "probabilistic forecasting",
            "predictive uncertainty",
        ),
    ),
)

_TASK_RELATED = {
    frozenset(("forecasting", "surrogate_simulation")): 0.6,
    frozenset(("inversion", "imaging")): 0.6,
    frozenset(("segmentation", "classification")): 0.5,
    frozenset(("detection", "classification")): 0.5,
    frozenset(("phase_picking", "detection")): 0.6,
    frozenset(("super_resolution", "interpolation")): 0.5,
    frozenset(("denoising", "interpolation")): 0.4,
    frozenset(("property_prediction", "interatomic_potential")): 0.5,
    frozenset(("generation", "surrogate_simulation")): 0.3,
    frozenset(("embedding", "classification")): 0.3,
    frozenset(("embedding", "segmentation")): 0.3,
    frozenset(("uncertainty_quantification", "forecasting")): 0.4,
    frozenset(("data_assimilation", "forecasting")): 0.4,
}

TASKS = Vocabulary(_TASK_CONCEPTS, _TASK_RELATED)


# Mappings from structured card metadata -----------------------------------

PIPELINE_TASKS = {
    "image-classification": "classification",
    "audio-classification": "classification",
    "text-classification": "classification",
    "tabular-classification": "classification",
    "image-segmentation": "segmentation",
    "object-detection": "detection",
    "feature-extraction": "embedding",
    "image-feature-extraction": "embedding",
    "time-series-forecasting": "forecasting",
    "tabular-regression": "property_prediction",
    "unconditional-image-generation": "generation",
    "fill-mask": "embedding",
}

GENESIS_SCIENCE_DOMAINS = {
    "materials": "materials",
    "biology": "life_science",
    "lightsource": "light_source",
    "light-source": "light_source",
    "fusion": "plasma_physics",
    "plasma": "plasma_physics",
    "climate": "climate",
    "weather": "weather",
    "physics-simulation": "pde_general",
    "physics_simulation": "pde_general",
    "chemistry": "chemistry",
    "hep": "particle_physics",
    "high-energy-physics": "particle_physics",
    "nuclear": "nuclear_engineering",
    "geoscience": "earth_system",
    "earth-science": "earth_system",
    "seismology": "seismology",
    "geophysics": "geophysics",
    "astrophysics": "astrophysics",
    "astronomy": "astrophysics",
    "fluids": "fluid_dynamics",
    "heliophysics": "space_physics",
    "space-weather": "space_physics",
    "cryosphere": "cryosphere",
    "condensed-matter": "condensed_matter",
}


def best_similarity(
    vocab: Vocabulary, wanted: list[str], offered: list[str]
) -> tuple[float, str | None, str | None]:
    """Best pairwise similarity between two id lists."""
    best = (0.0, None, None)
    for want in wanted:
        for offer in offered:
            score = vocab.similarity(want, offer)
            if score > best[0]:
                best = (score, want, offer)
    return best
