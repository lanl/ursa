from tests.agents.utils import RecordingChatModel
from ursa.model_cards.extract import CardDocument, extract_model_card
from ursa.model_cards.matching import blend_review, rank_models
from ursa.model_cards.requirements import (
    RequirementsExtraction,
    heuristic_requirements,
    merge_requirements,
    parse_requirements,
)
from ursa.model_cards.schema import (
    CardSource,
    ChannelSpec,
    ComputeInfo,
    IOSpec,
    LicenseInfo,
    ModelCard,
    UserRequirements,
)

from . import cards


def card(model_id, **fields):
    defaults = {
        "name": model_id,
        "source": CardSource(kind="file", uri=model_id),
    }
    return ModelCard(id=model_id, **{**defaults, **fields})


CATALOG = [
    card(
        "flow",
        summary="Surrogate for 2D incompressible fluid flow.",
        domains=["fluid_dynamics"],
        tasks=["surrogate_simulation"],
        inputs=[
            IOSpec(
                name="fields",
                kind="field",
                channels=[
                    ChannelSpec(name="velocity"),
                    ChannelSpec(name="pressure"),
                ],
            )
        ],
        outputs=[
            IOSpec(
                name="fields",
                kind="field",
                channels=[
                    ChannelSpec(name="velocity"),
                    ChannelSpec(name="pressure"),
                ],
            )
        ],
        license=LicenseInfo(spdx="mit", commercial_use=True),
        compute=ComputeInfo(estimated_weights_gb=0.4, cpu_supported=True),
    ),
    card(
        "flow-nc",
        summary="Larger fluid dynamics surrogate.",
        domains=["fluid_dynamics"],
        tasks=["surrogate_simulation"],
        license=LicenseInfo(spdx="cc-by-nc-4.0", commercial_use=False),
        compute=ComputeInfo(min_gpu_memory_gb=80),
    ),
    card(
        "weather",
        summary="Global weather forecasting at 0.25 degree.",
        domains=["weather"],
        tasks=["forecasting"],
        license=LicenseInfo(spdx="apache-2.0", commercial_use=True),
    ),
    card(
        "molecule",
        summary="Predicts quantum properties of molecules from SMILES.",
        domains=["molecular_chemistry"],
        tasks=["property_prediction"],
        inputs=[IOSpec(name="smi", kind="molecule", representation="SMILES")],
        outputs=[IOSpec(name="gap", units="hartree")],
    ),
    card(
        "plume",
        summary="Emulates volcanic ash plume dispersion for volcanology.",
        domains=["volcanology"],
    ),
]


def test_heuristic_requirements_read_constraints():
    req = heuristic_requirements(
        "I need a commercial-use model for 2D Navier-Stokes surrogate simulation with "
        "velocity and pressure on an RTX 3090, under 500M parameters, with uncertainty estimates."
    )
    assert req.domains == ["fluid_dynamics"]
    assert req.tasks[0] == "surrogate_simulation"
    assert req.gpu_memory_gb == 24.0
    assert req.commercial_use is True
    assert req.max_parameters == 500_000_000
    assert req.n_spatial_dims == 2
    assert req.needs_uncertainty is True
    assert {"velocity", "pressure"} <= {i.name for i in req.inputs}
    assert req.run_requested is False


def test_wanted_quantity_stops_where_the_data_file_starts(tmp_path):
    (tmp_path / "boiling_points.csv").write_text("smiles\nCCO\n")
    req = heuristic_requirements(
        "Predict the boiling points of the molecules in boiling_points.csv.",
        workspace=tmp_path,
    )
    assert [o.name for o in req.outputs] == ["boiling points"]
    assert req.data_paths == ["boiling_points.csv"]


def test_run_intent_and_data_files(tmp_path):
    (tmp_path / "state.nc").write_text("x", encoding="utf-8")
    req = heuristic_requirements(
        "Run the model on state.nc and predict temperature.", workspace=tmp_path
    )
    assert req.run_requested is True
    assert req.data_paths == ["state.nc"]
    assert any(i.format == "NetCDF" for i in req.inputs)
    assert any("temperature" in o.name for o in req.outputs)
    assert (
        heuristic_requirements(
            "Which model should I run for this? Recommend one."
        ).run_requested
        is False
    )


def test_llm_requirements_merge_with_heuristics():
    extracted = RequirementsExtraction(
        summary="Volcanic plume emulation",
        domains=["volcanology", "climate"],
        tasks=["surrogate simulation"],
        cpu_only=True,
    )
    req = merge_requirements(
        heuristic_requirements("Need a model on my laptop with a 16 GB GPU"),
        extracted,
    )
    # Labels outside the vocabulary are kept verbatim.
    assert req.domains[:2] == ["volcanology", "climate"]
    assert req.tasks == ["surrogate_simulation"]
    assert req.cpu_only is True
    assert req.gpu_memory_gb == 16.0

    llm = RecordingChatModel(structured_factory=lambda schema: extracted)
    assert parse_requirements("ash plumes", llm).domains[0] == "volcanology"


def test_rank_by_domain_task_and_io():
    req = heuristic_requirements(
        "Surrogate simulation of 2D fluid flow; I have velocity and pressure fields."
    )
    recs = rank_models(req, CATALOG)
    assert [r.model_id for r in recs[:2]] == ["flow", "flow-nc"]
    top = recs[0]
    assert top.fit == "strong"
    assert any("Fluid dynamics" in s for s in top.strengths)
    assert {c.criterion for c in top.criteria} == {
        "domain",
        "task",
        "inputs",
        "outputs",
        "relevance",
        "practicality",
        "evidence",
    }


def test_constraints_block_incompatible_models():
    req = UserRequirements(
        domains=["fluid_dynamics"],
        tasks=["surrogate_simulation"],
        commercial_use=True,
        gpu_memory_gb=24,
    )
    recs = rank_models(req, CATALOG)
    blocked = next(r for r in recs if r.model_id == "flow-nc")
    assert blocked.fit == "incompatible"
    assert any("commercial" in b for b in blocked.blockers)
    assert any("80 GB" in b for b in blocked.blockers)
    assert recs[-1].fit == "incompatible" or all(
        r.fit != "incompatible" for r in recs[recs.index(blocked) + 1 :]
    )


def test_unknown_domains_still_match_lexically():
    req = UserRequirements(
        summary="volcanic ash plume emulation",
        domains=["volcanology"],
        keywords=["volcanic", "ash", "plume"],
    )
    assert rank_models(req, CATALOG)[0].model_id == "plume"


def test_review_blending_keeps_blockers():
    req = UserRequirements(domains=["fluid_dynamics"], commercial_use=True)
    recs = rank_models(req, CATALOG)
    review = {
        "flow-nc": ("strong", "Excellent fit.", ["Check license"]),
        "weather": ("strong", "Good fit.", []),
    }
    blended = blend_review(recs, review)
    blocked = next(r for r in blended if r.model_id == "flow-nc")
    assert blocked.fit == "incompatible"
    assert "Check license" in blocked.gaps
    assert (
        next(r for r in blended if r.model_id == "weather").rationale
        == "Good fit."
    )


def test_real_layout_cards_rank_end_to_end():
    parsed = [
        extract_model_card(
            CardDocument(
                source=CardSource(kind="file", uri=name),
                text=text,
                name_hint=name,
            )
        )
        for name, text in (
            ("plasma", cards.GENESIS_CARD),
            ("crystal", cards.HF_CARD),
            ("ocean", cards.NVIDIA_CARD),
        )
    ]
    req = heuristic_requirements(
        "Forecast sea surface temperature with a global ocean model at 1 degree."
    )
    assert rank_models(req, parsed)[0].name == "OceanCast"
    req = heuristic_requirements(
        "Predict band gaps of inorganic crystals from CIF files."
    )
    assert rank_models(req, parsed)[0].name == "CrystalProp-S"
    req = heuristic_requirements(
        "Tokamak edge plasma turbulence rollout surrogate."
    )
    assert rank_models(req, parsed)[0].name == "PlasmaEdge-2D"


def test_requirements_separate_data_from_compute_and_read_task_verbs():
    req = heuristic_requirements(
        "Which model should I use to simulate heat conduction along a rod? "
        "I have a temperature profile in kelvin and only a laptop CPU."
    )
    assert req.tasks == ["surrogate_simulation"]
    assert req.cpu_only is True
    assert [i.name for i in req.inputs] == ["a temperature profile in kelvin"]

    req = heuristic_requirements("I have 24 GB of GPU memory and ERA5 fields.")
    assert [i.name for i in req.inputs] == ["ERA5 fields"]
    assert req.gpu_memory_gb == 24.0
    assert heuristic_requirements("I have a GPU cluster.").inputs == []
