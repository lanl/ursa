from ursa.model_cards.compat import assess_model, assess_models, card_line
from ursa.model_cards.data import DataProfile, VariableProfile
from ursa.model_cards.schema import (
    ArchitectureInfo,
    CardSource,
    ChannelSpec,
    DataDescription,
    Interface,
    IOSpec,
    ModelCard,
    SpatialSpec,
    TemporalSpec,
    UserRequirements,
)


def card(name, **fields):
    return ModelCard(
        id=f"test:{name}",
        name=name,
        source=CardSource(kind="file", uri=f"test://{name}"),
        **fields,
    )


MOLPROP = card(
    "molprop",
    domains=["molecular_chemistry"],
    tasks=["property_prediction"],
    inputs=[IOSpec(name="smi", kind="molecule", representation="SMILES")],
    outputs=[
        IOSpec(
            name="properties",
            description="Regression predictions for 12 quantum mechanical properties",
        )
    ],
    architecture=ArchitectureInfo(
        parameter_count=26_900_000, base_model="mist-28M"
    ),
    interfaces=[Interface(kind="mcp_tool", name="molprop", server="nomad")],
)
MOLPROP_TEXT = (
    "## Getting Started\n"
    "Gap-filling methods are described in the paper.\n"
    "# Returns predictions for 12 quantum mechanical properties (HOMO, LUMO, gap, r2)\n"
)
WEATHER = card(
    "wx",
    domains=["weather"],
    tasks=["forecasting"],
    inputs=[
        IOSpec(
            name="state",
            kind="tensor",
            channels=[
                ChannelSpec(name="t2m", units="K"),
                ChannelSpec(name="u10m"),
                ChannelSpec(name="z500"),
            ],
        )
    ],
    spatial=SpatialSpec(n_dims=2, resolution=["0.25 deg", "721 x 1440"]),
    temporal=TemporalSpec(time_step="6 hours"),
)


def molecules(labels=()):
    return DataProfile(
        path="/data/mols.csv",
        format="CSV",
        reader="pandas",
        kind="molecule",
        representation="SMILES",
        n_samples=3,
        labels=list(labels),
    )


def grid(names, *, units=None, levels=(), dims=2):
    return DataProfile(
        path="/data/era.nc",
        format="NetCDF3",
        reader="scipy",
        kind="field",
        spatial_dims=dims,
        grid=[181, 360],
        resolution="1 deg",
        time_steps=4,
        time_step="6 hours",
        variables=[
            VariableProfile(
                name=name,
                units=(units or {}).get(name),
                levels=list(levels) if name == "z" else [],
            )
            for name in names
        ],
    )


def wants(*quantities):
    return UserRequirements(
        outputs=[DataDescription(name=q) for q in quantities]
    )


def test_molecules_run_as_is_when_the_card_states_the_quantity():
    result = assess_model(
        MOLPROP,
        molecules(),
        wants("the HOMO-LUMO gap of these molecules"),
        card_text=MOLPROP_TEXT,
    )
    assert result.pathway == "inference"
    outputs = next(c for c in result.checks if c.aspect == "outputs")
    # The quote is the output statement, not the first line that says "gap".
    assert outputs.quote.startswith("Returns predictions")


def test_an_unlisted_property_needs_fine_tuning_with_targets():
    result = assess_model(
        MOLPROP,
        molecules(["solubility"]),
        wants("solubility"),
        card_text=MOLPROP_TEXT,
    )
    assert result.pathway == "fine_tune"
    notes = " ".join(result.fine_tune_notes)
    assert "solubility" in notes and "mist-28M" in notes
    assert "only as an inference tool" in notes and "0.4 GB" in notes


def test_gridded_data_converts_grid_and_units():
    profile = grid(
        ["t2m", "u10", "z"], units={"t2m": "degC"}, levels=[500.0, 850.0]
    )
    result = assess_model(WEATHER, profile)
    status = {c.aspect: c.status for c in result.checks}
    assert status["variables"] == "ok"  # u10 = u10m, z at 500 hPa = z500
    assert status["grid"] == "convert" and status["units"] == "convert"
    assert status["time step"] == "ok"
    assert result.pathway == "inference_after_conversion"
    assert len(result.conversions) == 2


def test_missing_variables_suggest_supplying_them():
    result = assess_model(WEATHER, grid(["t2m"]))
    assert result.pathway == "fine_tune"
    assert (
        "2 of 3" in result.summary
        and "Supply the missing variables" in result.summary
    )


def test_wrong_data_kind_or_dimensions_are_not_suitable():
    assert assess_model(MOLPROP, grid(["t2m"])).pathway == "not_suitable"
    profile = grid(["t2m", "u10", "z"], levels=[500.0], dims=3)
    assert assess_model(WEATHER, profile).pathway == "not_suitable"


def test_embedding_models_need_a_trained_head():
    encoder = card(
        "encoder",
        domains=["biomolecular"],
        tasks=["embedding"],
        inputs=[
            IOSpec(
                name="sequence",
                kind="sequence",
                representation="protein sequence",
            )
        ],
    )
    proteins = DataProfile(
        path="/data/p.fasta",
        format="FASTA",
        reader="text",
        kind="sequence",
        representation="protein sequence",
    )
    result = assess_model(encoder, proteins, wants("enzyme class"))
    assert result.pathway == "fine_tune"
    assert any(
        c.aspect == "task" and c.status == "mismatch" for c in result.checks
    )


def test_unstated_inputs_leave_the_decision_open():
    result = assess_model(card("bare"), molecules())
    assert result.pathway == "unknown"


def test_better_pathways_come_first():
    ordered = assess_models([WEATHER, MOLPROP], molecules())
    assert [a.name for a in ordered] == ["molprop", "wx"]
    assert ordered[1].pathway == "not_suitable"


def test_card_line_prefers_statements_about_outputs():
    assert card_line(MOLPROP_TEXT, {"gap"}).startswith("Returns predictions")
    assert card_line(MOLPROP_TEXT, {"solubility"}) is None
