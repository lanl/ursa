"""Recognition, compatibility, and run planning for models served over MCP.

The cards follow the patterns of Nomad's MIST cards, where the facts a
user needs (what the model predicts, what it was trained on) sit in a code
comment, an "Outputs" bullet, and a fine-tuning paragraph.
"""

from tests.model_cards import cards
from ursa.model_cards.compat import assess_model
from ursa.model_cards.data import heavy_atoms, inspect_data, read_items
from ursa.model_cards.extract import extract_model_card
from ursa.model_cards.patterns import find_units
from ursa.model_cards.running import (
    build_run_plan,
    flatten_result,
    item_argument,
    smiles_in,
)
from ursa.model_cards.schema import DataDescription, UserRequirements
from ursa.model_cards.sources import mcp_tool_document


def served_card():
    return extract_model_card(
        mcp_tool_document(
            name="quantaprop_qm",
            server="nomad",
            description="Predicts a molecular property for one SMILES string.",
            input_schema=cards.SERVED_PROPERTY_SCHEMA,
            card_text=cards.SERVED_PROPERTY_CARD,
            nomad=True,
        )
    )


def test_served_card_states_outputs_scope_and_schema_sources():
    card = served_card()
    (output,) = card.outputs
    assert output.name == "quantum properties"
    assert output.description.startswith("Regression predictions")
    assert [c.name for c in output.channels] == ["homo", "lumo", "gap"]
    # The channel list cites the code comment that names it.
    quote = card.provenance["outputs.quantum properties.channels"].quote
    assert quote.startswith("Returns predictions for 3 quantum properties")
    # "pooled to produce a single embedding vector" describes the encoder.
    assert "embedding" not in card.tasks
    assert card.architecture.base_model == "quanta-base-20M"
    assert card.training_scope.examples == 1200
    (scope,) = card.training_scope.ranges
    assert (scope.quantity, scope.maximum, scope.inclusive) == (
        "heavy atoms",
        9,
        True,
    )
    assert card.provenance["inputs.smi"].source == "mcp_schema"
    conflict = next(i for i in card.issues if i.code == "license_conflict")
    assert "No commercial use" in conflict.message


def test_bracketed_units_and_predicted_quantity_names():
    assert find_units("Regression prediction for boiling point (°C)") == "°C"
    assert find_units("pressure [hPa] at the surface") == "hPa"
    assert find_units("see option (A) below") is None


def test_items_and_sizes_come_from_the_file(tmp_path):
    assert [
        heavy_atoms(s)
        for s in [
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "[nH]1cccc1",
            "[2H]C([2H])([2H])Cl",
        ]
    ] == [14, 5, 2]
    smi = tmp_path / "mols.smi"
    smi.write_text(
        "SMILES\nCCO ethanol\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C caffeine\n"
    )
    profile = inspect_data(smi)
    assert profile.item_sizes == {"heavy atoms": [3, 14]}
    assert read_items(profile) == [
        ("CCO", "ethanol"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
    ]
    table = tmp_path / "props.csv"
    table.write_text(
        "name,smiles,solubility\nethanol,CCO,0.5\nbenzene,c1ccccc1,-1.6\n"
    )
    profile = inspect_data(table)
    assert profile.column == "smiles"
    assert read_items(profile, limit=1) == [("CCO", "ethanol")]


def test_training_range_flags_extrapolation_without_blocking_inference(
    tmp_path,
):
    card = served_card()
    smi = tmp_path / "mols.smi"
    smi.write_text("CCO\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\n", encoding="utf-8")
    wanted = UserRequirements(outputs=[DataDescription(name="HOMO-LUMO gap")])
    result = assess_model(
        card, inspect_data(smi), wanted, card_text=cards.SERVED_PROPERTY_CARD
    )
    assert result.pathway == "inference"
    check = next(c for c in result.checks if c.aspect == "training range")
    assert check.status == "unknown"
    assert check.detail.startswith("1 of 2 molecules lies outside")
    outputs = next(c for c in result.checks if c.aspect == "outputs")
    assert outputs.card_key == "outputs.quantum properties.channels"


def test_unlisted_quantity_fine_tunes_with_the_card_training_size(tmp_path):
    card = served_card()
    table = tmp_path / "sol.csv"
    table.write_text(
        "smiles,solubility\nCCO,0.5\nc1ccccc1,-1.6\n", encoding="utf-8"
    )
    wanted = UserRequirements(outputs=[DataDescription(name="solubility")])
    result = assess_model(card, inspect_data(table), wanted)
    assert result.pathway == "fine_tune"
    notes = " ".join(result.fine_tune_notes)
    assert "1,200 examples" in notes and "quanta-base-20M" in notes


def test_run_plans_one_call_per_molecule_of_the_file(tmp_path):
    card = served_card()
    (interface,) = [i for i in card.interfaces if i.kind == "mcp_tool"]
    assert item_argument(interface) == ("smi", "molecule")
    smi = tmp_path / "mols.smi"
    smi.write_text("CCO ethanol\nc1ccccc1 benzene\nCC(=O)O\n", encoding="utf-8")
    plan = build_run_plan(
        card, "Run it on mols.smi", profiles=[inspect_data(smi)], max_items=2
    )
    assert plan.batch == [{"smi": "CCO"}, {"smi": "c1ccccc1"}]
    assert plan.batch_names == ["ethanol", "benzene"]
    assert any("first 2 items" in w for w in plan.warnings)
    single = build_run_plan(
        card, "Run it on caffeine, smi CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    )
    assert single.arguments == {"smi": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}
    assert not single.missing_arguments and not single.batch


def test_smiles_in_text_and_result_columns():
    assert smiles_in("Use `CCO` please") == "CCO"
    assert smiles_in("Predict the gap of CC(=O)O.") == "CC(=O)O"
    assert smiles_in("Run the top model (e.g. QM9).") is None
    assert smiles_in('Run toy with {"temperature": [300, 310]}') is None
    result = {
        "molecule": "CCO",
        "properties": {
            "bp": {"value": 78.0, "units": "°C", "description": "Boiling point"}
        },
    }
    assert flatten_result(result) == {"molecule": "CCO", "bp [°C]": 78.0}
