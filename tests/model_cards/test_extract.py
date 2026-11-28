import yaml

from ursa.model_cards.extract import CardDocument, extract_model_card
from ursa.model_cards.schema import CardSource

from . import cards


def doc(
    text="",
    *,
    repo=None,
    config=None,
    hub=None,
    mcp=None,
    native=None,
    name=None,
):
    kind = "huggingface" if repo else "mcp" if mcp else "file"
    return CardDocument(
        source=CardSource(
            kind=kind, uri=f"hf://{repo}" if repo else "test://card"
        ),
        text=text,
        config=config,
        hub_info=hub,
        mcp_tool=mcp,
        native=native,
        repo_id=repo,
        name_hint=name,
    )


def spec(specs, name):
    return next(s for s in specs if s.name == name)


def test_genesis_card_metadata_and_formats():
    card = extract_model_card(
        doc(cards.GENESIS_CARD, repo="example-org/PlasmaEdge-2D")
    )

    assert card.source.dialect == "genesis"
    assert card.name == "PlasmaEdge-2D"
    assert (
        card.summary == "U-Net surrogate for 2D edge-plasma turbulence rollout."
    )
    assert card.risk_level == "general"
    assert card.projects == ["genesis", "demo"]
    assert {"plasma_physics", "pde_general"} <= set(card.domains)
    assert "surrogate_simulation" in card.tasks
    assert (
        card.license.spdx == "apache-2.0"
        and card.license.commercial_use is True
    )

    # Three named fields stacked channel-first in one array.
    (fields,) = card.inputs
    assert fields.kind == "field"
    assert fields.shape == ["3", "256", "512"]
    assert fields.dtype == "float32"
    assert fields.representation == "NumPy array"
    assert [c.name for c in fields.channels] == [
        "density",
        "electron_temperature",
        "potential",
    ]
    assert "zero mean and unit variance" in fields.normalization
    (outputs,) = card.outputs
    assert [c.name for c in outputs.channels] == [
        "density",
        "electron_temperature",
        "potential",
    ]
    assert outputs.shape == ["3", "256", "512"]
    assert card.temporal.time_step == "10 microseconds"
    assert card.compute.min_gpu_memory_gb == 12.0
    assert (
        card.compute.training_hardware.startswith("Inference requires") is False
    )
    assert card.out_of_scope == ["Operational plasma control decisions."]
    assert "Trained only on L-mode discharges." in card.limitations

    interface = next(i for i in card.interfaces if i.kind == "huggingface")
    assert interface.loader == "transformers.AutoModel"
    assert interface.trust_remote_code is True
    assert "remote_code" in card.issue_codes()


def test_hf_template_card_uses_model_index_and_key_values():
    card = extract_model_card(doc(cards.HF_CARD, repo="example/crystalprop-s"))

    assert card.source.dialect == "huggingface"
    assert "CrystalProp-S" in card.aliases
    assert card.developers == ["Example Materials Lab"]
    assert card.architecture.family == "graph neural network"
    assert card.architecture.parameter_count == 4_200_000
    assert card.compute.cpu_supported is True
    assert card.evaluation[0].metric == "Band gap MAE"
    assert card.evaluation[0].value == "0.21"
    assert card.training_data[0].name == "example/mp-2024"
    structure = card.inputs[0]
    assert structure.kind == "structure"
    assert "CIF" in structure.file_formats
    assert card.outputs[0].units == "eV"
    assert card.interfaces[0].loader == "crystalprop.CrystalProp"
    assert "materials_atomistic" in card.domains
    assert card.provenance["architecture.parameter_count"].quote.startswith(
        "The model has 4.2M parameters"
    )


def test_nvidia_card_reads_wrapped_shapes_channels_and_time():
    card = extract_model_card(doc(cards.NVIDIA_CARD, name="oceancast"))

    assert card.source.dialect == "nvidia"
    assert card.name == "OceanCast"
    assert card.architecture.parameter_count == 120_500_000
    assert card.domains[0] == "ocean"
    assert "forecasting" in card.tasks
    inputs = card.inputs[0]
    assert inputs.kind == "tensor" and inputs.representation == "PyTorch tensor"
    assert inputs.shape == ["batch", "variable", "latitude", "longitude"]
    assert [c.name for c in inputs.channels] == ["sst", "sss", "ssh", "mld"]
    assert [c.name for c in card.outputs[0].channels] == [
        "sst",
        "sss",
        "ssh",
        "mld",
    ]
    assert card.spatial.grid == "latitude-longitude"
    assert set(card.spatial.resolution) == {"1 deg", "180 x 360"}
    assert card.temporal.time_step == "5 days"
    assert card.temporal.horizon == "365 days"


def test_freeform_card_combines_text_and_config():
    card = extract_model_card(
        doc(
            cards.FREEFORM_CARD,
            repo="example/Vortex-B",
            config=cards.FREEFORM_CONFIG,
        )
    )

    fields = card.inputs[0]
    assert [c.name for c in fields.channels] == ["vorticity", "velocity"]
    assert fields.kind == "field"
    assert card.spatial.resolution[0] == "64 x 64"
    assert card.architecture.parameter_count == 85_000_000
    assert card.architecture.name == "VortexForPDE"
    assert card.license.commercial_use is False
    assert "pde_general" in card.domains
    assert "minimal_card" in card.issue_codes()
    assert not card.completeness.run_ready


def test_config_only_model_is_flagged_and_still_described():
    card = extract_model_card(
        doc("", repo="example/glitch-net", config=cards.CONFIG_ONLY)
    )

    assert card.source.dialect == "config_only"
    assert {"card_missing", "license_missing"} <= card.issue_codes()
    assert card.temporal.sampling_rate == "4096 Hz"
    assert card.temporal.window == "8192 samples"
    assert card.architecture.parameter_count == 812_345
    output = card.outputs[0]
    assert output.semantics == "class labels"
    assert [c.name for c in output.channels] == [
        "blip",
        "koi fish",
        "scattered light",
    ]
    assert "classification" in card.tasks


def test_gated_repository_without_card_text():
    hub = {"gated": "manual", "tags": ["license:other"], "siblings": []}
    card = extract_model_card(doc("", repo="example/gated", hub=hub))

    assert card.gated is True
    assert {"card_inaccessible", "gated_access"} <= card.issue_codes()


def test_mcp_tool_schema_defines_runtime_formats():
    molecule = extract_model_card(
        doc(cards.MOLECULE_TOOL_CARD, mcp=cards.MOLECULE_TOOL)
    )
    assert molecule.id == "mcp:nomad:molprop-qm"
    assert molecule.inputs[0].name == "smi"
    assert molecule.inputs[0].kind == "molecule"
    assert molecule.inputs[0].representation == "SMILES"
    assert molecule.interfaces[0].kind == "mcp_tool"
    assert molecule.interfaces[0].input_schema["required"] == ["smi"]
    assert molecule.outputs[0].units == "hartree"
    assert molecule.completeness.run_ready

    rollout = extract_model_card(doc("", mcp=cards.ROLLOUT_TOOL))
    state = spec(rollout.inputs, "initial_state")
    assert state.kind == "field" and state.representation == "WellFormat"
    assert spec(rollout.inputs, "duration").kind == "scalar"
    assert rollout.temporal.time_step == "0.05 s"
    assert rollout.source.dialect == "mcp_only"


def test_native_yaml_card_is_validated_and_normalized():
    native = yaml.safe_load(cards.NATIVE_CARD_YAML)
    card = extract_model_card(doc(native=native))

    assert card.source.dialect == "ursa"
    assert card.architecture.parameter_count == 3_100_000
    assert card.license.commercial_use is True
    assert spec(card.outputs, "forces").shape == ["n_atoms", "3"]
    assert spec(card.outputs, "forces").units == "eV/Å"
    assert card.interfaces[0].entry_point == "toymlip:load"
    assert card.provenance["inputs"].method == "declared"


def test_unfilled_template_text_is_not_extracted():
    card = extract_model_card(doc(cards.TEMPLATE_RESIDUE_CARD, name="template"))

    assert "template_residue" in card.issue_codes()
    assert card.inputs == [] and card.outputs == []
    assert card.license.spdx is None
    assert all("224" not in r for r in card.spatial.resolution)
    # BibTeX double braces are not template placeholders.
    assert (
        card.citations[0].title
        == "A Real Paper: Bibtex Braces Are Not Template Text"
    )


def test_family_table_counts_only_this_checkpoint():
    card = extract_model_card(
        doc(cards.FAMILY_TABLE_CARD, repo="example/Terra-EO-300M")
    )

    assert card.architecture.parameter_count == 300_000_000
    assert "conflicting_values" not in card.issue_codes()


def test_shared_metrics_table_uses_matching_row():
    card = extract_model_card(
        doc(cards.SHARED_METRICS_CARD, repo="example/FNO-rayleigh_benard")
    )

    assert [(m.metric, m.value) for m in card.evaluation] == [
        ("VRMSE", "0.6049")
    ]
    assert "shared_card" in card.issue_codes()


def test_license_text_overrides_permissive_front_matter():
    card = extract_model_card(
        doc(cards.LICENSE_CONFLICT_CARD, repo="example/ChemNet")
    )

    assert card.license.spdx == "apache-2.0"
    assert card.license.commercial_use is False
    assert "license_conflict" in card.issue_codes()
    assert any("No commercial use" in r for r in card.license.restrictions)


def test_conflicting_parameter_counts_are_reported():
    hub = {
        "safetensors": {"total": 50_000_000, "parameters": {"F32": 50_000_000}}
    }
    text = (
        "---\nlicense: mit\n---\n# Model\n\n- **Number of parameters:** 10M\n"
    )
    card = extract_model_card(doc(text, repo="example/model", hub=hub))

    assert card.architecture.parameter_count == 50_000_000
    assert "conflicting_values" in card.issue_codes()
    assert (
        card.provenance["architecture.parameter_count"].source
        == "hub_api.safetensors"
    )


def test_hub_inferred_pipeline_tag_is_weak_task_evidence():
    hub = {
        "pipeline_tag": "feature-extraction",
        "tags": ["transformers", "feature-extraction"],
    }
    text = "---\nlicense: mit\n---\n# Model\n\nA surrogate model for heat transfer simulations on 2D domains.\n"
    card = extract_model_card(doc(text, repo="example/heat", hub=hub))

    assert "embedding" not in card.tasks
    assert card.tasks == ["surrogate_simulation"]
    assert card.domains == ["heat_transfer"]


def test_statement_head_names_the_variable_and_units_are_read():
    text = (
        "# Heat\n\n## Inputs and outputs\n\n"
        "- Input `temperature`: temperature samples in kelvin on a uniform grid.\n"
        "- Output `temperature`: the profile after `time_s`, in kelvin.\n"
    )
    card = extract_model_card(doc(text, name="heat"))

    assert [s.name for s in card.outputs] == ["temperature"]
    assert card.outputs[0].units == "K"
    assert spec(card.inputs, "temperature").units == "K"


def test_tool_schema_descriptions_give_units():
    tool = {
        "name": "heat-1d",
        "server": "toy",
        "description": "Advance a 1D temperature profile.",
        "input_schema": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Temperature in kelvin.",
                },
                "diffusivity": {
                    "type": "number",
                    "description": "Diffusivity in m^2/s.",
                },
                "time": {
                    "type": "number",
                    "description": "Elapsed time in seconds.",
                },
            },
        },
    }
    card = extract_model_card(doc("", mcp=tool))

    assert {s.name: s.units for s in card.inputs} == {
        "temperature": "K",
        "diffusivity": "m^2/s",
        "time": "s",
    }


def test_unit_columns_fill_channel_units():
    text = (
        "# Weather model\n\n## Inputs and outputs\n\n"
        "| Variable | Units | Input/Output |\n|---|---|---|\n"
        "| t2m | \\\\(K\\\\) | Both |\n| tp | m | Output |\n\n"
        "## Variables\n\n| Short name | Name | Units |\n|---|---|---|\n"
        "| msl | Mean sea-level pressure | $\\mathrm{Pa}$ |\n\n"
        "## Outputs\n\nThe model predicts `msl`.\n"
    )
    card = extract_model_card(doc(text, name="weather"))

    channels = {c.name: c.units for s in card.inputs for c in s.channels}
    assert channels == {"t2m": "K"}
    out = {c.name: c.units for s in card.outputs for c in s.channels}
    assert out == {"t2m": "K", "tp": "m"}
    assert spec(card.outputs, "msl").units == "Pa"
