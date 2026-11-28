import json

import httpx
import pytest

from tests.agents.utils import RecordingChatModel
from ursa.model_cards.catalog import ModelCatalog
from ursa.model_cards.extract import CardDocument, extract_model_card
from ursa.model_cards.running import (
    build_run_plan,
    choose_interface,
    execution_brief,
    fill_arguments,
    parse_json_arguments,
    save_tool_result,
    unique_path,
    validate_arguments,
)
from ursa.model_cards.schema import CardSource, Interface
from ursa.model_cards.sources import (
    fetch_hf_document,
    find_card_sources,
    is_card_source,
    load_document,
    parse_source,
    read_local_document,
)

from . import cards


def extract(text="", **kwargs):
    return extract_model_card(
        CardDocument(
            source=CardSource(kind="file", uri=kwargs.pop("uri", "test://x")),
            text=text,
            **kwargs,
        )
    )


# Sources ---------------------------------------------------------------------


def test_parse_source_kinds(tmp_path):
    (tmp_path / "card.md").write_text("# Card", encoding="utf-8")
    assert parse_source("hf://org/model@v1").label == "hf://org/model@v1"
    assert (
        parse_source(
            "https://huggingface.co/org/model/blob/main/README.md"
        ).value
        == "org/model"
    )
    assert parse_source("https://example.org/card.yaml").kind == "url"
    assert parse_source("card.md", tmp_path).kind == "file"
    assert parse_source("org/model").kind == "huggingface"
    assert parse_source("") is None


def test_find_card_sources_ignores_slashed_phrases(tmp_path):
    (tmp_path / "cards").mkdir()
    (tmp_path / "cards" / "heat.md").write_text("# Heat", encoding="utf-8")
    text = (
        "Compare hf://org/flow-net with `camlab/Poseidon-T`, the card at "
        "https://example.org/models/card.md, and cards/heat.md. "
        "Its input/output format is 2D; see https://example.org/paper."
    )
    found = find_card_sources(text, tmp_path)
    assert found == [
        "hf://org/flow-net",
        "https://example.org/models/card.md",
        "hf://camlab/Poseidon-T",
        str((tmp_path / "cards" / "heat.md").resolve()),
    ]


def test_find_card_sources_skips_data_files(tmp_path):
    (tmp_path / "molecules.smi").write_text("CCO\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "boiling.csv").write_text("smiles\nCCO\n", "utf-8")
    (tmp_path / "cards").mkdir()
    (tmp_path / "cards" / "heat.md").write_text("# Heat", encoding="utf-8")
    (tmp_path / "models" / "vortex").mkdir(parents=True)
    text = (
        "I have `molecules.smi` and data/boiling.csv; the card is "
        "`cards/heat.md`, and the model folder is `models/vortex`."
    )
    assert find_card_sources(text, tmp_path) == [
        str((tmp_path / "cards" / "heat.md").resolve()),
        str((tmp_path / "models" / "vortex").resolve()),
    ]
    assert not is_card_source(parse_source("molecules.smi", tmp_path))
    assert is_card_source(parse_source("hf://org/model"))
    assert is_card_source(parse_source("https://example.org/card"))


def test_read_local_directory_and_native_cards(tmp_path):
    model_dir = tmp_path / "vortex"
    model_dir.mkdir()
    (model_dir / "README.md").write_text(cards.FREEFORM_CARD, encoding="utf-8")
    (model_dir / "config.json").write_text(
        json.dumps(cards.FREEFORM_CONFIG), encoding="utf-8"
    )
    doc = read_local_document(model_dir)
    assert doc.text.startswith("---") and doc.config["num_channels"] == 3
    assert doc.source.files == ["README.md", "config.json"]
    assert doc.name_hint == "vortex"

    native = tmp_path / "toymlip.yaml"
    native.write_text(cards.NATIVE_CARD_YAML, encoding="utf-8")
    card = extract_model_card(load_document(str(native)))
    assert card.name == "ToyMLIP" and card.source.dialect == "ursa"

    with pytest.raises(FileNotFoundError):
        read_local_document(tmp_path / "empty")


def mock_hub(routes):
    def handler(request: httpx.Request) -> httpx.Response:
        for path, (status, body) in routes.items():
            if request.url.path == path:
                if isinstance(body, (dict, list)):
                    return httpx.Response(status, json=body)
                return httpx.Response(status, text=body)
        return httpx.Response(404, text="Entry not found")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_hf_document_reads_card_config_and_metadata():
    client = mock_hub({
        "/api/models/example/Vortex-B": (
            200,
            {
                "id": "example/Vortex-B",
                "siblings": [
                    {"rfilename": "README.md"},
                    {"rfilename": "config.json"},
                ],
            },
        ),
        "/example/Vortex-B/raw/main/README.md": (200, cards.FREEFORM_CARD),
        "/example/Vortex-B/raw/main/config.json": (200, cards.FREEFORM_CONFIG),
    })
    doc = fetch_hf_document("example/Vortex-B", client=client)
    assert doc.repo_id == "example/Vortex-B"
    assert doc.config == cards.FREEFORM_CONFIG
    assert doc.source.files == ["README.md", "config.json", "hub_api"]
    card = extract_model_card(doc)
    assert card.id == "hf:example/Vortex-B" and card.name == "Vortex-B"


def test_fetch_hf_document_reports_gated_and_missing_repos():
    gated = mock_hub({
        "/api/models/org/gated": (
            200,
            {"id": "org/gated", "gated": "manual", "siblings": []},
        ),
        "/org/gated/raw/main/README.md": (
            401,
            "Access to model org/gated is restricted.",
        ),
    })
    doc = fetch_hf_document("org/gated", client=gated)
    assert doc.text == "" and "gated" in doc.access_error
    assert "card_inaccessible" in extract_model_card(doc).issue_codes()

    with pytest.raises(FileNotFoundError):
        fetch_hf_document(
            "org/missing",
            client=mock_hub({"/api/models/org/missing": (404, "nope")}),
        )


# Catalog ---------------------------------------------------------------------


def test_catalog_persists_and_resolves_names(tmp_path):
    catalog = ModelCatalog(tmp_path / "catalog")
    ocean = extract(
        cards.NVIDIA_CARD, uri="test://ocean", id_hint="hf:example/oceancast"
    )
    crystal = extract(cards.HF_CARD, uri="test://crystal")
    catalog.add_all([ocean, crystal])

    reloaded = ModelCatalog(tmp_path / "catalog")
    assert len(reloaded) == 2
    assert reloaded.get("hf:example/oceancast").name == "OceanCast"
    assert reloaded.get("example/oceancast").name == "OceanCast"
    assert reloaded.get("oceancast").name == "OceanCast"
    assert reloaded.get("CrystalProp-S").id == crystal.id
    assert reloaded.get("unknown-model") is None
    assert (
        reloaded.search("ocean sea surface forecast")[0][0].name == "OceanCast"
    )
    assert (
        reloaded.remove("OceanCast")
        and len(ModelCatalog(tmp_path / "catalog")) == 1
    )


def test_catalog_skips_corrupt_files(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert len(ModelCatalog(tmp_path)) == 0


# Running ---------------------------------------------------------------------


def test_interface_choice_and_argument_validation():
    mcp_card = extract(cards.MOLECULE_TOOL_CARD, mcp_tool=cards.MOLECULE_TOOL)
    interface = choose_interface(mcp_card)
    assert interface.kind == "mcp_tool"
    assert validate_arguments(interface.input_schema, {"smi": "CCO"}) == []
    assert validate_arguments(interface.input_schema, {}) == [
        "'smi' is a required property"
    ]
    assert validate_arguments(interface.input_schema, {"smi": 3}) == [
        "3 is not of type 'string'"
    ]


def test_fill_arguments_prefers_explicit_json_and_never_invents():
    interface = Interface(
        kind="mcp_tool",
        name="molprop-qm",
        input_schema=cards.MOLECULE_TOOL["input_schema"],
    )
    args, missing, errors = fill_arguments(
        interface, 'Run it with {"smi": "CCO"} please'
    )
    assert (args, missing, errors) == ({"smi": "CCO"}, [], [])

    args, missing, errors = fill_arguments(interface, "Run it on caffeine")
    assert args == {} and missing == ["smi"]

    llm = RecordingChatModel(
        structured_factory=lambda schema: schema(
            arguments_json='{"smi": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}'
        )
    )
    args, missing, errors = fill_arguments(interface, "Run it on caffeine", llm)
    assert args["smi"].startswith("CN1") and not missing and not errors


def test_parse_json_arguments():
    assert parse_json_arguments('use {"a": 1, "b": {"c": 2}} now') == {
        "a": 1,
        "b": {"c": 2},
    }
    assert parse_json_arguments("no json") is None


def test_run_plans_for_each_interface(tmp_path):
    mcp_card = extract(cards.MOLECULE_TOOL_CARD, mcp_tool=cards.MOLECULE_TOOL)
    plan = build_run_plan(mcp_card, 'Predict for {"smi": "CCO"}')
    assert plan.mode == "mcp_tool" and plan.arguments == {"smi": "CCO"}
    path = save_tool_result('{"gap": 0.25}', plan, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["result"] == {
        "gap": 0.25
    }

    # A served model runs on its server, so a card's remote-code note is
    # no warning for an MCP call.
    served = extract(
        cards.MOLECULE_TOOL_CARD
        + "\n\n```python\nAutoModel.from_pretrained('org/m', "
        "trust_remote_code=True)\n```\n",
        mcp_tool=cards.MOLECULE_TOOL,
    )
    assert any(issue.code == "remote_code" for issue in served.issues)
    plan = build_run_plan(served, 'Predict for {"smi": "CCO"}')
    assert plan.mode == "mcp_tool"
    assert not any("trust_remote_code" in w for w in plan.warnings)

    rollout = extract("", mcp_tool=cards.ROLLOUT_TOOL)
    assert (
        build_run_plan(rollout, "roll out 10 steps").mode == "execution_agent"
    )

    genesis = extract(cards.GENESIS_CARD, repo_id="example-org/PlasmaEdge-2D")
    plan = build_run_plan(genesis, "Roll out my plasma state for 20 steps")
    assert plan.mode == "execution_agent"
    assert plan.interface.loader == "transformers.AutoModel"
    brief = execution_brief(
        genesis, plan, "Roll out my plasma state for 20 steps"
    )
    assert "shape=(3, 256, 512)" in brief
    assert (
        "channels (in order): density, electron_temperature, potential" in brief
    )
    assert "zero mean and unit variance" in brief
    assert "trust_remote_code=True" in brief
    assert "Time step matches the trained step (10 microseconds)." in brief

    assert (
        build_run_plan(extract("# Nothing\n\nA model."), "run").mode
        == "unavailable"
    )


def test_unique_path_never_overwrites(tmp_path):
    first = unique_path(tmp_path / "runs", "result", ".json")
    first.write_text("{}", encoding="utf-8")
    second = unique_path(tmp_path / "runs", "result", ".json")

    assert second != first and not second.exists()
    assert second.name.startswith("result-") and second.suffix == ".json"
