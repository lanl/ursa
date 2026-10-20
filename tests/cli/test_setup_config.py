import yaml
from pydantic import SecretStr

from ursa.cli import setup_config as setup
from ursa.util.secrets import SecretReference


def test_setup_config_round_trip_preserves_placeholders(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_URL", "https://models.example/v1")
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company: {base_url: "${MODEL_URL}", api_key: {keyring: true}}
llm_model: {model: gpt-test, inference_provider: company}
mcp_servers:
  tools: {transport: stdio, command: tool, env: {API_KEY: "${MCP_SECRET}"}}
"""
    )

    config, llm_configured = setup.load_setup_config(path)
    assert llm_configured
    assert str(config.inference_providers["company"].base_url) == "${MODEL_URL}"
    setup.save_setup_config(path, config, {})
    assert "${MCP_SECRET}" in path.read_text()

    runtime = setup.prepare_provider_probe(
        config, "company", SecretStr("pending-secret")
    )
    assert str(runtime.base_url) == "https://models.example/v1"
    assert runtime.api_key.get_secret_value() == "pending-secret"
    assert config.inference_providers["company"].api_key == SecretReference(
        keyring=True
    )


def test_setup_defaults_build_a_complete_config(tmp_path):
    config, configured = setup.load_setup_config(tmp_path / "missing.yaml")
    defaults = yaml.safe_load(
        """
inference_providers: [{name: company, base_url: https://models.example/v1}]
llm_model: {model: gpt-test, inference_provider: company, template: hello}
emb_model: {model: embedding-test, inference_provider: company}
mcp_servers:
  tools: {transport: streamable_http, url: https://tools.example/mcp}
"""
    )
    assert setup.apply_setup_defaults(
        config, defaults, llm_configured=configured
    )
    assert config.inference_providers["company"].api_key == SecretReference(
        keyring=True
    )
    assert config.llm_model.model == "gpt-test"
    values = config.llm_model.editable_values()
    assert values["template"] == "hello" and "keyring" not in values
    assert config.emb_model.model == "embedding-test"
    assert set(config.mcp_servers) == {"tools"}
