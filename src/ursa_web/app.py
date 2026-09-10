"""Chainlit application for the URSA runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, Switch, TextInput
from chainlit.user import User
from langchain_core.messages import AIMessage

from ursa.cli.config import ChatModelConfig, EmbModelConfig, UrsaConfig
from ursa.cli.runtime import HITL
from ursa_web.data import JsonDataLayer


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _provider_options() -> list[str]:
    return ["openai", "anthropic", "google_genai", "ollama"]


def _saved_settings() -> dict[str, Any]:
    configured = Path(os.getenv("URSA_WORKSPACE", ".")).expanduser()
    candidates = [
        configured / ".ursa-data" / "web-settings.json",
        Path.cwd() / ".ursa-data" / "web-settings.json",
        Path.cwd() / ".ursa-web" / ".ursa-data" / "web-settings.json",
        Path(__file__).resolve().parents[2]
        / ".ursa-data"
        / "web-settings.json",
        Path(__file__).resolve().parents[2]
        / ".ursa-web"
        / ".ursa-data"
        / "web-settings.json",
    ]
    for path in dict.fromkeys(candidates):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return {}


def _save_settings(settings: dict[str, Any], workspace: Path) -> None:
    path = workspace / ".ursa-data" / "web-settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n")


@cl.data_layer
def data_layer() -> JsonDataLayer:
    """Persist Chainlit threads without an external database dependency."""
    workspace = _initial_config().workspace
    data_dir = workspace / ".ursa-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return JsonDataLayer(data_dir / "chainlit-threads.json")


@cl.header_auth_callback
async def local_user(_headers) -> User:
    """Identify local dashboard sessions for persisted thread history."""
    return User(identifier="local", display_name="Local user")


class _WebHITL(HITL):
    """HITL adapter that keeps URSA metadata out of the worktree."""

    async def _get_agent(self, name: str):
        agent = await super()._get_agent(name)
        data_dir = self.workspace / ".ursa-data"
        data_dir.mkdir(parents=True, exist_ok=True)
        if agent._agent is not None:
            agent._agent.den = data_dir
            agent._agent.telemetry.output_dir = data_dir / "metrics"
            agent._agent.telemetry.output_dir.mkdir(parents=True, exist_ok=True)
        return agent


def _initial_config() -> UrsaConfig:
    saved = _saved_settings()
    provider = os.getenv("URSA_LLM_PROVIDER", "openai")
    provider = saved.get("llm_provider", provider)
    model = saved.get("llm_model", os.getenv("URSA_LLM_MODEL", "gpt-4o-mini"))
    llm_endpoint = os.getenv("URSA_LLM_ENDPOINT")
    llm_endpoint = saved.get("llm_endpoint", llm_endpoint)
    embedding_model = os.getenv("URSA_EMBEDDING_MODEL")
    embedding_model = saved.get("embedding_model", embedding_model)
    embedding_provider = os.getenv("URSA_EMBEDDING_PROVIDER", "openai")
    embedding_provider = saved.get("embedding_provider", embedding_provider)
    embedding_endpoint = os.getenv("URSA_EMBEDDING_ENDPOINT")
    embedding_endpoint = saved.get("embedding_endpoint", embedding_endpoint)
    llm_api_key_env = saved.get(
        "llm_api_key_env", os.getenv("URSA_LLM_API_KEY_ENV", "OPENAI_API_KEY")
    )
    embedding_api_key_env = saved.get(
        "embedding_api_key_env",
        os.getenv("URSA_EMBEDDING_API_KEY_ENV", "OPENAI_API_KEY"),
    )
    llm_ssl_verify = saved.get(
        "llm_ssl_verify", _env_bool("URSA_LLM_SSL_VERIFY")
    )
    embedding_ssl_verify = saved.get(
        "embedding_ssl_verify", _env_bool("URSA_EMBEDDING_SSL_VERIFY")
    )

    providers: dict[str, dict[str, Any]] = {
        "openai": {
            "base_url": llm_endpoint or "https://api.openai.com/v1",
            "api_key": {"env": llm_api_key_env},
            "ssl_verify": llm_ssl_verify,
        }
    }
    if provider not in providers:
        providers[provider] = {
            "api_key": {"env": llm_api_key_env},
            "ssl_verify": llm_ssl_verify,
        }
    if embedding_model and embedding_provider not in providers:
        providers[embedding_provider] = {
            "api_key": {"env": embedding_api_key_env},
            "ssl_verify": embedding_ssl_verify,
        }
    llm = ChatModelConfig(
        model=model,
        model_provider=provider,
        base_url=llm_endpoint,
        inference_provider=None if llm_endpoint else provider,
        api_key={"env": llm_api_key_env},
        ssl_verify=llm_ssl_verify,
    )
    embedding = None
    if embedding_model:
        embedding = EmbModelConfig(
            model=embedding_model,
            model_provider=embedding_provider,
            base_url=embedding_endpoint,
            inference_provider=None
            if embedding_endpoint
            else embedding_provider,
            api_key={"env": embedding_api_key_env},
            ssl_verify=embedding_ssl_verify,
        )
    return UrsaConfig(
        workspace=Path(
            saved.get("workspace", os.getenv("URSA_WORKSPACE", "."))
        ).expanduser(),
        llm_model=llm,
        emb_model=embedding,
        inference_providers=providers,
    )


def _settings(runtime: HITL | None) -> list[Any]:
    config = runtime.config if runtime else _initial_config()
    llm = config.llm_model
    embedding = config.emb_model
    agent_names = list(runtime.agents) if runtime else ["chat"]
    saved_agent = _saved_settings().get("agent", "chat")
    if saved_agent not in agent_names:
        saved_agent = "chat"
    return [
        TextInput(
            id="workspace",
            label="Workspace directory",
            initial=str(config.workspace),
        ),
        Select(
            id="agent",
            label="URSA agent",
            values=agent_names,
            initial_value=saved_agent,
        ),
        TextInput(id="llm_model", label="LLM model", initial=llm.model),
        Select(
            id="llm_provider",
            label="LLM provider",
            values=_provider_options(),
            initial_value=llm.model_provider or "openai",
        ),
        TextInput(
            id="llm_endpoint",
            label="LLM endpoint (blank uses provider)",
            initial=llm.base_url or "",
        ),
        TextInput(
            id="llm_api_key_env",
            label="LLM API key environment variable",
            initial=os.getenv("URSA_LLM_API_KEY_ENV", "OPENAI_API_KEY"),
        ),
        Switch(
            id="llm_ssl_verify",
            label="Verify LLM TLS certificates",
            initial=llm.ssl_verify,
        ),
        TextInput(
            id="embedding_model",
            label="Embedding model (blank disables embeddings)",
            initial=embedding.model if embedding else "",
        ),
        Select(
            id="embedding_provider",
            label="Embedding provider",
            values=_provider_options(),
            initial_value=(embedding.model_provider if embedding else "openai")
            or "openai",
        ),
        TextInput(
            id="embedding_endpoint",
            label="Embedding endpoint (blank uses provider)",
            initial=embedding.base_url if embedding else "",
        ),
        TextInput(
            id="embedding_api_key_env",
            label="Embedding API key environment variable",
            initial=os.getenv("URSA_EMBEDDING_API_KEY_ENV", "OPENAI_API_KEY"),
        ),
        Switch(
            id="embedding_ssl_verify",
            label="Verify embedding TLS certificates",
            initial=embedding.ssl_verify if embedding else True,
        ),
    ]


@cl.on_chat_start
async def start() -> None:
    runtime = _WebHITL(_initial_config())
    cl.user_session.set("runtime", runtime)
    saved_agent = _saved_settings().get("agent", "chat")
    cl.user_session.set(
        "agent", saved_agent if saved_agent in runtime.agents else "chat"
    )
    await cl.ChatSettings(_settings(runtime)).send()
    await cl.Message(
        content=(
            "URSA is ready. Use **Settings** to change model endpoints or "
            "disable TLS certificate verification."
        )
    ).send()


@cl.on_chat_resume
async def resume(thread: dict[str, Any]) -> None:
    """Recreate the runtime when a persisted conversation is selected."""
    runtime = _WebHITL(_initial_config())
    cl.user_session.set("runtime", runtime)
    saved_agent = _saved_settings().get("agent", "chat")
    cl.user_session.set(
        "agent", saved_agent if saved_agent in runtime.agents else "chat"
    )
    if cl.user_session.get("agent") != "chat":
        return

    async with runtime.use_agent("chat") as wrapper:
        agent = wrapper._agent
        if agent is None:
            return
        state = None
        for step in thread.get("steps", []):
            content = str(step.get("output") or step.get("input") or "")
            if not content:
                continue
            if step.get("type") == "user_message":
                state = agent.format_query(content, state)
            elif step.get("type") == "assistant_message" and state is not None:
                state["messages"].append(AIMessage(content=content))
        wrapper.state = state


@cl.on_settings_update
async def settings_update(settings: dict[str, Any]) -> None:
    runtime = cl.user_session.get("runtime")
    if not isinstance(runtime, HITL):
        return
    cl.user_session.set("agent", str(settings.get("agent", "chat")))
    llm_endpoint = str(settings.get("llm_endpoint", "")).strip() or None
    embedding_endpoint = (
        str(settings.get("embedding_endpoint", "")).strip() or None
    )
    llm = ChatModelConfig(
        model=str(settings["llm_model"]).strip(),
        model_provider=str(settings["llm_provider"]),
        base_url=llm_endpoint,
        inference_provider=None
        if llm_endpoint
        else str(settings["llm_provider"]),
        api_key={
            "env": str(settings.get("llm_api_key_env", "")).strip()
            or "OPENAI_API_KEY"
        },
        ssl_verify=bool(settings["llm_ssl_verify"]),
    )
    embedding_name = str(settings.get("embedding_model", "")).strip()
    embedding = None
    if embedding_name:
        embedding_provider = str(settings["embedding_provider"])
        embedding = EmbModelConfig(
            model=embedding_name,
            model_provider=embedding_provider,
            base_url=embedding_endpoint,
            inference_provider=None
            if embedding_endpoint
            else embedding_provider,
            api_key={
                "env": str(settings.get("embedding_api_key_env", "")).strip()
                or "OPENAI_API_KEY"
            },
            ssl_verify=bool(settings["embedding_ssl_verify"]),
        )
    workspace = Path(
        str(settings.get("workspace", runtime.workspace))
    ).expanduser()
    agent_name = str(settings.get("agent", "chat"))
    try:
        await runtime.reconfigure_models(llm, embedding)
        workspace.mkdir(parents=True, exist_ok=True)
        runtime.workspace = workspace
        runtime.config.workspace = workspace
        _save_settings(
            {
                "workspace": str(workspace),
                "agent": agent_name,
                "llm_model": llm.model,
                "llm_provider": llm.model_provider,
                "llm_endpoint": llm_endpoint,
                "llm_api_key_env": str(settings.get("llm_api_key_env", ""))
                or "OPENAI_API_KEY",
                "llm_ssl_verify": llm.ssl_verify,
                "embedding_model": embedding.model if embedding else "",
                "embedding_provider": (
                    embedding.model_provider if embedding else "openai"
                ),
                "embedding_endpoint": embedding_endpoint,
                "embedding_api_key_env": str(
                    settings.get("embedding_api_key_env", "")
                )
                or "OPENAI_API_KEY",
                "embedding_ssl_verify": (
                    embedding.ssl_verify if embedding else True
                ),
            },
            workspace,
        )
    except (RuntimeError, ValueError) as exc:
        await cl.Message(content=f"Model update failed: `{exc}`").send()
        return
    await cl.Message(content="Model configuration updated.").send()


@cl.on_message
async def message(message: cl.Message) -> None:
    runtime = cl.user_session.get("runtime")
    if not isinstance(runtime, HITL):
        await cl.Message(content="The URSA runtime is not initialized.").send()
        return
    try:
        agent_name = cl.user_session.get("agent", "chat")
        result = await runtime.run_agent(agent_name, message.content)
    except (RuntimeError, ValueError) as exc:
        await cl.Message(content=f"URSA failed: `{exc}`").send()
        return
    await cl.Message(content=result).send()


@cl.on_chat_end
async def end() -> None:
    runtime = cl.user_session.get("runtime")
    if isinstance(runtime, HITL):
        await runtime.aclose()
