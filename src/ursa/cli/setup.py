"""Guided first-run configuration for URSA."""

from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import SecretStr
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    ContentSwitcher,
    Footer,
    Static,
    TextArea,
)

from ursa.cli.config import ChatModelConfig, InferenceProviderConfig
from ursa.cli.doctor import (
    CheckStatus,
    build_report_from_path,
    render_report,
    report_counts,
)
from ursa.cli.setup_config import (
    SetupError,
    apply_setup_defaults,
    format_setup_config_yaml,
    load_setup_config,
    load_setup_defaults,
    prepare_provider_probe,
    replace_inference_providers,
    replace_mcp_servers,
    replace_models,
    save_setup_config,
)
from ursa.cli.setup_entries import DynamicEntryTabs, EntryCandidate
from ursa.cli.setup_models import SetupModelEditor, SetupModelSelection
from ursa.util.crossplatform import default_user_config_path
from ursa.util.health import (
    probe_chat_config,
    probe_embedding_model,
    probe_inference_provider,
    probe_mcp_server,
    run_probe_with_deadline,
)
from ursa.util.http import inject_truststore_into_ssl
from ursa.util.secrets import SecretReference, keyring_usernames

__all__ = ["SetupError"]
_Path = Path | None
_doctor = build_report_from_path
_load_config = load_setup_config
_save = save_setup_config


def _friendly(exc: Exception, mcp: bool = False) -> str:
    messages, current = [], exc
    while current:
        messages.append(str(current).strip())
        current = current.__cause__ or current.__context__
    messages = list(dict.fromkeys(filter(None, messages)))
    joined = " ".join(messages).lower()
    if any(word in joined for word in ("401", "403", "unauthorized")):
        return (
            "The MCP server rejected the auth key."
            if mcp
            else "The API key was rejected. Check its value and permissions."
        )
    if "404" in joined or "not found" in joined:
        return "The endpoint was not found. Check the URL."
    if "timed out" in joined:
        return "The connection timed out. Check that the service is reachable."
    if any(word in joined for word in ("connection refused", "connecterror")):
        return "URSA could not connect. Check the address, network, and VPN."
    return messages[0] if messages else type(exc).__name__


def _probe_entry(candidate: EntryCandidate) -> Exception | None:
    if candidate.kind == "provider":
        args = (probe_inference_provider, candidate.runtime, True)
        kwargs = {}
    else:
        args = (probe_mcp_server, candidate.name, candidate.runtime)
        kwargs = {"timeout": 15}
    try:
        run_probe_with_deadline(*args, **kwargs)
    except Exception as exc:
        return exc
    return None


# fmt: off
SetupAction = Enum("SetupAction", {"EXIT": "exit", "TERMINAL": "terminal", "DASHBOARD": "dashboard"})
_DONE_ACTIONS = (("Launch URSA - Terminal", "terminal", "success"), ("Launch URSA - Dashboard", "dashboard", "success"), ("Exit", "exit", "default"))
_ENTRY_PAGE = {"provider": ("#provider-error", "providers", "Provider"), "mcp": ("#mcp-validation-status", "MCP servers", "MCP")}
_REVIEW_ACTIONS = ("back", "save", "recheck", "continue")
_PAGE_ORDER = ("welcome", "provider", "models", "mcp", "review", "done")
# fmt: on


class SetupApp(App[SetupAction]):
    CSS_PATH = "setup.tcss"
    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),
        Binding("ctrl+q", "cancel", "Quit", priority=True),
    ]
    DONE = {action.value: action for action in SetupAction}

    def __init__(self, path: _Path = None, defaults_json: _Path = None):
        super().__init__()
        self.path = (path or default_user_config_path()).expanduser()
        self.config, self.llm_configured = _load_config(self.path)
        if defaults_json is not None:
            self.llm_configured = apply_setup_defaults(
                self.config,
                load_setup_defaults(defaults_json),
                llm_configured=self.llm_configured,
            )
        self.page = "welcome"
        self.models_valid = False
        self.pending: dict[str, str] = {}
        self.doctor_summary = ""

    def _page(
        self,
        name: str,
        title: str,
        *content: Any,
        copy: str = "",
        actions: tuple = (),
        status: str | None = None,
    ) -> VerticalScroll:
        children = [Static(title, classes="page-title"), *content]
        if copy:
            children.insert(1, Static(copy, classes="page-copy"))
        if status:
            children.append(Static("", id=status, classes="form-message"))
            actions = (
                ("Back", f"{name}-back", "default"),
                ("Validate and continue", f"{name}-save", "primary"),
            )
        if actions:
            buttons = (
                Button(label, id=id, variant=variant)
                for label, id, variant in actions
            )
            children.append(Horizontal(*buttons, classes="setup-actions"))
        return VerticalScroll(
            *children, id=f"{name}-page", classes="setup-page"
        )

    def compose(self) -> ComposeResult:
        self.provider_tabs = DynamicEntryTabs(
            "provider", self.config.inference_providers, self._keys()
        )
        self.mcp_tabs = DynamicEntryTabs(
            "mcp", self.config.mcp_servers, self._keys()
        )
        # fmt: off
        pages = (
            self._page("welcome", "Welcome to URSA", Static(f"Configuration: {self.path}", classes="path-box"), copy="URSA found your existing user configuration. This guide will preserve unrelated settings while you review each setup step." if self.path.is_file() else "This guide will connect URSA to a model provider, choose your models, and optionally add MCP tools. Nothing is written until the final Save button.", actions=(("Begin setup", "welcome-next", "primary"),)),
            self._page("provider", "Connect an inference provider", self.provider_tabs, status="provider-error", copy="URSA verifies the API key by asking the provider for its model list. The key is hidden and is not written to YAML."),
            self._page("models", "Choose default models", Vertical(id="setup-model-host"), status="model-validation-status", copy="Choose the language model URSA will use. Embeddings are optional and may use a different inference provider."),
            self._page("mcp", "Configure an MCP server", self.mcp_tabs, status="mcp-validation-status", copy="MCP servers add tools and data sources to URSA. Choose a remote server URL or a local command."),
            self._page(
                "review", "Write configuration",
                Static("", id="review-summary", classes="review-box"),
                Collapsible(TextArea(language="yaml", show_line_numbers=True, read_only=True, id="review-full-config"), title="Full config", id="review-full", classes="compact-collapsible"),
                Collapsible(Static("Validation has not run.", id="review-validation-results", markup=False), title="URSA Doctor — not run", id="review-validation", classes="compact-collapsible"),
                Static("", id="review-error", classes="form-message"),
                actions=(("Back", "review-back", "default"), ("Write and validate", "review-save", "success"), ("Recheck", "review-recheck", "primary"), ("Continue", "review-continue", "success")),
                copy="API keys and tokens will go to your URSA keyring. The YAML file stores only keyring references.",
            ),
            self._page("done", "URSA is ready", Static("", id="done-copy"), actions=_DONE_ACTIONS),
        )
        # fmt: on
        yield Vertical(
            ContentSwitcher(*pages, initial="welcome-page", id="setup-pages"),
            id="setup-main",
        )
        yield Footer()

    def _show(self, page: str) -> None:
        previous = self.page
        self.page = page
        self.query_one("#setup-pages", ContentSwitcher).current = f"{page}-page"
        if page == "models" and page != previous:
            self.run_worker(
                self._prepare_models(), group="model-editor", exclusive=True
            )
        elif page == "review":
            self._review()

    async def action_back(self) -> None:
        if self.page == "models":
            self.workers.cancel_group(self, "model-check")
        if index := _PAGE_ORDER.index(self.page):
            self._show(_PAGE_ORDER[index - 1])

    def action_cancel(self) -> None:
        self.exit(SetupAction.EXIT)

    def _status(self, selector: str, text: str, failed: bool = False) -> None:
        widget = self.query_one(selector, Static)
        widget.update(text)
        widget.set_class(failed, "error")

    # fmt: off
    def _keys(self) -> list[str]:
        return keyring_usernames({"inference_providers": self.config.inference_providers, "mcp_servers": self.config.mcp_servers})

    def _discard_unused_pending(self, usernames) -> None:
        for username in set(usernames).difference(self._keys()):
            self.pending.pop(username, None)

    def _providers(self) -> dict[str, InferenceProviderConfig]:
        providers = {}
        for name, provider in self.config.inference_providers.items():
            username = provider.api_key.keyring_username(name) if isinstance(provider.api_key, SecretReference) else None
            secret = self.pending.get(username) if username else None
            providers[name] = prepare_provider_probe(self.config, name, SecretStr(secret) if secret else None)
        return providers

    def _begin_entry(self, kind: str, name: str | None = None) -> None:
        self._show(kind)
        (self.provider_tabs if kind == "provider" else self.mcp_tabs).select_name(name)
        self._status(_ENTRY_PAGE[kind][0], "")

    async def _prepare_models(self) -> None:
        providers = self._providers()
        chat = self.config.llm_model if self.llm_configured else ChatModelConfig(model="", inference_provider=next(iter(providers), None))
        host = self.query_one("#setup-model-host", Vertical)
        await host.remove_children()
        if self.page == "models":
            await host.mount(SetupModelEditor(providers, chat, self.config.emb_model))
    # fmt: on

    @on(SetupModelEditor.Submitted)
    def setup_models_submitted(self, event: SetupModelEditor.Submitted) -> None:
        event.stop()
        self._models_selected(event.selection)

    def _models_selected(self, selection: SetupModelSelection) -> None:
        self.config = replace_models(
            self.config, selection.chat, selection.embedding
        )
        self.doctor_summary = ""
        self.llm_configured = True
        self.models_valid = False
        self.run_worker(
            self._validate_models(), group="model-check", exclusive=True
        )

    async def _validate_models(self) -> None:
        providers = self._providers()
        statuses: dict[str, str] = {}

        def show_status(prefix: str, status: str) -> None:
            statuses[prefix] = status
            self._status(
                "#model-validation-status",
                "\n".join(statuses.values()),
                any(value.startswith("✗") for value in statuses.values()),
            )

        async def check(prefix: str, label: str, model: Any) -> bool:
            if model is None:
                return True
            show_status(prefix, f"Validating {label}...")
            try:
                await asyncio.to_thread(_probe_model, model, providers)
            except Exception as exc:
                valid = False
                message = _friendly(exc)
                status = f"✗ {label.capitalize()}: {message}"
                self.notify(message, severity="error", markup=False)
            else:
                valid, status = True, f"✓ {label.capitalize()} connected"
            show_status(prefix, status)
            return valid

        checks = (
            ("llm", "language model", self.config.llm_model),
            ("embedding", "embedding model", self.config.emb_model),
        )
        valid = all(await asyncio.gather(*(check(*item) for item in checks)))
        self.models_valid = valid
        if valid:
            self._begin_entry("mcp", next(iter(self.config.mcp_servers), None))
        else:
            await self._prepare_models()

    # fmt: off
    async def _validate_entry_page(self, kind: str) -> bool:
        provider = kind == "provider"
        tabs = self.provider_tabs if provider else self.mcp_tabs
        selector, label, subject = _ENTRY_PAGE[kind]
        other = self.mcp_tabs if provider else self.provider_tabs
        candidates = tabs.candidates(other.raw_names())
        if candidates is None:
            return False
        old_users = set(self._keys())
        if not candidates:
            if provider:
                self._status(selector, "At least one inference provider is required.", True)
                return False
            self.config = replace_mcp_servers(self.config, {})
            self.doctor_summary = ""
            self._discard_unused_pending(old_users)
            self._status(selector, "✓ No MCP servers configured")
            return True

        self._status(selector, f"Validating {label}...")
        results = await asyncio.gather(*(asyncio.to_thread(_probe_entry, item) for item in candidates))
        failures: list[EntryCandidate] = []
        for candidate, result in zip(candidates, results, strict=True):
            failed = result is not None
            tabs.set_draft_status(candidate.draft_id, _friendly(result, not provider) if failed else "✓ Connected", marker="✗ " if failed else "✓ ", failed=failed)
            if failed:
                failures.append(candidate)
        if failures:
            tabs.focus_error(failures[0].draft_id)
            names = ", ".join(candidate.name for candidate in failures)
            self._status(selector, f"{subject} validation failed: {names}", True)
            return False

        stored = {item.name: item.stored for item in candidates}
        renamed = {item.original_name: item.name for item in candidates if item.original_name is not None}
        replace: Any = replace_inference_providers if provider else replace_mcp_servers
        self.config = replace(self.config, stored, **({"renamed": renamed} if provider else {}))
        self.doctor_summary = ""
        self._discard_unused_pending(old_users)
        for item in candidates:
            if item.credential.pending and item.credential.username:
                self.pending[item.credential.username] = item.credential.pending
        self._status(selector, f"✓ All {label} connected")
        return True

    def _review(self) -> None:
        llm = self.config.llm_model
        emb = self.config.emb_model
        self.query_one("#review-summary", Static).update(
            f"Providers: {', '.join(self.config.inference_providers)}\n"
            f"Language model: {llm.model if llm else 'none'}\n"
            f"Embedding model: {emb.model if emb else 'none'}\n"
            f"MCP servers: {', '.join(self.config.mcp_servers) or 'none'}"
        )
        self.query_one("#review-full-config", TextArea).text = format_setup_config_yaml(self.config)
        valid = bool(self.config.inference_providers and self.models_valid)
        if not self.doctor_summary:
            self._status("#review-error", "")
            validation = self.query_one("#review-validation", Collapsible)
            validation.title = "URSA Doctor — not run"
            validation.remove_class("doctor-error", "doctor-warning")
            self.query_one("#review-validation-results", Static).update(
                "Validation has not run."
            )
            self._review_actions("ready", valid)

    async def _save_config(self) -> None:
        self._review_actions("running")
        self._status("#review-error", "Writing configuration…")
        try:
            path = await asyncio.to_thread(
                _save, self.path, self.config, self.pending
            )
        except Exception as exc:
            self._review()
            self._status("#review-error", f"Could not save: {exc}", True)
            return
        self.path = path
        self.pending.clear()
        self._status("#review-error", f"Saved to {path}")
        await self._run_doctor(auto_advance=True)

    def _review_actions(self, state: str, valid: bool = True) -> None:
        visible = {"ready": {"back", "save"}, "checked": {"recheck", "continue"}}.get(state, set())
        for action in _REVIEW_ACTIONS:
            button = self.query_one(f"#review-{action}", Button)
            button.set_class(action not in visible, "hidden")
            button.disabled = state == "running" or action == "save" and not valid
    # fmt: on

    async def _run_doctor(self, *, auto_advance: bool = False) -> None:
        self._review_actions("running")
        validation = self.query_one("#review-validation", Collapsible)
        results = self.query_one("#review-validation-results", Static)
        validation.title = "URSA Doctor — running…"
        validation.collapsed = False
        results.update("Running ursa self doctor…")
        report = await asyncio.to_thread(_doctor, self.path)
        counts = report_counts(report)
        warnings = counts[CheckStatus.WARN]
        errors = counts[CheckStatus.ERROR]
        self.doctor_summary = (
            f"Validation: {warnings} warning(s), {errors} error(s)"
        )
        validation.title = (
            f"URSA Doctor — {warnings} warning(s), {errors} error(s)"
        )
        validation.set_class(bool(errors), "doctor-error")
        validation.set_class(bool(warnings) and not errors, "doctor-warning")
        results.update(render_report(report))
        self._review_actions("checked")
        if auto_advance and not warnings and not errors:
            self._finish_setup()

    def _finish_setup(self) -> None:
        if not self.doctor_summary:
            return
        self.query_one("#done-copy", Static).update(
            f"Saved to {self.path}\n{self.doctor_summary}"
        )
        self._show("done")

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        id = event.button.id
        if id in self.DONE:
            self.exit(self.DONE[id])
        elif id == "welcome-next":
            self._begin_entry(
                "provider", next(iter(self.config.inference_providers), None)
            )
        elif id and id.endswith("-back"):
            await self.action_back()
        elif id in {"provider-save", "mcp-save"}:
            kind = id.removesuffix("-save")
            if await self._validate_entry_page(kind):
                self._show("models" if kind == "provider" else "review")
        elif id == "models-save":
            self.query_one(SetupModelEditor).submit()
        elif id == "review-save":
            await self._save_config()
        elif id == "review-recheck":
            await self._run_doctor()
        elif id == "review-continue":
            self._finish_setup()


def _probe_model(config, providers: dict[str, InferenceProviderConfig]) -> None:
    resolved = config.resolve_inference_provider(providers)
    if isinstance(config, ChatModelConfig):
        run_probe_with_deadline(probe_chat_config, resolved, "chat")
    else:
        run_probe_with_deadline(probe_embedding_model, resolved)


def run_setup(path: _Path = None, defaults_json: _Path = None) -> SetupAction:
    inject_truststore_into_ssl()
    return SetupApp(path, defaults_json).run() or SetupAction.EXIT
