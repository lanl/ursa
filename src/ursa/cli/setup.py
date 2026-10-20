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
)

from ursa.cli.config import (
    ChatModelConfig,
    InferenceProviderConfig,
)
from ursa.cli.doctor import (
    CheckResult,
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
from ursa.cli.setup_entries import (
    API_KEY_ENV,
    CUSTOM_PROVIDER_URL,
    MCP_TOKEN_NONE,
    PROVIDER_DEFAULT_URL,
    PROVIDER_DEFAULT_URLS,
    SPINNER,
    ApiKeyEntry,
    CredentialValue,
    DynamicEntryTabs,
    McpCandidate,
    McpDraftEditor,
    ProviderCandidate,
    ProviderDraftEditor,
)
from ursa.cli.setup_models import SetupModelEditor, SetupModelSelection
from ursa.cli.tui.widgets import CompactCollapsible, YamlTextArea
from ursa.util.crossplatform import default_user_config_path
from ursa.util.health import (
    probe_chat_config,
    probe_embedding_model,
    probe_inference_provider,
    probe_mcp_server,
    run_probe_with_deadline,
)
from ursa.util.http import inject_truststore_into_ssl
from ursa.util.secrets import (
    SecretReference,
    keyring_usernames,
)

__all__ = [
    "API_KEY_ENV",
    "CUSTOM_PROVIDER_URL",
    "MCP_TOKEN_NONE",
    "PROVIDER_DEFAULT_URL",
    "PROVIDER_DEFAULT_URLS",
    "SPINNER",
    "ApiKeyEntry",
    "CredentialValue",
    "DynamicEntryTabs",
    "McpCandidate",
    "McpDraftEditor",
    "ProviderCandidate",
    "ProviderDraftEditor",
    "SetupAction",
    "SetupApp",
    "SetupError",
    "apply_setup_defaults",
    "load_setup_defaults",
    "run_setup",
]


def _friendly(exc: Exception, mcp: bool = False) -> str:
    messages, current = [], exc
    while current:
        text = str(current).strip()
        if text and text not in messages:
            messages.append(text)
        current = current.__cause__ or current.__context__
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


_load_config = load_setup_config
_save = save_setup_config


def _doctor(path: Path) -> CheckResult:
    """Validate the exact configuration artifact written by setup."""
    return build_report_from_path(path)


SetupAction = Enum(
    "SetupAction",
    {"EXIT": "exit", "TERMINAL": "terminal", "DASHBOARD": "dashboard"},
)


class SetupApp(App[SetupAction]):
    CSS_PATH = "setup.tcss"
    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),
        Binding("ctrl+q", "cancel", "Quit", priority=True),
    ]
    DONE = {
        "setup-launch-terminal": SetupAction.TERMINAL,
        "setup-launch-dashboard": SetupAction.DASHBOARD,
        "setup-exit": SetupAction.EXIT,
    }

    def __init__(
        self,
        path: Path | None = None,
        defaults_json: Path | None = None,
    ) -> None:
        super().__init__()
        self.path = (path or default_user_config_path()).expanduser()
        self.config, self.llm_configured = _load_config(self.path)
        if defaults_json is not None:
            self.llm_configured = apply_setup_defaults(
                self.config,
                load_setup_defaults(defaults_json),
                llm_configured=self.llm_configured,
            )
        self.page, self.history = "welcome", []
        self.model_ok = {"llm": None, "embedding": None}
        self.pending: dict[str, str] = {}
        self.saved_path: Path | None = None
        self.doctor_report: CheckResult | None = None

    @staticmethod
    def _buttons(*specs: tuple) -> Horizontal:
        return Horizontal(
            *(
                Button(label, id=id, variant=variant, disabled=disabled)
                for label, id, variant, disabled in specs
            ),
            classes="setup-actions",
        )

    def _page(
        self,
        name: str,
        title: str,
        *content: Any,
        copy: str = "",
        actions: tuple = (),
    ) -> VerticalScroll:
        children = [Static(title, classes="page-title")]
        if copy:
            children.append(Static(copy, classes="page-copy"))
        children.extend(content)
        if actions:
            children.append(self._buttons(*actions))
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
        pages = (
            self._page(
                "welcome",
                "Welcome to URSA",
                Static(f"Configuration: {self.path}", classes="path-box"),
                copy=(
                    "URSA found your existing user configuration. This guide will preserve unrelated settings while you review each setup step."
                    if self.path.is_file()
                    else "This guide will connect URSA to a model provider, choose your models, and optionally add MCP tools. Nothing is written until the final Save button."
                ),
                actions=(("Begin setup", "welcome-next", "primary", False),),
            ),
            self._page(
                "provider",
                "Connect an inference provider",
                self.provider_tabs,
                Static("", id="provider-error", classes="form-message"),
                copy="URSA verifies the API key by asking the provider for its model list. The key is hidden and is not written to YAML.",
                actions=(
                    ("Back", "provider-back", "default", False),
                    (
                        "Validate and continue",
                        "provider-save",
                        "primary",
                        False,
                    ),
                ),
            ),
            self._page(
                "models",
                "Choose default models",
                Vertical(id="setup-model-host"),
                Static(
                    "", id="model-validation-status", classes="form-message"
                ),
                copy="Choose the language model URSA will use. Embeddings are optional and may use a different inference provider.",
                actions=(
                    ("Back", "models-back", "default", False),
                    (
                        "Validate and continue",
                        "models-save",
                        "primary",
                        False,
                    ),
                ),
            ),
            self._page(
                "mcp",
                "Configure an MCP server",
                self.mcp_tabs,
                Static("", id="mcp-validation-status", classes="form-message"),
                copy="MCP servers add tools and data sources to URSA. Choose a remote server URL or a local command.",
                actions=(
                    ("Back", "mcp-back", "default", False),
                    (
                        "Validate and continue",
                        "mcp-save",
                        "primary",
                        False,
                    ),
                ),
            ),
            self._page(
                "review",
                "Write configuration",
                Static("", id="review-summary", classes="review-box"),
                CompactCollapsible(
                    YamlTextArea(
                        read_only=True,
                        id="review-full-config",
                    ),
                    title="Full config",
                    id="review-full",
                ),
                CompactCollapsible(
                    Static(
                        "Validation has not run.",
                        id="review-validation-results",
                        markup=False,
                    ),
                    title="Validation — not run",
                    id="review-validation",
                ),
                Static("", id="review-error", classes="form-message"),
                actions=(
                    ("Back", "review-back", "default", False),
                    (
                        "Write and validate",
                        "review-save",
                        "success",
                        True,
                    ),
                    (
                        "Retry validation",
                        "review-retry",
                        "primary",
                        False,
                    ),
                    ("Finish", "review-finish", "success", False),
                ),
                copy="API keys and tokens will go to your URSA keyring. The YAML file stores only keyring references.",
            ),
            self._page(
                "done",
                "URSA is ready",
                Static("", id="done-copy"),
                actions=tuple(
                    (label, id, variant, False)
                    for id, label, variant in (
                        (
                            "setup-launch-terminal",
                            "Launch URSA - Terminal",
                            "success",
                        ),
                        (
                            "setup-launch-dashboard",
                            "Launch URSA - Dashboard",
                            "success",
                        ),
                        ("setup-exit", "Exit", "default"),
                    )
                ),
            ),
        )
        yield Vertical(
            ContentSwitcher(*pages, initial="welcome-page", id="setup-pages"),
            id="setup-main",
        )
        yield Footer()

    def _show(self, page: str, remember: bool = True) -> None:
        previous = self.page
        if remember and page != self.page:
            self.history.append(self.page)
        self.page = page
        self.query_one("#setup-pages", ContentSwitcher).current = f"{page}-page"
        if page == "models" and page != previous:
            self.run_worker(
                self._prepare_models(), group="model-editor", exclusive=True
            )
        elif page == "review":
            self._review()

    async def action_back(self) -> None:
        if self.history:
            self._show(self.history.pop(), False)

    def action_cancel(self) -> None:
        self.exit(SetupAction.EXIT)

    def _status(self, selector: str, text: str, failed: bool = False) -> None:
        widget = self.query_one(selector, Static)
        widget.update(text)
        widget.set_class(failed, "error")

    def _keys(self) -> list[str]:
        config = {
            "inference_providers": self.config.inference_providers,
            "mcp_servers": self.config.mcp_servers,
        }
        return keyring_usernames(config)

    def _discard_unused_pending(self, usernames) -> None:
        active = set(self._keys())
        for username in set(usernames) - active:
            self.pending.pop(username, None)

    def _provider(self, name: str) -> InferenceProviderConfig:
        reference = self.config.inference_providers[name].api_key
        staged_secret = None
        if isinstance(reference, SecretReference):
            username = reference.keyring_username(name)
            secret = self.pending.get(username) if username else None
            staged_secret = SecretStr(secret) if secret else None
        return prepare_provider_probe(self.config, name, staged_secret)

    def _begin_provider(self, name: str | None = None) -> None:
        self._show("provider")
        self.provider_tabs.select_name(name)
        self._status("#provider-error", "")

    async def _validate_provider_page(self) -> bool:
        """Validate every draft, probe all candidates, then commit once."""
        candidates = self.provider_tabs.candidates(self.mcp_tabs.raw_names())
        if candidates is None or not candidates:
            if candidates == []:
                self._status(
                    "#provider-error",
                    "At least one inference provider is required.",
                    True,
                )
            return False
        typed = [
            candidate
            for candidate in candidates
            if isinstance(candidate, ProviderCandidate)
        ]
        self._status("#provider-error", "Validating providers...")

        async def probe(candidate: ProviderCandidate) -> Exception | None:
            try:
                await asyncio.to_thread(
                    run_probe_with_deadline,
                    probe_inference_provider,
                    candidate.runtime,
                    True,
                )
            except Exception as exc:
                return exc
            return None

        results = await asyncio.gather(*(probe(item) for item in typed))
        failures = [
            (candidate, result)
            for candidate, result in zip(typed, results, strict=True)
            if result is not None
        ]
        for candidate, result in zip(typed, results, strict=True):
            if result is None:
                self.provider_tabs.set_draft_status(
                    candidate.draft_id,
                    "✓ Connected",
                    marker="✓ ",
                )
            else:
                self.provider_tabs.set_draft_status(
                    candidate.draft_id,
                    _friendly(result),
                    marker="✗ ",
                    failed=True,
                )
        if failures:
            first, _ = failures[0]
            self.provider_tabs.focus_error(first.draft_id)
            names = ", ".join(candidate.name for candidate, _ in failures)
            self._status(
                "#provider-error", f"Provider validation failed: {names}", True
            )
            return False

        old_users = set(
            keyring_usernames({
                "inference_providers": self.config.inference_providers
            })
        )
        providers = {item.name: item.stored for item in typed}
        renamed = {
            item.original_name: item.name
            for item in typed
            if item.original_name is not None
        }
        self.config = replace_inference_providers(
            self.config,
            providers,
            renamed=renamed,
        )
        self._discard_unused_pending(old_users)
        for item in typed:
            if item.credential.pending and item.credential.username:
                self.pending[item.credential.username] = item.credential.pending
        self._status("#provider-error", "✓ All providers connected")
        return True

    async def _prepare_models(self) -> None:
        providers = {
            name: self._provider(name)
            for name in self.config.inference_providers
        }
        chat = (
            self.config.llm_model
            if self.llm_configured
            else ChatModelConfig(
                model="", inference_provider=next(iter(providers), None)
            )
        )
        host = self.query_one("#setup-model-host", Vertical)
        await host.remove_children()
        if self.page == "models":
            await host.mount(
                SetupModelEditor(providers, chat, self.config.emb_model)
            )

    @on(SetupModelEditor.Submitted)
    def setup_models_submitted(self, event: SetupModelEditor.Submitted) -> None:
        event.stop()
        self._models_selected(event.selection)

    def _models_selected(self, selection: SetupModelSelection) -> None:
        self.config = replace_models(
            self.config,
            selection.chat,
            selection.embedding,
        )
        self.llm_configured = True
        self.run_worker(
            self._validate_models(), group="model-check", exclusive=True
        )

    async def _validate_models(self) -> None:
        providers = {
            name: self._provider(name)
            for name in self.config.inference_providers
        }
        labels = {
            "llm": "language model",
            "embedding": "embedding model",
        }
        statuses: dict[str, str] = {}

        def show_statuses() -> None:
            self._status(
                "#model-validation-status",
                "\n".join(
                    statuses[prefix]
                    for prefix in ("llm", "embedding")
                    if prefix in statuses
                ),
                any(status.startswith("✗") for status in statuses.values()),
            )

        async def check(prefix: str) -> bool:
            model = (
                self.config.llm_model
                if prefix == "llm"
                else self.config.emb_model
            )
            if model is None:
                self.model_ok[prefix] = True
                return True
            statuses[prefix] = f"Validating {labels[prefix]}..."
            show_statuses()
            try:
                await asyncio.to_thread(_probe_model, model, providers)
            except Exception as exc:
                self.model_ok[prefix] = False
                message = _friendly(exc)
                statuses[prefix] = f"✗ {labels[prefix].capitalize()}: {message}"
                show_statuses()
                self.notify(message, severity="error", markup=False)
                return False
            else:
                self.model_ok[prefix] = True
                statuses[prefix] = f"✓ {labels[prefix].capitalize()} connected"
                show_statuses()
                return True

        valid = all(await asyncio.gather(check("llm"), check("embedding")))
        if valid:
            self._begin_mcp(next(iter(self.config.mcp_servers), None))
        else:
            await self._prepare_models()

    def _begin_mcp(self, name: str | None = None) -> None:
        self._show("mcp")
        self.mcp_tabs.select_name(name)
        self._status("#mcp-validation-status", "")

    async def _validate_mcp_page(self) -> bool:
        """Validate every draft, probe all candidates, then commit once."""
        candidates = self.mcp_tabs.candidates(self.provider_tabs.raw_names())
        if candidates is None:
            return False
        typed = [
            candidate
            for candidate in candidates
            if isinstance(candidate, McpCandidate)
        ]
        if not typed:
            old_users = set(
                keyring_usernames({"mcp_servers": self.config.mcp_servers})
            )
            self.config = replace_mcp_servers(self.config, {})
            self._discard_unused_pending(old_users)
            self._status(
                "#mcp-validation-status", "✓ No MCP servers configured"
            )
            return True

        self._status("#mcp-validation-status", "Validating MCP servers...")

        async def probe(candidate: McpCandidate) -> Exception | None:
            try:
                await asyncio.to_thread(
                    run_probe_with_deadline,
                    probe_mcp_server,
                    candidate.name,
                    candidate.runtime,
                    timeout=15,
                )
            except Exception as exc:
                return exc
            return None

        results = await asyncio.gather(*(probe(item) for item in typed))
        failures = [
            (candidate, result)
            for candidate, result in zip(typed, results, strict=True)
            if result is not None
        ]
        for candidate, result in zip(typed, results, strict=True):
            if result is None:
                self.mcp_tabs.set_draft_status(
                    candidate.draft_id,
                    "✓ Connected",
                    marker="✓ ",
                )
            else:
                self.mcp_tabs.set_draft_status(
                    candidate.draft_id,
                    _friendly(result, True),
                    marker="✗ ",
                    failed=True,
                )
        if failures:
            first, _ = failures[0]
            self.mcp_tabs.focus_error(first.draft_id)
            names = ", ".join(candidate.name for candidate, _ in failures)
            self._status(
                "#mcp-validation-status",
                f"MCP validation failed: {names}",
                True,
            )
            return False

        old_users = set(
            keyring_usernames({"mcp_servers": self.config.mcp_servers})
        )
        servers = {item.name: item.stored for item in typed}
        self.config = replace_mcp_servers(self.config, servers)
        self._discard_unused_pending(old_users)
        for item in typed:
            if item.credential.pending and item.credential.username:
                self.pending[item.credential.username] = item.credential.pending
        self._status("#mcp-validation-status", "✓ All MCP servers connected")
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
        self.query_one(
            "#review-full-config", YamlTextArea
        ).text = format_setup_config_yaml(self.config)
        valid = (
            bool(self.config.inference_providers)
            and self.model_ok["llm"] is True
        )
        if self.saved_path is None:
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
        self.saved_path = path
        self.pending.clear()
        self._status("#review-error", f"Saved to {path}")
        await self._run_doctor()

    def _review_actions(self, state: str, valid: bool = True) -> None:
        visible = {
            "review-back": state == "ready",
            "review-save": state == "ready",
            "review-retry": state == "failed",
            "review-finish": state in {"validated", "failed"},
        }
        for id, show in visible.items():
            button = self.query_one(f"#{id}", Button)
            button.set_class(not show, "hidden")
            button.disabled = (
                state == "running" or id == "review-save" and not valid
            )
        finish = self.query_one("#review-finish", Button)
        finish.label = "Finish anyway" if state == "failed" else "Finish"

    async def _run_doctor(self) -> None:
        if self.saved_path is None:
            return
        self._review_actions("running")
        validation = self.query_one("#review-validation", Collapsible)
        validation.title = "Validation — running…"
        validation.collapsed = False
        self.query_one("#review-validation-results", Static).update(
            "Running ursa self doctor…"
        )
        report = await asyncio.to_thread(_doctor, self.saved_path)
        self.doctor_report = report
        counts = report_counts(report)
        warnings = counts[CheckStatus.WARN]
        errors = counts[CheckStatus.ERROR]
        validation.title = (
            f"Validation — {warnings} warning(s), {errors} error(s)"
        )
        validation.set_class(bool(errors), "doctor-error")
        validation.set_class(bool(warnings) and not errors, "doctor-warning")
        self.query_one("#review-validation-results", Static).update(
            render_report(report)
        )
        if not warnings and not errors:
            self._finish_setup()
        else:
            self._review_actions("failed" if errors else "validated")

    def _finish_setup(self) -> None:
        if self.saved_path is None:
            return
        summary = f"Saved to {self.saved_path}"
        if self.doctor_report is not None:
            counts = report_counts(self.doctor_report)
            summary += (
                f"\nValidation: {counts[CheckStatus.WARN]} warning(s), "
                f"{counts[CheckStatus.ERROR]} error(s)"
            )
        self.query_one("#done-copy", Static).update(summary)
        self._show("done")

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        id = event.button.id
        if id in self.DONE:
            self.exit(self.DONE[id])
        elif id == "welcome-next":
            self._begin_provider(
                next(iter(self.config.inference_providers), None)
            )
        elif id and id.endswith("-back"):
            await self.action_back()
        elif id == "provider-save":
            if await self._validate_provider_page():
                self._show("models")
        elif id == "models-save":
            self.query_one(SetupModelEditor).submit()
        elif id == "mcp-save":
            if await self._validate_mcp_page():
                self._show("review")
        elif id == "review-save":
            await self._save_config()
        elif id == "review-retry":
            await self._run_doctor()
        elif id == "review-finish":
            self._finish_setup()


def _probe_model(config, providers: dict[str, InferenceProviderConfig]) -> None:
    resolved = config.resolve_inference_provider(providers)
    if isinstance(config, ChatModelConfig):
        run_probe_with_deadline(probe_chat_config, resolved, "chat")
    else:
        run_probe_with_deadline(probe_embedding_model, resolved)


def run_setup(
    path: Path | None = None,
    defaults_json: Path | None = None,
) -> SetupAction:
    inject_truststore_into_ssl()
    app = SetupApp(path, defaults_json)
    return app.run() or SetupAction.EXIT
