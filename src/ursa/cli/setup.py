"""Guided first-run configuration for URSA."""

from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path
from typing import Any, cast

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

from ursa.cli.config import (
    ChatModelConfig,
    InferenceProviderConfig,
)
from ursa.cli.doctor import (
    CheckResult,
    CheckStatus,
    render_report,
    report_counts,
)
from ursa.cli.doctor import (
    build_report_from_path as _doctor,
)
from ursa.cli.setup_config import (
    SetupError as SetupError,
)
from ursa.cli.setup_config import (
    apply_setup_defaults,
    format_setup_config_yaml,
    load_setup_defaults,
    prepare_provider_probe,
    replace_inference_providers,
    replace_mcp_servers,
    replace_models,
)
from ursa.cli.setup_config import (
    load_setup_config as _load_config,
)
from ursa.cli.setup_config import (
    save_setup_config as _save,
)
from ursa.cli.setup_entries import (
    DynamicEntryTabs,
    McpCandidate,
    ProviderCandidate,
)
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
from ursa.util.mcp import ServerParameters
from ursa.util.secrets import (
    SecretReference,
    keyring_usernames,
)


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


async def _probe_entry(
    candidate: ProviderCandidate | McpCandidate,
) -> Exception | None:
    if isinstance(candidate, ProviderCandidate):
        args = (probe_inference_provider, candidate.runtime, True)
        kwargs = {}
    else:
        args = (probe_mcp_server, candidate.name, candidate.runtime)
        kwargs = {"timeout": 15}
    try:
        await asyncio.to_thread(run_probe_with_deadline, *args, **kwargs)
    except Exception as exc:
        return exc
    return None


SetupAction = Enum(
    "SetupAction",
    {"EXIT": "exit", "TERMINAL": "terminal", "DASHBOARD": "dashboard"},
)
_DONE_ACTIONS = (
    ("Launch URSA - Terminal", "setup-launch-terminal", "success"),
    ("Launch URSA - Dashboard", "setup-launch-dashboard", "success"),
    ("Exit", "setup-exit", "default"),
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
        buttons = (
            Button(label, id=id, variant=variant, disabled=id == "review-save")
            for label, id, variant in specs
        )
        return Horizontal(*buttons, classes="setup-actions")

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

    def _step_page(
        self,
        name: str,
        title: str,
        content: Any,
        status_id: str,
        copy: str,
    ) -> VerticalScroll:
        return self._page(
            name,
            title,
            content,
            Static("", id=status_id, classes="form-message"),
            copy=copy,
            actions=(
                ("Back", f"{name}-back", "default"),
                ("Validate and continue", f"{name}-save", "primary"),
            ),
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
                actions=(("Begin setup", "welcome-next", "primary"),),
            ),
            self._step_page(
                "provider",
                "Connect an inference provider",
                self.provider_tabs,
                "provider-error",
                "URSA verifies the API key by asking the provider for its model list. The key is hidden and is not written to YAML.",
            ),
            self._step_page(
                "models",
                "Choose default models",
                Vertical(id="setup-model-host"),
                "model-validation-status",
                "Choose the language model URSA will use. Embeddings are optional and may use a different inference provider.",
            ),
            self._step_page(
                "mcp",
                "Configure an MCP server",
                self.mcp_tabs,
                "mcp-validation-status",
                "MCP servers add tools and data sources to URSA. Choose a remote server URL or a local command.",
            ),
            self._page(
                "review",
                "Write configuration",
                Static("", id="review-summary", classes="review-box"),
                Collapsible(
                    TextArea(
                        language="yaml",
                        show_line_numbers=True,
                        read_only=True,
                        id="review-full-config",
                    ),
                    title="Full config",
                    id="review-full",
                    classes="compact-collapsible",
                ),
                Collapsible(
                    Static(
                        "Validation has not run.",
                        id="review-validation-results",
                        markup=False,
                    ),
                    title="URSA Doctor — not run",
                    id="review-validation",
                    classes="compact-collapsible",
                ),
                Static("", id="review-error", classes="form-message"),
                actions=(
                    ("Back", "review-back", "default"),
                    ("Write and validate", "review-save", "success"),
                    ("Recheck", "review-recheck", "primary"),
                    ("Continue", "review-continue", "success"),
                ),
                copy="API keys and tokens will go to your URSA keyring. The YAML file stores only keyring references.",
            ),
            self._page(
                "done",
                "URSA is ready",
                Static("", id="done-copy"),
                actions=_DONE_ACTIONS,
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
        return keyring_usernames({
            "inference_providers": self.config.inference_providers,
            "mcp_servers": self.config.mcp_servers,
        })

    def _discard_unused_pending(self, usernames) -> None:
        for username in set(usernames) - set(self._keys()):
            self.pending.pop(username, None)

    def _providers(self) -> dict[str, InferenceProviderConfig]:
        providers = {}
        for name, provider in self.config.inference_providers.items():
            username = (
                provider.api_key.keyring_username(name)
                if isinstance(provider.api_key, SecretReference)
                else None
            )
            secret = self.pending.get(username) if username else None
            providers[name] = prepare_provider_probe(
                self.config, name, SecretStr(secret) if secret else None
            )
        return providers

    def _begin_entry(self, kind: str, name: str | None = None) -> None:
        self._show(kind)
        provider = kind == "provider"
        tabs = self.provider_tabs if provider else self.mcp_tabs
        tabs.select_name(name)
        selector = "#provider-error" if provider else "#mcp-validation-status"
        self._status(selector, "")

    async def _prepare_models(self) -> None:
        providers = self._providers()
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
        providers = self._providers()
        labels = {"llm": "language model", "embedding": "embedding model"}
        statuses: dict[str, str] = {}

        def show_status(prefix: str, status: str) -> None:
            statuses[prefix] = status
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
            show_status(prefix, f"Validating {labels[prefix]}...")
            try:
                await asyncio.to_thread(_probe_model, model, providers)
            except Exception as exc:
                self.model_ok[prefix] = False
                message = _friendly(exc)
                show_status(
                    prefix,
                    f"✗ {labels[prefix].capitalize()}: {message}",
                )
                self.notify(message, severity="error", markup=False)
                return False
            self.model_ok[prefix] = True
            show_status(prefix, f"✓ {labels[prefix].capitalize()} connected")
            return True

        valid = all(await asyncio.gather(check("llm"), check("embedding")))
        if valid:
            self._begin_entry("mcp", next(iter(self.config.mcp_servers), None))
        else:
            await self._prepare_models()

    async def _validate_entry_page(self, kind: str) -> bool:
        """Validate every draft on an entry page, then commit them at once."""
        provider = kind == "provider"
        tabs = self.provider_tabs if provider else self.mcp_tabs
        other_tabs = self.mcp_tabs if provider else self.provider_tabs
        selector = "#provider-error" if provider else "#mcp-validation-status"
        candidates = tabs.candidates(other_tabs.raw_names())
        if candidates is None:
            return False
        old_users = set(self._keys())
        if not candidates:
            if provider:
                self._status(
                    selector,
                    "At least one inference provider is required.",
                    True,
                )
                return False
            self.config = replace_mcp_servers(self.config, {})
            self._discard_unused_pending(old_users)
            self._status(selector, "✓ No MCP servers configured")
            return True

        label = "providers" if provider else "MCP servers"
        self._status(selector, f"Validating {label}...")

        results = await asyncio.gather(*map(_probe_entry, candidates))
        failures: list[ProviderCandidate | McpCandidate] = []
        for candidate, result in zip(candidates, results, strict=True):
            tabs.set_draft_status(
                candidate.draft_id,
                "✓ Connected"
                if result is None
                else _friendly(result, not provider),
                marker="✓ " if result is None else "✗ ",
                failed=result is not None,
            )
            if result is not None:
                failures.append(candidate)
        if failures:
            tabs.focus_error(failures[0].draft_id)
            names = ", ".join(candidate.name for candidate in failures)
            self._status(
                selector,
                f"{'Provider' if provider else 'MCP'} validation failed: {names}",
                True,
            )
            return False

        stored = {item.name: item.stored for item in candidates}
        if provider:
            renamed = {
                item.original_name: item.name
                for item in candidates
                if item.original_name is not None
            }
            self.config = replace_inference_providers(
                self.config,
                cast(dict[str, InferenceProviderConfig], stored),
                renamed=renamed,
            )
        else:
            self.config = replace_mcp_servers(
                self.config, cast(dict[str, ServerParameters], stored)
            )
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
        editor = self.query_one("#review-full-config", TextArea)
        editor.text = format_setup_config_yaml(self.config)
        valid = bool(self.config.inference_providers and self.model_ok["llm"])
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
        await self._run_doctor(auto_advance=True)

    def _review_actions(self, state: str, valid: bool = True) -> None:
        visible = {
            "ready": {"back", "save"},
            "checked": {"recheck", "continue"},
        }.get(state, set())
        for action in ("back", "save", "recheck", "continue"):
            button = self.query_one(f"#review-{action}", Button)
            button.set_class(action not in visible, "hidden")
            button.disabled = (
                state == "running" or action == "save" and not valid
            )

    async def _run_doctor(self, *, auto_advance: bool = False) -> None:
        if self.saved_path is None:
            return
        self._review_actions("running")
        validation = self.query_one("#review-validation", Collapsible)
        results = self.query_one("#review-validation-results", Static)
        validation.title = "URSA Doctor — running…"
        validation.collapsed = False
        results.update("Running ursa self doctor…")
        report = await asyncio.to_thread(_doctor, self.saved_path)
        self.doctor_report = report
        counts = report_counts(report)
        warnings = counts[CheckStatus.WARN]
        errors = counts[CheckStatus.ERROR]
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
        if self.saved_path is None:
            return
        summary = f"Saved to {self.saved_path}"
        if self.doctor_report is not None:
            counts = report_counts(self.doctor_report)
            summary += f"\nValidation: {counts[CheckStatus.WARN]} warning(s), {counts[CheckStatus.ERROR]} error(s)"
        self.query_one("#done-copy", Static).update(summary)
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


def run_setup(
    path: Path | None = None,
    defaults_json: Path | None = None,
) -> SetupAction:
    inject_truststore_into_ssl()
    app = SetupApp(path, defaults_json)
    return app.run() or SetupAction.EXIT
