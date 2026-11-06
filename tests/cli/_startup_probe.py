"""Fresh-process probe for CLI import deferral regression tests."""

import json
import os
import sys
from pathlib import Path


def fake_model(self):
    from langchain_core.language_models.fake_chat_models import (
        FakeListChatModel,
    )
    from langchain_core.utils.function_calling import convert_to_openai_tool

    class Model(FakeListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self.bind(tools=[convert_to_openai_tool(t) for t in tools])

    return Model(
        responses=["Hello!"],
        custom_get_token_ids=lambda text: list(range(len(text.split()))),
    )


def probe(scenario: str, result: Path) -> None:
    from ursa.util import crossplatform

    config_path = Path(os.environ["XDG_CONFIG_HOME"]) / "ursa" / "config.yaml"
    crossplatform.user_config_paths = lambda: [config_path]
    from ursa.cli import main, runtime
    from ursa.cli.config import ChatModelConfig

    ChatModelConfig.init_chat_model = fake_model
    runtime.validate_model_provider = lambda *args: None

    def ready() -> None:
        result.write_text(json.dumps(sorted(sys.modules)))

    if scenario == "exec":
        main(["exec", "say hello"])
        ready()
        return

    from ursa.cli.tui.app import UrsaTextualApp
    from ursa.cli.tui.widgets import PromptArea

    original_mount, original_run = UrsaTextualApp.on_mount, UrsaTextualApp.run

    def focused(app):
        prompt = app.query_one(PromptArea)
        assert app.focused is prompt and not prompt.disabled
        prompt.insert("hello")
        assert prompt.text == "hello"
        ready()
        app.exit()

    def mounted(app):
        original_mount(app)
        app.call_after_refresh(focused, app)

    UrsaTextualApp.on_mount = mounted
    UrsaTextualApp.run = lambda app: original_run(app, headless=True)
    main([])


if __name__ == "__main__":
    probe(sys.argv[1], Path(sys.argv[2]))
