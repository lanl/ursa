"""Fresh-process CLI probe; the fake model is loaded only on first use."""

import json
import sys
from pathlib import Path
from time import perf_counter


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


def probe(scenario, result):
    from ursa.cli import main, runtime
    from ursa.cli.config import ChatModelConfig

    ChatModelConfig.init_chat_model = fake_model
    runtime.validate_model_provider = lambda *args: None

    def ready():
        result.write_text(json.dumps([perf_counter(), sorted(sys.modules)]))

    if scenario == "exec":

        class Output:
            responded = False

            def write(self, text):
                self.responded |= "Hello!" in text
                return sys.__stdout__.write(text)

            def flush(self):
                sys.__stdout__.flush()
                if self.responded:
                    ready()
                    self.responded = False

            def __getattr__(self, name):
                return getattr(sys.__stdout__, name)

        sys.stdout = Output()
        main(["exec", "say hello"])
    else:
        from ursa.cli.tui.app import UrsaTextualApp
        from ursa.cli.tui.widgets import PromptArea

        original_mount, original_run = (
            UrsaTextualApp.on_mount,
            UrsaTextualApp.run,
        )

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
