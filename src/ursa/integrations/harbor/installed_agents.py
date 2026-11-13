"""Harbor agents that reuse a selected, prebuilt Singularity install layer."""

from harbor.agents.installed.claude_code import ClaudeCode as _ClaudeCode
from harbor.agents.installed.codex import Codex as _Codex
from harbor.agents.installed.goose import Goose as _Goose

from ursa.integrations.harbor.singularity import (
    DockerfileSingularityEnvironment,
)


class _CachedInstall:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._requested_install_version = self._version

    async def install(self, environment):
        if isinstance(environment, DockerfileSingularityEnvironment):
            spec = environment._agent_spec
            if spec is not None:
                name, _, requested_version = spec.partition("@")
                if (
                    name != self.name()
                    or (requested_version or None)
                    != self._requested_install_version
                ):
                    raise ValueError(
                        f"Selected install layer {spec!r} does not match "
                        f"{self.name()} version {self._requested_install_version!r}"
                    )
                if environment._agent_layer_path is None:
                    raise RuntimeError(
                        "The selected agent install layer is not loaded"
                    )
                return
        await super().install(environment)

    async def run(self, instruction, environment, context):
        original_url = self._base_url
        try:
            if isinstance(environment, DockerfileSingularityEnvironment):
                endpoint = self.model_connection.execution_base_url
                if endpoint:
                    forwarded = await environment._forward_model_url(endpoint)
                    if forwarded != endpoint:
                        self._base_url = forwarded
            return await super().run(instruction, environment, context)
        finally:
            self._base_url = original_url


class Goose(_CachedInstall, _Goose):
    pass


class ClaudeCode(_CachedInstall, _ClaudeCode):
    pass


class Codex(_CachedInstall, _Codex):
    pass
