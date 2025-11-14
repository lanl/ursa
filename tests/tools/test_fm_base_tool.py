import logging
from types import SimpleNamespace

import pytest

from ursa.tools import fm_base_tool

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None  # type: ignore[assignment]

requires_torch = pytest.mark.skipif(
    torch is None, reason="PyTorch is not installed"
)


if torch is not None:

    class DummyTorchModuleTool(fm_base_tool.TorchModuleTool):
        fm: object
        name: str = "dummy_torch_tool"
        description: str = "Test harness for TorchModuleTool"

        def postprocess(self, model_output):
            if isinstance(model_output, torch.Tensor):
                for item in model_output:
                    yield item.item()
                return
            yield from super().postprocess(model_output)

        def _run(self, *args, **kwargs):
            return super()._run(**kwargs)

        async def _arun(self, *args, **kwargs):
            raise NotImplementedError

    class DoubleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls: list[list[int]] = []

        def forward(self, value):
            self.forward_calls.append(value.detach().cpu().tolist())
            return value * 2

else:
    DummyTorchModuleTool = None  # type: ignore[assignment]
    DoubleModule = None  # type: ignore[assignment]


def test_batched_chunks_iterables():
    result = list(fm_base_tool.batched([1, 2, 3, 4, 5], 2))
    assert result == [(1, 2), (3, 4), (5,)]


def test_batched_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        next(fm_base_tool.batched([1, 2], 0))


class DummyToolManager:
    def __init__(self, warn_on_duplicates=False):
        self._tools = {}
        self.warn_on_duplicate_tools = warn_on_duplicates


class DummyServer:
    def __init__(self, warn_on_duplicates=False):
        self._tool_manager = DummyToolManager(warn_on_duplicates)


def test_fastmcp_add_basetool_registers_new_tool(monkeypatch):
    server = DummyServer()
    expected_tool = SimpleNamespace(name="unit-test-tool")
    input_tool = object()

    def fake_to_fastmcp(tool):
        assert tool is input_tool
        return expected_tool

    monkeypatch.setattr(fm_base_tool, "to_fastmcp", fake_to_fastmcp)

    result = fm_base_tool.fastmcp_add_basetool(server, input_tool)

    assert result is expected_tool
    assert server._tool_manager._tools["unit-test-tool"] is expected_tool


def test_fastmcp_add_basetool_warns_on_duplicate(monkeypatch, caplog):
    server = DummyServer(warn_on_duplicates=True)
    existing_tool = SimpleNamespace(name="duplicate-tool")
    server._tool_manager._tools["duplicate-tool"] = existing_tool

    caplog.set_level(logging.WARNING)

    replacement_tool = SimpleNamespace(name="duplicate-tool")
    monkeypatch.setattr(
        fm_base_tool, "to_fastmcp", lambda tool: replacement_tool
    )

    result = fm_base_tool.fastmcp_add_basetool(server, object())

    assert result is existing_tool
    assert f"Tool already exists: {replacement_tool.name}" in caplog.messages


@requires_torch
def test_batch_as_completed_processes_batches():
    assert (
        torch is not None
        and DummyTorchModuleTool is not None
        and DoubleModule is not None
    )  # for static type checkers
    model = DoubleModule()
    tool = DummyTorchModuleTool(
        fm=model, batch_size=2, device=torch.device("cpu")
    )

    outputs = list(
        tool.batch_as_completed([{"value": 1}, {"value": 2}, {"value": 3}])
    )

    assert outputs == [2, 4, 6]
    assert model.forward_calls == [[1, 2], [3]]


@requires_torch
def test_batch_returns_list():
    assert (
        torch is not None
        and DummyTorchModuleTool is not None
        and DoubleModule is not None
    )  # for static type checkers
    model = DoubleModule()
    tool = DummyTorchModuleTool(
        fm=model, batch_size=2, device=torch.device("cpu")
    )

    outputs = tool.batch([{"value": 4}, {"value": 5}])

    assert outputs == [8, 10]
    assert model.forward_calls == [[4, 5]]


@requires_torch
def test_invoke_returns_single_prediction():
    assert (
        torch is not None
        and DummyTorchModuleTool is not None
        and DoubleModule is not None
    )  # for static type checkers
    model = DoubleModule()
    tool = DummyTorchModuleTool(fm=model, device=torch.device("cpu"))

    output = tool.invoke({"value": 10})

    assert output == 20
    assert model.forward_calls == [[10]]
