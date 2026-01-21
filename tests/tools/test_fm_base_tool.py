import random
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ursa.tools.fm_base_tool import TorchModuleTool


@pytest.fixture(scope="function")
def xor_tool():
    try:
        import torch
        from torch import nn

    except ImportError:
        pytest.skip("torch is not installed")

    from ursa.tools.fm_base_tool import TorchModuleTool

    class TwoThings(BaseModel):
        thing1: bool = Field(description="If Thing-1 is a Cat")
        thing2: bool
        """ If Thing 2 is a Hat (Not in tool def)"""

    class XOrResult(BaseModel):
        result: bool

    class XOrTool(TorchModuleTool):
        args_schema: type[BaseModel] = Field(TwoThings)
        output_schema: type[BaseModel] = Field(XOrResult)

        @classmethod
        def from_pretrained(cls):
            fm = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
            )
            return XOrTool(
                name="XOrTool",
                description="Compute XOR with a neural network",
                fm=fm,
            )

        def preprocess(self, input: Sequence):
            obs = torch.stack([
                torch.tensor([x.thing1, x.thing2], dtype=float) for x in input
            ])
            return obs.to(self.device, dtype=torch.float32)

        def postprocess(self, model_output) -> Iterable:
            for obs in model_output:
                yield self.output_schema(result=obs.item() >= 0)

    return XOrTool.from_pretrained()


def test_inference(xor_tool):
    out = xor_tool(
        xor_tool.args_schema(
            thing1=random.random() > 0.5, thing2=random.random() > 0.5
        )
    )
    assert out.__class__.__name__ == "XOrResult"


def test_batch(xor_tool):
    batch = [
        xor_tool.args_schema(
            thing1=random.random() > 0.5,
            thing2=random.random() > 0.5,
        )
        for _ in range(32)
    ]
    out = xor_tool.batch(batch)
    assert len(out) == len(batch)


def test_batch_as_completed(xor_tool):
    batch = [
        xor_tool.args_schema(
            thing1=random.random() > 0.5,
            thing2=random.random() > 0.5,
        )
        for _ in range(32)
    ]
    out = xor_tool.batch_as_completed(batch)
    assert isinstance(out, Iterable)
    assert len(list(out)) == len(batch)


async def test_add_to_fastmcp(xor_tool: "TorchModuleTool"):
    server = FastMCP()
    tools = await server.list_tools()
    assert len(tools) == 0
    tool = xor_tool.add_to_fastmcp(server)
    tools = await server.list_tools()
    assert len(tools) == 1

    # Check metadata
    tool.fn_metadata.arg_model is not None
    input_args = tool.fn_metadata.arg_model.model_fields

    # Check arg_model matches
    assert "thing1" in input_args
    assert input_args["thing1"].description is not None
    assert "thing2" in input_args
    assert input_args["thing2"].description is None


@pytest.fixture
async def xor_mcp_client(xor_tool: "TorchModuleTool"):
    mcp_server = FastMCP()
    xor_tool.add_to_fastmcp(mcp_server)
    async with Client(transport=mcp_server) as mcp_client:
        tools = await mcp_client.list_tools()
        print(tools)
        assert len(tools) == 1
        yield mcp_client


async def test_client(xor_mcp_client: Client[FastMCPTransport]):
    result = await xor_mcp_client.call_tool(
        "XOrTool", arguments={"thing1": True, "thing2": False}
    )
    assert result.data is not None
