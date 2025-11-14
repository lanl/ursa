import logging

from itertools import islice
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_mcp_adapters.tools import FastMCPTool, FuncMetadata, to_fastmcp
from mcp.server.fastmcp import FastMCP
from ..util.gate_optional import needs

if TYPE_CHECKING:
    import torch


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def fastmcp_add_basetool(server: FastMCP, tool: BaseTool):
    """Adds a LangChain Tool to a FastMCP Server"""
    fasttool = to_fastmcp(tool)
    if fasttool.name not in server._tool_manager._tools:
        server._tool_manager._tools[fasttool.name] = fasttool
    elif server._tool_manager.warn_on_duplicate_tools:
        logging.warning(f"Tool already exists: {fasttool.name}")
    return server._tool_manager._tools[fasttool.name]


def current_accelerator():
    from torch.accelerator import current_accelerator

    return current_accelerator()


class GenericIO(BaseModel):
    value: Any


@needs("torch", extra="fm")
class TorchModuleTool(BaseTool):
    fm: "torch.nn.Module"
    batch_size: int = 1
    device: "torch.device" = Field(default_factory=current_accelerator)

    # A Pydantic Model defining the input to fm
    # `preprocess` will get a list[args_schema] as it's input
    args_schema: type[BaseModel] = Field(default=GenericIO)

    # A Pydantic model defining the output from the fm
    # `postprocess` should yield items of this type
    output_schema: type[BaseModel] = Field(default=GenericIO)

    def _forward(self, model_inputs):
        """Call the model with the result of `preprocess`"""
        return self.fm(**model_inputs).to("cpu")

    def preprocess(self, input: Sequence):
        """Convert tool input into the form accepted by the model"""
        from torch.utils.data import default_collate

        return default_collate(list(input))

    def postprocess(self, model_output) -> Iterable:
        """Extract the tool output from the model output"""
        yield from model_output

    def batch(self, inputs: list, **kwargs):
        return list(self.batch_as_completed(inputs, **kwargs))

    def batch_as_completed(self, inputs: Sequence, **kwargs):
        for batch in batched(inputs, n=self.batch_size):
            from torch import inference_mode

            with inference_mode():
                batch = self.preprocess(batch)
                y = self._forward(batch)
                yield from self.postprocess(y)

    def _run(self, **kwargs):
        from torch import inference_mode

        with inference_mode():
            schema_instance = self.args_schema(**kwargs)
            if hasattr(schema_instance, "model_dump"):
                model_inputs = schema_instance.model_dump()
            else:
                model_inputs = schema_instance.dict()
            batch = self.preprocess([model_inputs])
            y = self._forward(batch)
            return next(iter(self.postprocess(y)))
