import logging
from collections.abc import Iterable, Sequence
from itertools import islice
from typing import Annotated, Generic, TypeVar, final

import torch
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import (
    ArgModelBase,
    FuncMetadata,
)
from pydantic import BaseModel, ConfigDict, Field, create_model
from torch.accelerator import current_accelerator
from torch.utils.data import default_collate


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


# Type definitions for the various stages of TorchModuleTool
Input = TypeVar("In", bound=BaseModel)
Output = TypeVar("Out", bound=BaseModel)
ModelInput = TypeVar("ModelInput")
ModelOutput = TypeVar("ModelOutput")

#: Name of the Nomad-compatible tool that serves model cards.
MODEL_CARD_TOOL_NAME = "get_model_card"


def default_device():
    return current_accelerator(check_available=True) or torch.device("cpu")


class TorchModuleTool(
    BaseModel, Generic[Input, Output, ModelInput, ModelOutput]
):
    """
    A helper class for exposing a PyTorch model as an MCP tool for inference.
    Provides default methods for running the following pipeline:

    1. Preprocess a sequence of `Inputs` into a `ModelInput`
    2. Pass `ModelInput` through the PyTorch model getting `ModelOutput`
    3. Postprocess `ModelOutput` into a suitable sequence of `Outputs`

    Complex models (i.e. multi-GPU) may not be fully supported by this class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fm: torch.nn.Module
    """ The underlying PyTorch model used for inference """

    name: str = None
    """ A short name for the foundation model """

    description: str
    """ What the foundation model does / how to use it """

    args_schema: type[Input]
    """ The input schema for the model """

    output_schema: type[Output]
    """ The output_schema for the model """

    batch_size: int = 1
    """ Inputs to the model will be batched into set of at most this size """

    device: torch.device = Field(default_factory=default_device)
    """ The accelerator on which the model is placed """

    model_card: str | None = None
    """ Markdown model card (README text) served by ``get_model_card`` """

    def preprocess(self, input: Sequence[Input]) -> ModelInput:
        """
        Convert tool input into the form accepted by the model
        The input will be of type `list[args_schema]` with a length
        of `batch_size`

        Defaults to `torch.data.default_collate`
        """

        return default_collate(list(input))

    def _forward(self, model_inputs: ModelInput) -> ModelOutput:
        """Process a batch of observations with the model"""
        return self.fm(model_inputs).to("cpu")

    def postprocess(self, model_output: ModelOutput) -> Iterable[Output]:
        """Postprocess the model's raw output into a relevant tool output format"""
        yield from model_output

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs
    ) -> "TorchModuleTool":
        """Instantiate tool from a pretrained checkpoint either on disk,
        or automatically downloaded from a repository (i.e. HuggingFace)
        """
        raise NotImplementedError()

    def model_post_init(self, __context) -> None:
        # Move the model to the indicated device
        self.fm = self.fm.to(self.device)

        # Default to the class name
        if self.name is None:
            self.name = self.__class__.__name__

    @final
    def batch(self, inputs: list[Input], **kwargs) -> list[Output]:
        return list(self.batch_as_completed(inputs, **kwargs))

    @final
    def batch_as_completed(
        self,
        inputs: list[Input],
        max_concurency: int | None = None,
    ) -> Iterable[Output]:
        n = max_concurency or self.batch_size
        for batch in batched(inputs, n=n):
            from torch import inference_mode

            with inference_mode():
                batch = self.preprocess(batch)
                y = self._forward(batch)
                yield from self.postprocess(y)

    @final
    def __call__(self, input: Input):
        with torch.inference_mode():
            batch = self.preprocess([input])
            y = self._forward(batch)
            return next(iter(self.postprocess(y)))

    @final
    def __to_fastmcp(self) -> FastMCPTool:
        field_definitions = {
            field: (field_info.annotation, field_info)
            for field, field_info in self.args_schema.model_fields.items()
        }
        arg_model = create_model(
            f"{self.name}arguments",
            **field_definitions,
            __base__=ArgModelBase,
        )
        fn_metadata = FuncMetadata(
            arg_model=arg_model,
            output_model=self.output_schema,
            output_schema=self.output_schema.model_json_schema(),
        )

        async def fn(**input) -> self.output_schema:
            x = self.args_schema(**input)
            return self(x)

        return FastMCPTool(
            fn=fn,
            name=self.name,
            description=self.description,
            parameters=self.args_schema.model_json_schema(),
            fn_metadata=fn_metadata,
            is_async=True,
        )

    @final
    def add_to_fastmcp(self, server: FastMCP) -> FastMCPTool:
        """Add `self` as a tool to `server`"""
        fasttool = self.__to_fastmcp()

        if fasttool.name not in server._tool_manager._tools:
            server._tool_manager._tools[fasttool.name] = fasttool

        elif server._tool_manager.warn_on_duplicate_tools:
            logging.warning(f"Tool already exists: {fasttool.name}")

        if self.model_card:
            register_model_card(server, fasttool.name, self.model_card)

        return fasttool


def register_model_card(server: FastMCP, tool_name: str, card: str) -> None:
    """Serve ``card`` for ``tool_name`` through a shared ``get_model_card`` tool.

    The tool matches Nomad's model-card interface, so URSA's ModelCardAgent
    discovers models served here exactly as it discovers Nomad SciFMs.
    """
    cards: dict[str, str] | None = getattr(server, "_ursa_model_cards", None)
    if cards is None:
        cards = {}
        server._ursa_model_cards = cards

        def get_model_card(
            tool_name: Annotated[
                str,
                Field(
                    description="Name of the model tool to fetch the card for"
                ),
            ],
        ) -> str:
            if tool_name not in cards:
                known = ", ".join(sorted(cards)) or "none"
                raise KeyError(
                    f"No model registered with the name '{tool_name}'. "
                    f"Known models: {known}."
                )
            return cards[tool_name]

        server.add_tool(
            get_model_card,
            name=MODEL_CARD_TOOL_NAME,
            description="Return the Markdown model card for a model tool.",
        )
    cards[tool_name] = card
