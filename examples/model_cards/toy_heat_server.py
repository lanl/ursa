"""Serve a small physics model over MCP with a Nomad-compatible model card.

The model is an exact spectral solver of the 1D heat equation on a periodic
domain, wrapped in URSA's ``TorchModuleTool``. It is not a learned model; it
stands in for a scientific foundation model so the full discover, recommend,
and run loop of ``ModelCardAgent`` can be tried on a laptop. Because the
solution is exact, every output can be checked against theory.

Run it as an MCP server over stdio (URSA starts it for you from a config):

    python toy_heat_server.py
"""

from collections.abc import Iterable, Sequence

import torch
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from ursa.tools.fm_base_tool import TorchModuleTool

MODEL_CARD = """---
license: bsd-3-clause
tags:
- project:ursa-example
- type:model
- science:physics-simulation
- risk:general
pipeline_tag: other
---

# Toy Heat Propagator 1D

Toy Heat Propagator 1D advances a temperature profile on a periodic 1D rod by
solving the heat equation exactly in Fourier space. It is an analytic solver
wrapped as a model tool, not a learned model, and serves as a stand-in for a
heat transfer surrogate when testing agent workflows.

## Model short description

Exact spectral surrogate for 1D heat conduction on a periodic domain.

## Inputs and outputs

- Input `temperature`: temperature samples in kelvin on a uniform periodic grid, 8 to 4096 points.
- Input `diffusivity_m2_s`: thermal diffusivity in units of m^2/s.
- Input `time_s`: elapsed time in seconds.
- Input `length_m`: domain length in meters (default 1.0).
- Output `temperature`: the temperature profile after `time_s`, in kelvin, on the input grid.

## Compute Infrastructure

### Hardware

Inference can run on CPU or a single GPU.

# Intended Uses

## Intended Use

Demonstrating model-card discovery, recommendation, and MCP execution in URSA.

## Out-of-Scope Use Cases

Non-periodic boundaries, variable diffusivity, or any production heat transfer analysis.

# Limitations

## Limitations

- Assumes constant diffusivity and periodic boundaries.
- Accuracy is limited only by floating-point round-off.
"""


class HeatInput(BaseModel):
    temperature: list[float] = Field(
        description="Temperature samples in kelvin on a uniform periodic 1D grid.",
        min_length=8,
        max_length=4096,
    )
    diffusivity_m2_s: float = Field(
        gt=0, description="Thermal diffusivity in m^2/s."
    )
    time_s: float = Field(ge=0, description="Elapsed time in seconds.")
    length_m: float = Field(
        default=1.0, gt=0, description="Domain length in meters."
    )


class HeatOutput(BaseModel):
    temperature: list[float] = Field(
        description="Temperature in kelvin after time_s, on the input grid."
    )
    time_s: float = Field(description="Elapsed time in seconds.")
    mean_temperature_k: float = Field(
        description="Spatial mean temperature in kelvin (conserved)."
    )


class SpectralHeatPropagator(torch.nn.Module):
    """u(t) = IFFT(FFT(u0) * exp(-alpha k^2 t)) on a periodic grid."""

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        u0 = batch["temperature"]
        n = u0.shape[-1]
        spacing = batch["length_m"] / n
        # rfftfreq defaults to float32; keep the wavenumbers in the input's
        # precision so float64 profiles stay exact to round-off.
        k = (
            2
            * torch.pi
            * torch.fft.rfftfreq(n, dtype=u0.dtype, device=u0.device)
        )
        k = k / spacing.unsqueeze(-1)
        decay = torch.exp(
            -batch["diffusivity_m2_s"].unsqueeze(-1)
            * k**2
            * batch["time_s"].unsqueeze(-1)
        )
        return torch.fft.irfft(torch.fft.rfft(u0) * decay, n=n)


class HeatTool(TorchModuleTool[HeatInput, HeatOutput, dict, torch.Tensor]):
    args_schema: type[HeatInput] = HeatInput
    output_schema: type[HeatOutput] = HeatOutput

    @classmethod
    def from_pretrained(cls, name_or_path: str = "toy-heat-1d", **kwargs):
        return cls(
            name=name_or_path,
            description=(
                "Advance a periodic 1D temperature profile (kelvin) by time_s "
                "seconds for a given thermal diffusivity (m^2/s)."
            ),
            fm=SpectralHeatPropagator(),
            model_card=MODEL_CARD,
            device=torch.device("cpu"),
        )

    def preprocess(self, inputs: Sequence[HeatInput]) -> dict:
        # One profile per call; profiles may differ in length.
        (item,) = inputs
        as_tensor = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
        return {
            "temperature": torch.tensor(
                [item.temperature], dtype=torch.float64
            ),
            "diffusivity_m2_s": as_tensor(item.diffusivity_m2_s),
            "time_s": as_tensor(item.time_s),
            "length_m": as_tensor(item.length_m),
            "inputs": inputs,
        }

    def _forward(self, model_inputs: dict) -> dict:
        model_inputs["output"] = self.fm(model_inputs).cpu()
        return model_inputs

    def postprocess(self, model_output: dict) -> Iterable[HeatOutput]:
        for item, row in zip(model_output["inputs"], model_output["output"]):
            yield HeatOutput(
                temperature=[float(v) for v in row],
                time_s=item.time_s,
                mean_temperature_k=float(row.mean()),
            )


def build_server() -> FastMCP:
    server = FastMCP("toy-physics-models")
    HeatTool.from_pretrained().add_to_fastmcp(server)
    return server


if __name__ == "__main__":
    build_server().run()
