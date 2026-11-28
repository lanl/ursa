"""Read model cards, recommend a model, and run it with ModelCardAgent.

The script starts ``toy_heat_server.py`` as an MCP server, lets the agent
discover its model through the model-card tool, reads a few public
foundation-model cards from Hugging Face, and then asks for a recommendation
and a run.

Usage:
    uv run python run_example.py
    uv run python run_example.py --model openai:gpt-5-mini --offline
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

from langchain.chat_models import init_chat_model

from ursa.agents import ModelCardAgent
from ursa.util.mcp import start_mcp_client

HERE = Path(__file__).parent
PUBLIC_CARDS = [
    "hf://camlab-ethz/Poseidon-B",
    "hf://polymathic-ai/walrus",
    "hf://nvidia/fourcastnet3",
    "hf://mist-models/mist-26.9M-kkgx0omx-qm9",
]


async def main(model: str, offline: bool) -> None:
    llm = init_chat_model(model)
    client = start_mcp_client({
        "toy": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(HERE / "toy_heat_server.py")],
        }
    })
    workspace = HERE / "workspace"
    agent = ModelCardAgent(
        llm=llm, workspace=workspace, catalog_dir=workspace / "model_cards"
    )
    await agent.add_mcp_tools(client)

    prompts = []
    if not offline:
        prompts.append(
            "Read "
            + ", ".join(PUBLIC_CARDS)
            + " and describe their inputs and outputs."
        )
    prompts.append(
        "Recommend a model to simulate heat conduction along a periodic rod. "
        "I have a temperature profile in kelvin and only a laptop CPU."
    )
    profile = [
        300.0 + 40.0 * math.exp(-((((i / 128) - 0.5) / 0.05) ** 2))
        for i in range(128)
    ]
    arguments = {
        "temperature": profile,
        "diffusivity_m2_s": 1e-3,
        "time_s": 20.0,
    }
    prompts.append(f"Run toy-heat-1d with {json.dumps(arguments)}")

    for prompt in prompts:
        print(f"\n>>> {prompt[:120]}{'...' if len(prompt) > 120 else ''}\n")
        state = await agent.ainvoke(prompt)
        print(agent.format_result(state))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="openai:gpt-5-mini")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip reading cards from Hugging Face",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model, args.offline))
