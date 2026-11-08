"""Measure a variable agent population on local or cluster workers.

Run from the repository root with ``python -m examples.runtime_workers.scaling``.
Real model runs deliberately require Nemotron 3.5 Super; --transport-only measures
placement and messaging without making model requests.
"""

import argparse
import asyncio
import json
import os
import socket
import threading
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from ursa.cli.config import UrsaConfig, resolve_ursa_config
from ursa.runtime import AgentManager, AgentSpec
from ursa.util.secrets import SecretReference

MODEL = "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
MODEL_ALIASES = {MODEL, "nemotron-3.5-super"}


class ScalingAgent:
    def __init__(self, config, context):
        self.context = context
        self.state = {"calls": 0, "messages": []}
        self.agent = None
        if config["model"] is not None:
            from ursa.agents.chat_agent import BasicChatAgent

            self.agent = BasicChatAgent(
                llm=config["model"].init_chat_model(),
                enable_metrics=False,
                workspace=config["workspace"],
            )

    async def ainvoke(self, prompt, config=None):
        self.state["calls"] += 1
        started = perf_counter()
        if self.agent is None:
            await asyncio.sleep(0.02)
            reply = "transport check"
        else:
            query = await asyncio.to_thread(self.agent.format_query, prompt)
            result = await self.agent.ainvoke(query)
            reply = self.agent.format_result(result)
            if not reply.strip():
                raise RuntimeError(
                    "Nemotron returned an empty reply; check the token limit"
                )
        return {
            "reply": reply,
            "calls": self.state["calls"],
            "seconds": perf_counter() - started,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "thread": threading.get_ident(),
        }

    async def on_message(self, message):
        self.state["messages"].append({
            "sender": message.sender,
            "payload": message.payload,
        })

    async def received(self):
        return self.state["messages"]

    async def aclose(self):
        if self.agent is not None:
            await self.agent.aclose()
            self.agent.close()


async def measure(args):
    config = resolve_ursa_config(UrsaConfig.from_file(args.config))
    model = None
    if not args.transport_only:
        model = config.llm_model
        if model.model not in MODEL_ALIASES:
            raise ValueError(f"This benchmark requires {MODEL}")
        if isinstance(model.api_key, SecretReference):
            model = model.model_copy(
                update={"api_key": model.api_key.resolve()}
            )
    records = []
    active = 0
    async with AgentManager(
        runtime=config.runtime, startup_timeout=180
    ) as manager:
        for population in args.agents:
            for index in range(population, active):
                await manager.unregister(f"agent-{index}", force=True)
            for index in range(active, population):
                await manager.register(
                    AgentSpec(
                        f"agent-{index}",
                        "examples.runtime_workers.scaling:ScalingAgent",
                        {
                            "model": model,
                            "workspace": config.workspace / f"agent-{index}",
                        },
                    )
                )
            active = population
            started = perf_counter()
            results = await asyncio.gather(
                *(
                    manager.invoke(
                        f"agent-{index}",
                        f"Tell one short, original science joke, numbered {index}. Do not use tools.",
                    )
                    for index in range(population)
                )
            )
            for index in range(population):
                await manager.send_message(
                    f"agent-{index}", f"wave-{population}", sender="coordinator"
                )
            assert all(
                messages[-1]
                == {"sender": "coordinator", "payload": f"wave-{population}"}
                for messages in await asyncio.gather(
                    *(
                        manager.call(f"agent-{index}", "received")
                        for index in range(population)
                    )
                )
            )
            record = {
                "agents": population,
                "seconds": perf_counter() - started,
                "workers": [
                    asdict(worker) for worker in manager.worker_snapshots()
                ],
                "results": results,
            }
            records.append(record)
            args.output.write_text(json.dumps(records, indent=2))
            print(
                json.dumps({
                    "agents": population,
                    "seconds": record["seconds"],
                    "workers": len(record["workers"]),
                }),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--agents", type=int, nargs="+", default=[4, 16, 32, 4])
    parser.add_argument("--transport-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("scaling-results.json")
    )
    args = parser.parse_args()
    if any(count < 1 for count in args.agents):
        parser.error("agent populations must be positive")
    asyncio.run(measure(args))


if __name__ == "__main__":
    main()
