# scripts/mcp/run_local_mcp_agent.py
import ast
import asyncio
import json
import os
from pathlib import Path

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import PrivateAttr

from ursa.agents.execution_agent import ExecutionAgent


class MinimalPlanLLM(BaseChatModel):
    _fixtures_root: str = PrivateAttr()

    def __init__(self, fixtures_root: str):
        super().__init__()
        self._fixtures_root = fixtures_root

    @property
    def _llm_type(self) -> str:
        return "minimal-plan"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self._tools = tools
        return self

    def _count_tool_msgs(self, messages) -> int:
        return sum(isinstance(m, ToolMessage) for m in messages)

    def _parse_payload(self, content):
        # langchain-mcp often returns [json_str, None]; also sometimes dict
        if isinstance(content, list) and content:
            content = content[0]
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            obj = None
            try:
                obj = json.loads(content)
            except Exception:
                try:
                    obj = ast.literal_eval(content)
                except Exception:
                    obj = None
            if isinstance(obj, dict):
                return obj
        return {}

    def _last_tool_payloads(self, messages, n):
        out = []
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                out.append(self._parse_payload(m.content))
                if len(out) == n:
                    break
        out.reverse()
        return out

    def _pick_blob_or_b64(self, payload):
        # prefer blob_ref (if your agent’s sanitizer added it); else bytes_b64
        # supports top-level or under "result"
        for d in (payload, payload.get("result", {})):
            if not isinstance(d, dict):
                continue
            if "blob_ref" in d and d["blob_ref"]:
                return {"blob_ref": d["blob_ref"]}
            if "bytes_b64" in d and d["bytes_b64"]:
                return {"data_b64": d["bytes_b64"]}
        return {}

    def invoke(self, messages, config=None, **kwargs):
        tool_msgs = self._count_tool_msgs(messages)
        r = self._fixtures_root

        # Phase 0: fetch three files from localfs MCP
        if tool_msgs == 0:
            return AIMessage(
                content="Fetch three, then write them out.",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "fs_get_object_bytes",
                        "args": {
                            "root": r,
                            "key": "tiny.txt",
                            "allow_large": True,
                        },
                    },
                    {
                        "id": "c2",
                        "name": "fs_get_object_bytes",
                        "args": {
                            "root": r,
                            "key": "small.csv",
                            "allow_large": True,
                        },
                    },
                    {
                        "id": "c3",
                        "name": "fs_get_object_bytes",
                        "args": {
                            "root": r,
                            "key": "tiny.bin",
                            "allow_large": True,
                        },
                    },
                ],
            )

        # Phase 1: after 3 ToolMessages, write using blob_ref OR data_b64 from last 3 tool msgs
        if tool_msgs == 3:
            p1, p2, p3 = self._last_tool_payloads(messages, 3)
            a1 = self._pick_blob_or_b64(p1)
            a2 = self._pick_blob_or_b64(p2)
            a3 = self._pick_blob_or_b64(p3)
            return AIMessage(
                content="Write fetched files into data/.",
                tool_calls=[
                    {
                        "id": "w1",
                        "name": "write_bytes",
                        "args": {
                            "filename": "data/tiny.txt",
                            **a1,
                            "allow_text": True,
                        },
                    },
                    {
                        "id": "w2",
                        "name": "write_bytes",
                        "args": {
                            "filename": "data/small.csv",
                            **a2,
                            "allow_text": True,
                        },
                    },
                    {
                        "id": "w3",
                        "name": "write_bytes",
                        "args": {"filename": "data/tiny.bin", **a3},
                    },
                ],
            )

        # Phase 2: read a range and append it to tiny.txt
        if tool_msgs == 6:
            return AIMessage(
                content="Read 5 bytes then append to tiny.txt.",
                tool_calls=[
                    {
                        "id": "r1",
                        "name": "fs_read_range_bytes",
                        "args": {
                            "root": r,
                            "key": "tiny.txt",
                            "start": 0,
                            "length": 5,
                        },
                    },
                ],
            )

        if tool_msgs == 7:
            # take last tool payload (the range read)
            (p,) = self._last_tool_payloads(messages, 1)
            a = self._pick_blob_or_b64(p)
            return AIMessage(
                content="Append the chunk to data/tiny.txt",
                tool_calls=[
                    {
                        "id": "a1",
                        "name": "append_bytes",
                        "args": {
                            "filename": "data/tiny.txt",
                            **a,
                            "expected_offset": 0,
                        },
                    },
                ],
            )

        # Done
        return AIMessage(content="Done.")

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        msg = self.invoke(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=msg)])


async def main():
    ws = "workspaces/localfs_mcp_demo"
    os.makedirs(ws, exist_ok=True)

    # Seed demo fixtures (these are what the MCP server will serve)
    fixtures = Path("tests/mcp/fixtures").resolve()
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "tiny.txt").write_text("hello world", encoding="utf-8")
    (fixtures / "small.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (fixtures / "tiny.bin").write_bytes(b"\x00\x01\x02")

    # Configure an MCP stdio server (spawned automatically by the adapter)
    client = MultiServerMCPClient({
        "localfs": {
            "transport": "stdio",
            "command": "python",
            "args": [
                "scripts/mcp/localfs_mcp_server.py",
                "--root",
                str(fixtures),
                "--stdio",
            ],
        }
    })

    tools = await client.get_tools()

    # Wire up the agent with our tiny planner and the MCP tools
    agent = ExecutionAgent(
        llm=MinimalPlanLLM(str(fixtures)),
        extra_tools=tools,
        tool_log=True,
        log_state=False,
    )

    state = {
        "messages": [
            SystemMessage(content="You are a test agent"),
            HumanMessage(content="download and write"),
        ],
        "workspace": ws,
        "symlinkdir": {},
        "current_progress": "",
        "code_files": [],
    }

    await agent.ainvoke(state)
    print("Done. Files in:", ws)
    for p in sorted(Path(ws).glob("data/*")):
        print(" -", p, f"({p.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
