# pytest -q tests/mcp/test_agent_with_fake_flow.py
import ast
import json
import os
from pathlib import Path

import pytest
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from ursa.agents.execution_agent import ExecutionAgent

# ---- Tiny FakeLLM that drives a fixed plan ---------------------------------


class FakeLLM(BaseChatModel):
    """A tiny deterministic LLM that emits a fixed sequence of tool calls
    based on how many ToolMessages it sees so far.
    """

    # minimal impl to satisfy BaseChatModel
    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        msg = self.invoke(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # no-op tool binding so ExecutionAgent can call .bind_tools(...)
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # keep a handle if you want to assert on it in tests
        self._bound_tools = tools
        return self

    def _count_tools(self, messages):
        return sum(1 for m in messages if isinstance(m, ToolMessage))

    @property
    def _llm_type(self) -> str:
        return "fake"

    def invoke(self, messages, **kwargs):
        # Very small state machine keyed on how many tools have replied
        tool_msgs = self._count_tools(messages)

        # helper: fetch blob_ref from the ToolMessage that replied to a given tool_call_id
        def _blob_ref_for(call_id: str):
            # Walk newest -> oldest to find the ToolMessage answering `call_id`
            for m in reversed(messages):
                if (
                    isinstance(m, ToolMessage)
                    and getattr(m, "tool_call_id", None) == call_id
                ):
                    c = m.content
                    # dict content (best case)
                    if isinstance(c, dict):
                        if "blob_ref" in c:
                            return c["blob_ref"]
                        if (
                            isinstance(c.get("result"), dict)
                            and "blob_ref" in c["result"]
                        ):
                            return c["result"]["blob_ref"]
                    # string content -> try JSON, then safe python-literal parse
                    if isinstance(c, str):
                        # JSON first
                        try:
                            obj = json.loads(c)
                            if isinstance(obj, dict):
                                if "blob_ref" in obj:
                                    return obj["blob_ref"]
                                if (
                                    isinstance(obj.get("result"), dict)
                                    and "blob_ref" in obj["result"]
                                ):
                                    return obj["result"]["blob_ref"]
                        except Exception:
                            pass
                        # single-quoted dicts, etc.
                        try:
                            obj = ast.literal_eval(c)
                            if isinstance(obj, dict):
                                if "blob_ref" in obj:
                                    return obj["blob_ref"]
                                if (
                                    isinstance(obj.get("result"), dict)
                                    and "blob_ref" in obj["result"]
                                ):
                                    return obj["result"]["blob_ref"]
                        except Exception:
                            pass
            return None

        # Phase 0: propose fetching three files (unchanged)
        if tool_msgs == 0:
            return AIMessage(
                content="Fetch three files via MCP, then write them.",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "fs_get_object_bytes",
                        "args": {
                            "root": "FIXTURE_ROOT",
                            "key": "tiny.txt",
                            "allow_large": True,
                        },
                    },
                    {
                        "id": "c2",
                        "name": "fs_get_object_bytes",
                        "args": {
                            "root": "FIXTURE_ROOT",
                            "key": "small.csv",
                            "allow_large": True,
                        },
                    },
                    {
                        "id": "c3",
                        "name": "fs_get_object_bytes",
                        "args": {
                            "root": "FIXTURE_ROOT",
                            "key": "tiny.bin",
                            "allow_large": True,
                        },
                    },
                ],
            )

        # Phase 1: now use real blob_refs from prior ToolMessages
        if tool_msgs == 3:
            b1 = _blob_ref_for("c1")
            b2 = _blob_ref_for("c2")
            b3 = _blob_ref_for("c3")
            return AIMessage(
                content="Write bytes to data/.",
                tool_calls=[
                    {
                        "id": "w1",
                        "name": "write_bytes",
                        "args": {
                            "filename": "data/tiny.txt",
                            "blob_ref": b1,
                            "allow_text": True,
                        },
                    },
                    {
                        "id": "w2",
                        "name": "write_bytes",
                        "args": {
                            "filename": "data/small.csv",
                            "blob_ref": b2,
                            "allow_text": True,
                        },
                    },
                    {
                        "id": "w3",
                        "name": "write_bytes",
                        "args": {"filename": "data/tiny.bin", "blob_ref": b3},
                    },
                ],
            )

        # Phase 2 (use real r1 blob_ref when appending)
        if tool_msgs == 6:
            return AIMessage(
                content="Read 5 bytes and append.",
                tool_calls=[
                    {
                        "id": "r1",
                        "name": "fs_read_range_bytes",
                        "args": {
                            "root": "FIXTURE_ROOT",
                            "key": "tiny.txt",
                            "start": 0,
                            "length": 5,
                        },
                    },
                ],
            )

        if tool_msgs == 7:
            br = _blob_ref_for("r1")
            return AIMessage(
                content="Append chunk to data/tiny.txt",
                tool_calls=[
                    {
                        "id": "a1",
                        "name": "append_bytes",
                        "args": {
                            "filename": "data/tiny.txt",
                            "blob_ref": br,
                            "expected_offset": 0,
                        },
                    },
                ],
            )

        return AIMessage(content="Done.")


# ---- Fixtures: make some local files for the localfs server ----------------


@pytest.fixture
def fixtures_dir(tmp_path):
    d = tmp_path / "fixtures"
    d.mkdir()
    (d / "tiny.txt").write_text("hello world", encoding="utf-8")
    (d / "small.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (d / "tiny.bin").write_bytes(b"\x00\x01\x02")
    return str(d)


# ---- Local tools that mimic the MCP server (no subprocess needed) ----------


def _fs_get_object_bytes(root: str, key: str, allow_large: bool = False):
    p = Path(root) / key
    b = p.read_bytes()
    import base64

    return {
        "root": root,
        "key": key,
        "size": len(b),
        "bytes_b64": base64.b64encode(b).decode("ascii"),
    }


def _fs_read_range_bytes(root: str, key: str, start: int, length: int):
    p = Path(root) / key
    b = p.read_bytes()
    end = min(len(b), start + length)
    chunk = b[start:end] if 0 <= start < len(b) else b""
    import base64

    return {
        "root": root,
        "key": key,
        "start": start,
        "size": len(chunk),
        "bytes_b64": base64.b64encode(chunk).decode("ascii"),
    }


def _fs_head_object(root: str, key: str):
    p = Path(root) / key
    return {"root": root, "key": key, "size": p.stat().st_size, "metadata": {}}


def _make_localfs_tools():
    return [
        StructuredTool.from_function(
            func=_fs_get_object_bytes,
            name="fs_get_object_bytes",
            description="Get file as bytes_b64",
        ),
        StructuredTool.from_function(
            func=_fs_read_range_bytes,
            name="fs_read_range_bytes",
            description="Read a byte range, returns bytes_b64",
        ),
        StructuredTool.from_function(
            func=_fs_head_object,
            name="fs_head_object",
            description="Head-like metadata",
        ),
    ]


@pytest.mark.asyncio
async def test_fake_flow_works(tmp_path, fixtures_dir, monkeypatch):
    # Patch our FakeLLM to substitute FIXTURE_ROOT in outgoing tool args
    class FakeLLMWithRoot(FakeLLM):
        def invoke(self, messages, config=None, **kwargs):
            # call parent with config intact
            msg = super().invoke(messages, config=config, **kwargs)

            # helper: parse ToolMessage.content (stringified dict) and pull blob_ref
            def _blob_ref_for(tool_call_id: str):
                for m in reversed(messages):
                    if (
                        isinstance(m, ToolMessage)
                        and getattr(m, "tool_call_id", None) == tool_call_id
                    ):
                        c = m.content
                        if isinstance(c, dict):
                            if "blob_ref" in c:
                                return c["blob_ref"]
                            if (
                                isinstance(c.get("result"), dict)
                                and "blob_ref" in c["result"]
                            ):
                                return c["result"]["blob_ref"]
                        if isinstance(c, str):
                            # try JSON then safe python literal (handles single-quoted dicts)
                            try:
                                obj = json.loads(c)
                            except Exception:
                                try:
                                    obj = ast.literal_eval(c)
                                except Exception:
                                    obj = None
                            if isinstance(obj, dict):
                                if "blob_ref" in obj:
                                    return obj["blob_ref"]
                                if (
                                    isinstance(obj.get("result"), dict)
                                    and "blob_ref" in obj["result"]
                                ):
                                    return obj["result"]["blob_ref"]
                return None

            # rewrite FIXTURE_ROOT -> actual fixtures_dir in any pending tool args
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    args = tc.get("args", {})
                    if args.get("root") == "FIXTURE_ROOT":
                        args["root"] = fixtures_dir

                # If we're at the "write_bytes" phase, fill blob_refs from c1/c2/c3
                # Your FakeLLM (base) triggers this after 3 ToolMessages.
                # Map writes in order to earlier fetches (c1, c2, c3).
                write_calls = [
                    tc
                    for tc in msg.tool_calls
                    if tc.get("name") == "write_bytes"
                ]
                if write_calls:
                    ids_in_order = ["c1", "c2", "c3"]
                    refs = [_blob_ref_for(i) for i in ids_in_order]
                    # fill any None left-to-right with whatever we found
                    j = 0
                    for tc in write_calls:
                        a = tc.setdefault("args", {})
                        if not a.get("blob_ref"):
                            # advance j to next non-None ref
                            while j < len(refs) and refs[j] is None:
                                j += 1
                            if j < len(refs):
                                a["blob_ref"] = refs[j]
                                j += 1

                # If we're at the "append_bytes" phase, fill from r1
                append_calls = [
                    tc
                    for tc in msg.tool_calls
                    if tc.get("name") == "append_bytes"
                ]
                if append_calls:
                    rref = _blob_ref_for("r1")
                    for tc in append_calls:
                        a = tc.setdefault("args", {})
                        if not a.get("blob_ref"):
                            a["blob_ref"] = rref

            return msg

        # satisfy BaseChatModel abstract method (keeps test self-contained)
        def _generate(self, messages, stop=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult

            m = self.invoke(messages, **kwargs)
            return ChatResult(generations=[ChatGeneration(message=m)])

        # no-op tool binding so ExecutionAgent can call .bind_tools(...)
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            self._bound_tools = tools
            return self

    ws = tmp_path / "ws"
    os.makedirs(ws, exist_ok=True)

    agent = ExecutionAgent(
        llm=FakeLLMWithRoot(),
        log_state=False,
        tool_log=False,
        extra_tools=_make_localfs_tools(),
    )

    # Drive a tiny conversation
    state = {
        "messages": [
            SystemMessage(content="You are test agent"),
            HumanMessage(content="please fetch & write files"),
        ],
        "workspace": str(ws),
        "symlinkdir": {},
        "current_progress": "",
        "code_files": [],
    }

    await agent.ainvoke(state)

    # Verify artifacts landed
    assert (ws / "data/tiny.txt").exists()
    assert (ws / "data/small.csv").exists()
    assert (ws / "data/tiny.bin").exists()

    # After the append, tiny.txt should start with the 5 bytes
    assert (
        (ws / "data/tiny.txt").read_text(encoding="utf-8").startswith("hello")
    )
