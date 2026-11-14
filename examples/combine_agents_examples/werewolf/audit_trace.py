"""Audit actual model tool batches and peer messages from a shared checkpoint.

Run with ``uv run .../audit_trace.py CHECKPOINT_DB THREAD_ID --output audit.json``.
Reads all available checkpoints; --merge retains evidence across later pruning.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def audit(database: Path, thread_id: str, previous: dict | None = None) -> dict:
    if previous is not None and previous["thread_id"] != thread_id:
        raise ValueError("Cannot merge audits of different game threads")
    previous = previous or {}
    serde = JsonPlusSerializer()
    batches = list(previous.get("tool_batches", []))
    sends = list(previous.get("messages", []))
    seen = {
        (batch["agent"], tuple(call["id"] for call in batch["calls"]))
        for batch in batches
    }
    responses = {}
    agents = set(previous.get("agents", []))
    with closing(
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        checkpoints = db.execute(
            "SELECT thread_id, type, checkpoint FROM checkpoints "
            "ORDER BY checkpoint_id"
        )
        for thread, kind, data in checkpoints:
            if thread != thread_id and not thread.startswith(
                thread_id + ":root/"
            ):
                continue
            agent = (
                "root" + thread[len(thread_id) + 5 :]
                if thread != thread_id
                else "root"
            )
            agents.add(agent)
            checkpoint = serde.loads_typed((kind, data))
            for message in checkpoint.get("channel_values", {}).get(
                "messages", []
            ):
                if message.type == "tool":
                    responses[(agent, message.tool_call_id)] = {
                        "status": getattr(message, "status", "success"),
                        "content": str(message.content),
                    }
                calls = getattr(message, "tool_calls", [])
                if not calls:
                    continue
                key = (agent, tuple(call["id"] for call in calls))
                if key in seen:
                    continue
                seen.add(key)
                batches.append({
                    "agent": agent,
                    "timestamp": checkpoint["ts"],
                    "calls": calls,
                })
                for call in calls:
                    if call["name"] != "send_message":
                        continue
                    target = call["args"]["target"]
                    targets = target if isinstance(target, list) else [target]
                    resolved = [
                        t
                        if t == "root" or t.startswith("root/")
                        else "root/" + t
                        for t in targets
                    ]
                    sends.append({
                        "agent": agent,
                        "call_id": call["id"],
                        "timestamp": checkpoint["ts"],
                        "targets": resolved,
                        "message": call["args"]["message"],
                    })
    for send in sends:
        key = (send["agent"], send["call_id"])
        send["result"] = responses.get(key, send.get("result"))
    successful = [
        send
        for send in sends
        if send["result"]
        and send["result"]["status"] == "success"
        and "Message queued" in send["result"]["content"]
    ]
    parallel = [batch for batch in batches if len(batch["calls"]) > 1]
    peer = [
        send
        for send in successful
        if send["agent"] != "root"
        and any(t not in {"root", send["agent"]} for t in send["targets"])
    ]
    return {
        "thread_id": thread_id,
        "agents": sorted(agents),
        "tool_turns": len(batches),
        "parallel_tool_turns": len(parallel),
        "parallel_turns_by_agent": dict(Counter(b["agent"] for b in parallel)),
        "parallel_turns_by_width": dict(
            Counter(len(b["calls"]) for b in parallel)
        ),
        "non_spawn_parallel_turns": sum(
            not any(call["name"] == "spawn_agent" for call in b["calls"])
            for b in parallel
        ),
        "successful_peer_sends": len(peer),
        "successful_peer_group_sends": sum(len(s["targets"]) > 1 for s in peer),
        "parallel_batches": parallel,
        "tool_batches": batches,
        "messages": sends,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("thread_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    previous = (
        json.loads(args.output.read_text())
        if args.merge and args.output.exists()
        else None
    )
    report = audit(args.database, args.thread_id, previous)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        k: v
        for k, v in report.items()
        if k not in {"parallel_batches", "tool_batches", "messages"}
    }
    print(json.dumps(summary, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
