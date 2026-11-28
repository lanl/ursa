"""List the models that the #fm agent can find on your MCP servers.

Run it from this folder with the config file you give to `ursa --config`:

    uv run python check_cards.py nomad.yaml

For each server it prints the tools, then calls `get_model_card` for each
tool the way #fm does. #fm counts each MODEL line. It skips each NO CARD
line: that is normal for helper tools, such as a search tool, but a model
listed as NO CARD points to a server problem, which the message names.
"""

import asyncio
import sys
from pathlib import Path

from ursa.cli.config import load_config_file
from ursa.util.mcp import load_mcp_tools_by_server, start_mcp_client

CARD_TOOL = "get_model_card"


async def main(path: Path) -> int:
    servers = load_config_file(path).get("mcp_servers") or {}
    if not servers:
        print(f"{path} has no mcp_servers section, so #fm sees no models.")
        return 1
    client = start_mcp_client(servers)
    ok = 0
    for server, tools in await load_mcp_tools_by_server(client):
        print(f"Server '{server}': connected, {len(tools)} tool(s)")
        card_tool = next((t for t in tools if t.name == CARD_TOOL), None)
        if card_tool is None:
            print(
                f"  No {CARD_TOOL} tool, so #fm cannot read model cards here."
            )
            continue
        for tool in tools:
            if tool.name == CARD_TOOL:
                continue
            try:
                card = await card_tool.ainvoke({"tool_name": tool.name})
            except Exception as exc:  # noqa: BLE001 - report every failure
                print(f"  NO CARD  {tool.name}: {exc}")
                continue
            ok += 1
            print(
                f"  MODEL    {tool.name}: card of {len(str(card))} characters"
            )
    print(f"#fm should find {ok} model(s).")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(asyncio.run(main(Path(sys.argv[1]))))
