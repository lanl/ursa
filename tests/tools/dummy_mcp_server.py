#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastmcp",
# ]
# ///
from fastmcp import FastMCP

mcp = FastMCP()


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


if __name__ == "__main__":
    mcp.run(show_banner=False)
