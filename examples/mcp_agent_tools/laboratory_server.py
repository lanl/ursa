from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Laboratory measurements", json_response=True)


@mcp.tool()
def list_measurements() -> list[dict[str, str | float]]:
    """Return a small set of example laboratory measurements."""
    return [
        {"sample": "alloy-a", "temperature_k": 298.0, "strength_mpa": 512.0},
        {"sample": "alloy-b", "temperature_k": 298.0, "strength_mpa": 547.0},
        {"sample": "alloy-c", "temperature_k": 350.0, "strength_mpa": 489.0},
    ]


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
