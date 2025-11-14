from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import StdioServerParameters
from mcp.client.session_group import (
    ServerParameters,
    SseServerParameters,
    StreamableHttpParameters,
)


def transport(sp: ServerParameters):
    if isinstance(sp, StdioServerParameters):
        return "stdio"
    elif isinstance(sp, StreamableHttpParameters):
        return "streamable_http"
    elif isinstance(sp, SseServerParameters):
        return "sse"
    else:
        raise RuntimeError("Transport for {sp} is unknown")


def start_mcp_client(server_configs: dict[str, ServerParameters]):
    config = {
        server: {**config.model_dump(), "transport": transport(config)}
        for server, config in server_configs.items()
    }
    return MultiServerMCPClient(config)
