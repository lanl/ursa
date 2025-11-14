# Molecular Foundation Model MCP Server
A [MCP](https://modelcontextprotocol.io) Server for serving molecular foundation models including:

- [MIST](https://arxiv.org/abs/2510.18900)

## Usage

- Start an mcp server: `./molfm-mcp mcp path/to/model`
- Query a model for a prediction: `./molfm-mcp query path/to/model MOLECULES...`
- Search PubChem for a molecule: `./molfm-mcp serach caffine`

## Example Config

Put the following in `ursa.yml` and then launch ursa with `uv run ursa run --config ursa.yml`

```yaml
# Other Ursa configuration settings

mcp_servers:
  molfm-mcp:
    command: bin/molfm-mcp
    args:
      - mcp
      - ./models
    cwd: mcp/molfm-mcp # or similar
# another_mcp_server:
# ...
```
