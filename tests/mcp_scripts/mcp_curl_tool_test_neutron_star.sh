# curl -X POST http://localhost:8000/run \
# -H "Content-Type: application/json" \
# -H "Accept: application/json, text/event-stream" \
# -H "MCP-Protocol-Version: 2025-06-18" \
# -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"clientInfo":{"name":"curl-client","version":"1.0"}}}'
# 

curl -X POST http://localhost:8000/run \
-H "Content-Type: application/json" \
-d '{
    "agent": "chat",
    "query": "What are current best scientific constaints on neutron star radius?"
}'
