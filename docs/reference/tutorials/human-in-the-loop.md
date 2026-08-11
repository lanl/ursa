# URSA Human-in-the-Loop Agent Interface Documentation

## Previous Human-in-the-Loop Example has been Deprecated

Launch the HITL interface with:

`$ ursa`

Use `/` inside the app to browse commands, `#` to choose an agent behavior,
and `@` to insert a workspace file or directory. Run `ursa --help` for
startup and configuration options.


## Basic Usage

Plain text is handled by the default chat agent. To use another agent, type
`#` and choose it from the fuzzy-searchable picker. The selected macro is
inserted at the start of the prompt:

```
#execute Make me a histogram of the first 10000 prime number spacings
```

The `#` prefix is required. Bare text such as `execute ...` is sent to chat.
Use `/agents` to see every configured agent and its options, `/status` for
models, endpoints, token usage, MCP servers, and the active persistent agent,
and `/keymap` for the complete keyboard map.


Some additional documentation on the URSA github repo: [LINK](https://github.com/lanl/ursa)
with more to come.

See the main CLI guide for configuration and persistent-agent usage.
