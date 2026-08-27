## Command Line Usage

You can install `ursa` as a command line app with `pip install`; or with [`uv`](https://docs.astral.sh/uv/) via

```bash
uv tool install ursa-ai
```

To use the command line app, run

```
ursa --llm_model.model openai:gpt-5.2
```

This starts the full-screen terminal app. Type `/` to browse commands,
`#` to choose an agent behavior, or `@` to insert a workspace path.
See [Getting Started - CLI](getting-started/cli.md#full-screen-interface-controls)
for prompt editing, multiline input, clipboard, and exit behavior.

You can chat with an LLM by simply typing into the terminal.

```
How are you?
Thanks for asking! I’m doing well. How are you today? What can I help you with?
```

Use the required `#` macro to route a prompt to another agent behavior:

```
#plan Write a python script to do linear regression using only numpy.
```

Agent macros route only the prompt in which they appear. Output from a previous
agent is not automatically appended to the next prompt; quote or reference any
needed result explicitly when switching behaviors.

You can get a list of available command line options via
```
ursa --help
```
