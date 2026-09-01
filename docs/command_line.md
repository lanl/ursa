## Command Line Usage

You can install `ursa` as a command line app with `pip install`; or with [`uv`](https://docs.astral.sh/uv/) via

```bash
uv tool install --python 3.13 'ursa-ai[dashboard]'
```

To use the command line app, run

```bash
ursa --llm_model.model openai:gpt-5.2
```

This starts the full-screen terminal app. Type `/` to browse commands,
`#` to choose an agent behavior, or `@` to insert a workspace path.

You can chat with an LLM by simply typing into the terminal.

```text
How are you?
Thanks for asking! I’m doing well. How are you today? What can I help you with?
```

Use the required `#` macro to route a prompt to another agent behavior:

```text
#plan Write a python script to do linear regression using only numpy.
```

If you run subsequent agents, the last output will be appended to the prompt for the next agent.

So, to run the Planning Agent followed by the Execution Agent:
```text
#plan Write a python script to do linear regression using only numpy.

...

#execute Execute the plan.
```

You can get a list of available command line options via
```bash
ursa --help
```

Inspect the running URSA and Python installation with:

```bash
ursa self status
```

When installed with `uv tool install`, run `ursa self update` to update URSA
or `ursa self modify --help` to change extras, additional packages, or the
selected version/source revision.
