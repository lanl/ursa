## Command Line Usage

You can install `ursa` as a command line app with `pip install`; or with [`uv`](https://docs.astral.sh/uv/) via

```bash
uv tool install ursa-ai
```

To use the command line app, run

```
ursa --llm_model.model openai:gpt-5.2
```

This will start a REPL in your terminal.

```
  __  ________________ _
 / / / / ___/ ___/ __ `/
/ /_/ / /  (__  ) /_/ /
\__,_/_/  /____/\__,_/

For help, type: ? or help. Exit with Ctrl+d.
ursa>
```

Within the REPL, you can get help by typing `?` or `help`.

You can chat with an LLM by simply typing into the terminal.

```
ursa> How are you?
Thanks for asking! I’m doing well. How are you today? What can I help you with?
```

You can run various agents by typing the name of the agent. For example,

```
ursa> plan
plan: Write a python script to do linear regression using only numpy.
```

Or by prepending the agent name to the query:

```shell
ursa> plan Write a python script to do linear regression using only numpy.
```

If you run subsequent agents, the last output will be appended to the prompt for the next agent.

So, to run the Planning Agent followed by the Execution Agent:
```
ursa> plan
plan: Write a python script to do linear regression using only numpy.

...

ursa> execute
execute: Execute the plan.
```

You can get a list of available command line options via
```
ursa --help
```

