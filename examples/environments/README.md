# Build an agent team

Give one URSA agent responsibility for a result, then let it delegate focused
work to specialists. In this example, a principal investigator coordinates a
research specialist and a data analyst. They share a workspace and return one
synthesized answer.

Start with the team before trying the larger symposium:

1. Inspect [`agent_team.yaml`](agent_team.yaml) and notice the PI, the two members, and the tools
   each member may use.
2. Run the team from Python and watch the PI divide the task.
3. Change the roles or prompt, then run it again to see how the delegation
   changes.
4. When you are ready to compare independent solutions, open
   [`agent_symposium.yaml`](agent_symposium.yaml). It places a nested team and an independent solver in
   a review-and-synthesis workflow.

Read [Agent teams](../../docs/environments/agent-teams.md) and
[Agent symposia](../../docs/environments/agent-symposia.md) for the concepts and full
configuration reference.

## Run the team

Clone URSA, open a terminal at the repository root, and set your OpenAI API key.
Then install this example's dependencies and run its small Python entry point.

=== "macOS/Linux"

    ```bash
    cd examples/environments
    export OPENAI_API_KEY="your-api-key"
    uv sync
    uv run python run_team.py
    ```

=== "Windows PowerShell"

    ```powershell
    Set-Location examples\environments
    $env:OPENAI_API_KEY = "your-api-key"
    uv sync
    uv run python run_team.py
    ```

The runner loads [`agent_team.yaml`](agent_team.yaml), initializes the configured chat model, and
asks the team to compare two approaches to a data-analysis task. Edit the task
inside [`run_team.py`](run_team.py) to give the team a problem of your own.

```python
--8<-- "examples/environments/run_team.py"
```

If you use another model provider, configure its credentials and update the
model in `run_team.py`. See [Models and inference
providers](../../docs/configuration/models.md) for supported configurations.

## Shape the team

Edit the roles and prompts in [`agent_team.yaml`](agent_team.yaml). Keep each role specific: the PI
should coordinate and synthesize, while each member should own a distinct kind
of work. The included team configuration is short enough to use as a starting
point:

```yaml
--8<-- "examples/environments/agent_team.yaml"
```

The PI and members can use web or execution tools according to their `config`
blocks. Review those permissions before you launch the team, especially when
you point it at a non-temporary workspace. The [environment
documentation](../../docs/environments/index.md) explains workspaces, persistence, member
models, and execution behavior in more detail.

## Run it from the dashboard

Install the dashboard-enabled URSA tool, then launch it:

=== "macOS/Linux"

    ```bash
    uv tool install --python 3.13 'ursa[dashboard]'
    ursa-dashboard
    ```

=== "Windows PowerShell"

    ```powershell
    uv tool install --python 3.13 "ursa[dashboard]"
    ursa-dashboard
    ```

Open the displayed local URL, configure your model under **Settings**, and open
**Environment runs**. Create a team, replace the starter definition with the
contents of [`agent_team.yaml`](agent_team.yaml), enter a task, validate the YAML, and launch the
run. The dashboard shows the environment graph and live work timeline.

Follow the [dashboard getting-started guide](../../docs/getting-started/dashboard.md)
for credential storage, workspace selection, run history, and cancellation.

## Try the symposium

After the team works, use [`agent_symposium.yaml`](agent_symposium.yaml) as the next exercise. The
symposium sends the same problem to a nested team and an independent solver,
asks them to review and revise their work, and has an organizer synthesize the
result. You can launch that YAML from **Environment runs** in the dashboard, or
load it with `AgentSymposiumEnvironment.from_yaml()` as shown in the [Python
scripts guide](../../docs/getting-started/python-scripts.md#compose-agents-with-environments).
