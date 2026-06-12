# Getting Started - Plan-Execute From YAML

The `plan_execute_from_yaml.py` workflow is useful for longer tasks that benefit from explicit planning, execution, checkpointing, and restart prompts.

!!! note "Current status"
    This workflow currently lives in the examples directory. It is useful today for long, complex jobs, and future URSA versions are expected to integrate this workflow more directly into the main URSA interface.

## Prerequisites

- URSA is installed.
- You have access to an LLM endpoint.
- You are running from a clone of the URSA repository or otherwise have the `examples/two_agent_examples/plan_execute/` files available.
- You have a dedicated workspace for generated files.

## 1. Choose an example YAML file

The repository includes example task files under:

```text
examples/two_agent_examples/plan_execute/
```

A good first run is:

```text
examples/two_agent_examples/plan_execute/example_from_yaml.yaml
```

Other examples include `pi_multiple_ways.yaml`, `openchami_boot_docs_example.yaml`, and larger scientific demos.

## 2. Run the workflow

```bash
python examples/two_agent_examples/plan_execute/plan_execute_from_yaml.py \
  --config examples/two_agent_examples/plan_execute/example_from_yaml.yaml \
  --workspace ./plan-execute-workspace
```

The runner may ask you to:

- select or confirm an LLM model,
- choose single or hierarchical planning,
- resume from a checkpoint or start fresh.

For a first run, choose the simplest/single-planning option and start fresh.

## 3. Review the generated plan

The planning stage decomposes the YAML task into steps. Read the plan before allowing a long workflow to proceed, especially if the execution step may write files, install packages, or run commands.

## 4. Review outputs in the workspace

The workspace contains generated code, outputs, checkpoints, and artifacts created during the run. Treat it as the durable record of the job.

## 5. Resume or inspect a run

For detailed checkpointing, resume behavior, and troubleshooting, see the [Plan-Execute checkpointing reference](../Plan-Execute-Runner-Checkpointing-Guide.md).

## Notes from the scientific demo workflow

The demo material in `README_dark.html` used the same style of workflow for a longer Bayesian nuclear-data example:

```bash
python plan_execute_from_yaml.py \
  --config bayesian_nuclear_data.yaml \
  --workspace bayesian_nuclear_data
```

That example highlights why YAML-driven plan-execute workflows are useful: they let you describe a complex scientific task once, run it in a dedicated workspace, and keep checkpoints and artifacts for later review.

## Where next?

- [Plan-Execute checkpointing reference](../Plan-Execute-Runner-Checkpointing-Guide.md)
- [Python scripts guide](python-scripts.md)
- [Sandboxing and information control](../best-practices/sandboxing.md)
