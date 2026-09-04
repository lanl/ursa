# Examples

Choose an example based on what you want URSA to do. Each published example is
a self-contained folder with setup instructions, dependencies, inputs, and
expected results. Start with an example tagged `beginner` if this is your first
time using an execution agent, the dashboard, or MCP.

<!-- example-catalog -->

## Adding an example

Use one example per folder with this minimum structure:

```text
examples/category/my_example/
├── README.md       # purpose, setup, run, expected results, and cleanup
├── example.yaml    # documentation metadata
├── pyproject.toml  # isolated Python dependencies, even when empty
└── ...             # scripts, inputs, images, and other example files
```

The metadata requires a title, summary, and tags. Examples are ordered
alphabetically by title:

```yaml
title: My example
summary: One sentence explaining what the reader will learn.
tags:
  - execution-agent
  - beginner
```

The documentation build discovers every `example.yaml`. It publishes the
folder's `README.md` at a matching `/examples/.../` URL and adds it to the
catalog. Keep relative README links local to the example folder; the published
page rewrites links to adjacent files so they open the version-matched source
on GitHub.

### Common tags

Use lowercase, hyphenated tags. Prefer these common tags so related examples
remain easy to find:

| Tag | Use for |
| --- | --- |
| `guided` | A narrative walkthrough with ordered setup, execution, and review steps |
| `source-only` | A focused source example intended primarily to be configured and run |
| `beginner` | A good first example with minimal prerequisites |
| `tui` | Workflows driven through URSA's terminal user interface |
| `dashboard` | Workflows using the browser dashboard |
| `python-api` | Direct use of URSA classes from Python |
| `execution-agent` | Tasks that run commands or create workspace artifacts |
| `mcp` | MCP servers, clients, or tools attached to URSA agents |
| `multi-agent` | Teams, symposia, or other composed-agent workflows |
| `simulation` | Running or analyzing a scientific simulation |
| `plotting` | Producing plots or other visual artifacts |
| `optimization` | Search, experiment selection, or mathematical optimization |
