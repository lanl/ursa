# Agent Symposia

An Agent Symposium is URSA's peer-review collaboration pattern. Instead of asking
one coordinator to divide the work, a symposium gives the same problem to multiple
participants. Each participant works independently, reviews the submitted writeups,
revises its own answer, and then an organizer synthesizes the final report.

The pattern is inspired by a research symposium or review panel: different
participants may bring different tools, assumptions, models, and reasoning styles.
The goal is not just to get more text. The goal is to make disagreement visible,
let participants catch one another's mistakes, and give the final organizer a
richer evidence base for synthesis.

## Why independence matters

A single model can be confidently wrong. It may make an arithmetic mistake,
overlook a constraint, hallucinate an unsupported claim, or choose a brittle
implementation. A symposium gives you several chances to catch those failures:

- members work independently before seeing each other's answers,
- reviewers critique strengths, weaknesses, missing checks, evidence quality, and
  reproducibility,
- members revise only their own work after seeing review feedback,
- the organizer compares consensus and disagreement before producing the final
  answer.

A particularly valuable use of symposia is assigning **different LLMs to different
members**. For example, one participant might use a strong coding model, another a
local open model, and another a model from a different provider. Different models
make different errors. When their outputs are forced through review and synthesis,
one model can catch mistakes another model missed.

This does not make the result automatically correct. It does make the workflow more
resilient than asking one model to solve, critique, and bless its own answer in a
single pass.

## When a symposium is the right shape

Use an Agent Symposium when you want:

- independent approaches to the same complex problem,
- critique before final synthesis,
- comparison between tools, models, or methods,
- a nested team to compete or collaborate with an independent solver,
- a final answer that reports consensus, disagreement, uncertainty, and evidence
  quality.

Use an [Agent Team][agent-teams] instead when one PI should assign specialist
subtasks and produce one integrated answer without independent peer-review rounds.

## What happens during a symposium run

A symposium run has four phases:

1. **Independent work.** Each member receives the original task and writes a
   self-contained solution.
2. **Peer review.** Each member receives all submitted writeups, including its own,
   and writes critical reviews. Review prompts explicitly instruct members not to
   edit or overwrite reviewed work.
3. **Revision.** Each member revises only its own work using the reviews and what
   it learned from comparing submissions.
4. **Synthesis.** The organizer receives final writeups and peer reviews, then
   produces the final symposium report.

Initial work, reviews, and revisions are run asynchronously across members when
using the environment's async implementation. Synchronous callers can still use
`invoke(...)`; async callers should use `await ainvoke(...)`.

## Minimal symposium YAML

```yaml
name: example_symposium
group: default
description: >
  A symposium where several participants solve, review, revise, and synthesize
  a complex technical problem.
revision_rounds: 1

organizer:
  name: organizer
  role: Final synthesizer and judge of evidence quality
  agent: ChatAgent
  config:
    use_web: false

members:
  - name: implementation_path
    role: Builds and tests a concrete solution
    agent: ExecutionAgent
    config:
      use_web: false

  - name: review_path
    role: Looks for hidden assumptions, missing checks, and weak evidence
    agent: ChatAgent
    config:
      use_web: false
```

The repository also includes an example at
`examples/environments/agent_symposium.yaml`. That example demonstrates a nested
[`AgentTeamEnvironment`][ursa.environments.AgentTeamEnvironment] as one
symposium member and an independent
[`ExecutionAgent`][ursa.agents.execution_agent.ExecutionAgent]
as another member.

## Run a symposium from Python

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentSymposiumEnvironment

llm = init_chat_model(model="openai:gpt-4o-mini")

symposium = AgentSymposiumEnvironment.from_yaml(
    "examples/environments/agent_symposium.yaml",
    llm=llm,
)

result = symposium.invoke(
    "Compare two solution strategies for this problem and recommend one."
)

print(result["final"])
```

The returned mapping includes more than the final answer:

- `task`: the normalized task,
- `initial_writeups`: each member's first independent answer,
- `review_rounds`: the review output from each round,
- `reviews`: the latest review round,
- `final_writeups`: each member's revised answer,
- `organizer_result`: the organizer's raw result,
- `final`: best-effort text extracted from the organizer result.

That structure is useful when you want to audit how the final synthesis was made.

## Use different LLMs for different members

Member-specific `model:` blocks let symposium members use different model
providers or endpoints. This is one of the main reasons to use a symposium: model
diversity can expose blind spots.

```yaml
name: multi_model_symposium
group: default
revision_rounds: 1

organizer:
  name: organizer
  role: Synthesizes consensus, disagreement, and evidence quality
  agent: ChatAgent
  model:
    model: openai:gpt-4o-mini
    api_key_env: OPENAI_API_KEY

members:
  - name: coding_model
    role: Prioritizes implementation, tests, and reproducibility
    agent: ExecutionAgent
    model:
      model: openai:gpt-4o-mini
      api_key_env: OPENAI_API_KEY

  - name: local_critic
    role: Reviews assumptions from an independent local-model perspective
    agent: ChatAgent
    model:
      model: ollama:llama3.1
      base_url: http://localhost:11434

  - name: alternate_provider
    role: Looks for conceptual mistakes and alternative explanations
    agent: ChatAgent
    model:
      model: anthropic:claude-3-5-sonnet-latest
      api_key_env: ANTHROPIC_API_KEY
```

The environment-level `llm` is still required when you construct the symposium.
Any member without a `model:` block inherits that default model.

## Nested teams as symposium members

A symposium member can itself be an `AgentTeamEnvironment`. This is useful when
you want one participant to be a coordinated team and another participant to be an
independent solver.

```yaml
members:
  - name: team_path
    role: Nested team pursuing a coordinated solution
    agent: AgentTeamEnvironment
    config:
      config:
        name: nested_solution_team
        description: A nested team used as one symposium participant.
        pi:
          name: pi
          role: Nested-team PI
          agent: ExecutionAgent
        members:
          - name: implementer
            role: Implements and tests a solution
            agent: ExecutionAgent
          - name: critic
            role: Reviews the nested team's reasoning
            agent: ChatAgent

  - name: independent_path
    role: Independent solver for comparison
    agent: ExecutionAgent
```

The nested `config: config:` shape is intentional in this implementation: the
outer member's `config` contains constructor keyword arguments for
`AgentTeamEnvironment`, and that constructor receives its own team `config`.

## Revision rounds

`revision_rounds` controls how many review-and-revision cycles occur. The
implementation runs at least one round, even if the configured value is less than
one.

Start with one round. Additional rounds can improve difficult work, but they also
increase cost and runtime because each round invokes every member for review and
revision.

## Saving a symposium configuration

```python
from ursa.environments import AgentSymposiumConfig, save_symposium_config

config = AgentSymposiumConfig(name="design_review", group="default")
path = save_symposium_config(config)
print(path)
```

When no path is supplied, URSA writes to:

```text
~/.cache/ursa/<group>/environments/agent_symposia/<name>/symposium.yaml
```

## Practical guidance

- Give each member a different reason to exist: implementation, critique,
  literature review, risk analysis, or an alternative model/provider.
- Ask the organizer to report consensus and disagreement, not just a polished
  answer.
- Use member-specific model configs when you want different models to catch each
  other's mistakes.
- Keep artifacts in dedicated workspaces when members can write or run code.
- Inspect `initial_writeups`, `reviews`, and `final_writeups` for important
  decisions; do not rely only on the final synthesis for high-stakes work.
