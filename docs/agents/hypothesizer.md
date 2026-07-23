# HypothesizerAgent Documentation

[`HypothesizerAgent`][ursa.agents.hypothesizer_agent.HypothesizerAgent]
maintains a persistent Markdown **hypothesis space** for an ongoing question or
investigation. The current implementation is not the old three-agent debate
workflow; that adversarial review behavior has been renamed and reworked as
[`DeepReviewAgent`][ursa.agents.deep_review_agent.DeepReviewAgent].

Use `HypothesizerAgent` when you want a durable, shareable artifact that records competing hypotheses, relative likelihoods, evidence for and against each hypothesis, uncertainties, change summaries, and recommended next evidence.

## Basic usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import HypothesizerAgent

llm = init_chat_model("openai:gpt-5.4-mini")
agent = HypothesizerAgent(llm=llm, workspace="project_workspace")

state = agent.invoke(
    "Why did the latest alloy simulation produce a lower melting point than expected?"
)

print(agent.format_result(state))
```

By default, the agent writes the hypothesis-space artifact to:

```text
<agent den>/experiences/hypothesis_space.md
```

When the agent is not persisted by name/checkpointer, the den is usually the configured workspace. When `agent_name` persistence is used, the den is the persisted agent directory for that group/name.

## Persistent hypothesis-space file

The key design feature is durability. The full hypothesis artifact is stored as a Markdown experience file so that it can be read later by this agent or other agents, even if conversation context has been summarized away.

For example, a later `ChatAgent` or `ExecutionAgent` can use the `read_experience` tool to bring the current hypothesis space back into context.

The default filename is:

```python
hypothesis_space.md
```

You can choose another safe Markdown filename:

```python
agent = HypothesizerAgent(
    llm=llm,
    experience_filename="melting_point_hypotheses.md",
)
```

Filenames must be simple relative Markdown filenames:

- non-empty;
- not absolute paths;
- not path-like values containing `/` or `..`;
- ending in `.md`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | required | Language model used to update the Markdown artifact. |
| `experience_filename` | `str` | `"hypothesis_space.md"` | Experience file that stores the hypothesis-space artifact. |
| `**kwargs` | `dict` | `{}` | Passed to `BaseAgent`, including workspace, persistence, group, thread/checkpoint, and metrics options. |

## Input formats

You can invoke the agent with a string:

```python
state = agent.invoke("Initial question or new evidence goes here.")
```

A string is treated as both the topic and the new information for the first update.

You can also pass a mapping for more explicit control:

```python
state = agent.invoke({
    "query": "Why did the alloy simulation underpredict melting point?",
    "new_information": "The run used potential X and an NPT equilibration of only 5 ps.",
    "context": "Previous execution agent found no obvious input-script syntax errors.",
    "experience_filename": "melting_point_hypotheses.md",
})
```

For backward compatibility, the input key `question` is accepted as a fallback for `query`.

## Follow-up updates

`format_query` treats follow-up text as new information to incorporate into the existing hypothesis space.

```python
state = agent.invoke("Why is experiment A inconsistent with simulation B?")

query = agent.format_query(
    "New evidence: the experimental sample contained 2% impurity C.",
    state=state,
)
state = agent.invoke(query)
```

The agent reads the previous Markdown artifact, asks the model to update it, and writes the revised artifact back to the same experience file.

## State model

`HypothesizerState` is a `TypedDict` with these important fields:

- `query` — original or current question/topic.
- `new_information` — new evidence, clarification, or instruction for the latest update.
- `context` — optional additional context from another agent behavior or user notes.
- `experience_filename` — Markdown experience file storing the artifact.
- `hypothesis_space_markdown` — latest full Markdown artifact.
- `summary` — compact summary of where the artifact was written and current hypothesis headings.
- `revision_history` — short descriptions of updates made in this thread/run.
- `last_updated` — ISO timestamp for the latest update.

`agent.format_result(state)` returns the full `hypothesis_space_markdown` when available, otherwise the compact `summary`.

## Artifact content

The update prompt asks the model to produce only Markdown and to maintain a concise but useful hypothesis space. A good artifact should include:

- clear hypothesis IDs such as `H1`, `H2`, `H3`;
- relative likelihoods, with a note on whether they are mutually exclusive probabilities or independent plausibility scores;
- evidence for and against each hypothesis;
- preserved prior evidence unless contradicted;
- assumptions and uncertainties;
- an explanation of what changed in the latest update;
- recommended next evidence or work that chat/execution agents could gather.

If the model returns an unusable response, the implementation falls back to a basic Markdown hypothesis-space scaffold.

## Graph behavior

The current graph has a single node:

```text
update_hypothesis_space
```

That node:

1. validates the experience filename;
2. reads any existing hypothesis-space Markdown artifact;
3. builds an update prompt from `query`, `new_information`, `context`, and the previous artifact;
4. invokes the LLM;
5. normalizes/falls back to Markdown if needed;
6. writes the updated artifact to the experiences directory;
7. returns updated state with the artifact, summary, revision history, and timestamp.

## Relationship to DeepReviewAgent

Older docs used “HypothesizerAgent” to mean a three-role iterative debate
system: solution generator, critic, and competitor/stakeholder simulator. That
workflow now lives in
[`DeepReviewAgent`][ursa.agents.deep_review_agent.DeepReviewAgent].

Choose between them as follows:

- Use `HypothesizerAgent` to maintain a durable evolving hypothesis-space document.
- Use `DeepReviewAgent` to perform an adversarial multi-pass review and synthesize a final answer/report.
