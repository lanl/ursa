# DeepReviewAgent Documentation

[`DeepReviewAgent`][ursa.agents.deep_review_agent.DeepReviewAgent] is the renamed
and reworked version of the older “Hypothesizer” adversarial review workflow. It
runs an iterative three-role process—solution generation, critique, and
competitor/stakeholder perspective—then synthesizes a final answer and a LaTeX
report.

Do not confuse this agent with
[`HypothesizerAgent`][ursa.agents.hypothesizer_agent.HypothesizerAgent]. The
current HypothesizerAgent maintains a persistent hypothesis-space Markdown
artifact. DeepReviewAgent performs a multi-pass adversarial review. See the
[HypothesizerAgent guide][hypothesizeragent-documentation].

## Basic usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import DeepReviewAgent

llm = init_chat_model("openai:gpt-5.4-mini")
agent = DeepReviewAgent(llm=llm, max_iterations=2, use_web=False)

state = agent.invoke(
    "What strategies could a small local bookstore use to compete with large online retailers?"
)

print(agent.format_result(state))
```

`format_result` returns the final synthesized `solution`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | required | Language model used for all review phases and final synthesis. |
| `max_iterations` | `int` | `2` | Number of solution/critique/perspective cycles to run before final synthesis. |
| `use_web` | `bool` | `False` | Add web, OSTI, and arXiv search tools. |
| `extra_tools` | `list[BaseTool] \| None` | `None` | Additional tools to expose to role phases. |
| `**kwargs` | `dict` | `{}` | Passed to `BaseAgent` / `AgentWithTools`, including workspace, persistence, group, RAG tools, MCP, and checkpoint options. |

## Tool model

The old deep-review behavior used hidden DuckDuckGo searches. The current implementation does **not** perform hidden web search. External information access is routed through explicit LangChain tools.

Tools always available:

- `list_workspace_files` — list files available in the current workspace.
- `read_file` — read workspace files and documents.

Tools added only when `use_web=True`:

- `run_web_search`
- `run_osti_search`
- `run_arxiv_search`

Other tools can be supplied through:

- `extra_tools` at construction time;
- persistent RAG tools through `rag_tools`;
- MCP tools through the CLI/MCP configuration where supported by `AgentWithTools`.

Each role phase is explicitly told what tools are available. If `use_web=False`, the phase instructions say that no web/search tools are available and that the model should not claim web searches.

## Iterative review process

`DeepReviewAgent` maintains three role outputs per iteration:

1. **Agent 1 — solution/hypothesis generator**
   - Produces an initial solution on the first iteration.
   - On later iterations, revises the solution using the previous solution, critique, and competitor/stakeholder perspective.
   - Must explicitly describe how the new solution differs from the previous one.

2. **Agent 2 — critic**
   - Reviews the proposed solution.
   - Identifies flaws, assumptions, missing evidence, and improvements.

3. **Agent 3 — competitor/stakeholder/adversarial perspective**
   - Simulates how a competitor, government agency, stakeholder, or adversarial party might respond.
   - Highlights objections, incentives, strategic concerns, and alternative interpretations.

After all iterations complete, the agent synthesizes the overall refined solution from the evolution of all role outputs.

## Graph behavior

The graph uses phase nodes plus a shared tool node:

```text
agent1 -> agent2 -> agent3 -> increment_iteration
   ^        ^        ^              |
   |        |        |              v
 tool_node routes back to active phase
```

More specifically:

1. `agent1` runs the solution generator.
2. If the response contains tool calls, `tool_node` executes them and routes back to the active phase.
3. When `agent1` completes without tool calls, the graph proceeds to `agent2`.
4. `agent2` follows the same tool-or-complete pattern, then proceeds to `agent3`.
5. `agent3` follows the same pattern, then increments `current_iteration`.
6. If `current_iteration < max_iterations`, the graph loops back to `agent1`.
7. Otherwise it runs `finalize`, then `summarize_as_latex`, then `print_sites` and finishes.

Tool routing is controlled by `active_phase`, so tool results return to the role that requested them.

## State model

`DeepReviewState` includes:

- `question` — original question or task.
- `question_search_query` — normalized short query derived from the question.
- `current_iteration` — current iteration count.
- `max_iterations` — iteration limit.
- `agent1_solution` — list of solution outputs, one per completed iteration.
- `agent2_critiques` — list of critique outputs.
- `agent3_perspectives` — list of competitor/stakeholder/adversarial outputs.
- `solution` — final synthesized answer.
- `summary_report` — LaTeX report summarizing the review process.
- `visited_sites` — URLs found in tool output messages.
- `messages` — accumulated role/tool transcript.
- `active_phase` — current phase used for tool routing.

String inputs are normalized into this state automatically. Mapping inputs may use `query` as a fallback for `question`.

## Outputs and artifacts

The agent returns state containing:

- the final `solution`;
- all intermediate role outputs;
- `summary_report`, a LaTeX document containing an executive summary, final solution, and appendix-style process summary;
- `visited_sites`, if URLs appeared in tool outputs;
- a timestamped text file in the agent den containing iteration details used to create the report.

## Choosing DeepReviewAgent vs HypothesizerAgent

- Choose `DeepReviewAgent` for intensive adversarial review of a question with a synthesized final answer/report.
- Choose `HypothesizerAgent` for maintaining a durable hypothesis-space Markdown file that other agents can read and update over time.
