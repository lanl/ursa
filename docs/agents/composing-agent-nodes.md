# Composing agents as graph nodes

Use `BaseAgent.add_agent_node` when a workflow has a fixed child-agent stage.
The child is compiled as a LangGraph subgraph, but the parent sees one node with
an explicit state boundary.

## Define both state adapters

`add_agent_node` requires two adapters:

- `input_fn` converts parent state into the child agent's state.
- `output_fn` converts completed child state into a partial parent-state update.

There are no default adapters. `BaseAgent.format_query()` accepts a user-facing
prompt and `format_result()` returns a user-facing string; those contracts are
different from graph-state conversion.

```python
from typing import TypedDict, cast

from langgraph.types import Overwrite

from ursa.agents.base import BaseAgent
from ursa.agents.planning_agent import PlanningAgent, PlanningState


class ResearchState(TypedDict, total=False):
    question: str
    plan: object


class ResearchWorkflow(BaseAgent[ResearchState]):
    state_type = ResearchState

    def __init__(self, llm, **kwargs):
        super().__init__(llm, **kwargs)
        self.planner = PlanningAgent(
            llm,
            workspace=self.workspace,
            group=self.group,
            thread_id=self.thread_id,
            enable_metrics=False,
        )

    @staticmethod
    def _planner_input(state: ResearchState) -> PlanningState:
        return cast(
            PlanningState,
            {
                "task": state["question"],
                "messages": Overwrite([]),
                "review": "",
                "reflection_steps": 1,
            },
        )

    @staticmethod
    def _planner_output(state: PlanningState) -> ResearchState:
        return {"plan": state["plan"]}

    def _build_graph(self) -> None:
        self.add_agent_node(
            "planner",
            self.planner,
            input_fn=self._planner_input,
            output_fn=self._planner_output,
        )
        self.graph.set_entry_point("planner")
        self.graph.set_finish_point("planner")
```

The output adapter must return a mapping accepted by the parent state schema.
Return only the channels that the child is responsible for updating.

## Handle reducer-backed state deliberately

LangGraph reducers still apply at both boundaries. If a child state uses
`add_messages`, a plain list merges with messages already stored in the child's
checkpoint namespace. Use `Overwrite(...)` when each child run needs an isolated
transcript. Use an ordinary list when accumulating history is intentional.

## Parent and child ownership

The parent owns the compiled runtime, store, checkpointer, and nested checkpoint
namespaces. The node name becomes the child's checkpoint namespace. Construct
embedded children without independent checkpointers.

Registered children are available through the read-only `agent_nodes` mapping.
The parent propagates storage setup, async-only tool detection, compiled-graph
invalidation, and `close()`/`aclose()` across that composition. This also lets
integrations such as the dashboard discover tool-capable children without
depending on workflow-specific attribute names.

The parent runtime context is shared with the child graph. Align the child's
workspace, group, thread, and model configuration with the parent when creating
the child. Agent methods that directly use child instance fields, such as a
child-specific `den`, still use those fields.

## Choose subgraphs only for fixed stages

Use `add_agent_node` for a child that always occupies a known graph stage. A
dynamic delegation tool that selects among agents at model runtime is a different
pattern and should remain a tool or environment-level delegation mechanism.

Both synchronous and asynchronous parent invocation are supported. Prefer
`ainvoke()` when any composed child has asynchronous-only tools.
