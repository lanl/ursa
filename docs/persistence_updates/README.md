# URSA Updated Usage Guides

This folder contains concise user guides for recent URSA changes around persistent agents, groups, tools, RAG tools, the dashboard, and prompt refinement.

## Guides

1. [Persistent named agents](01_persistent_named_agents.md)
2. [Groups and endpoint security](02_groups_and_security.md)
3. [Agent behaviors, tools, ChatAgent, and ExecutionAgent](03_agent_behaviors_and_tools.md)
4. [PromptingAgent guide](04_prompting_agent.md)
5. [Persistent RAG agents and RAG tools](05_persistent_rag_agents.md)
6. [Web dashboard workflow](06_web_dashboard_workflow.md)
7. [Quick command reference](07_quick_reference.md)

## Key idea

URSA now separates a persistent **agent identity** from the **behavior** used to operate on it.

A named agent stores durable state under a group. You can use that same named agent with different behaviors, such as Chat, Execution, Planning + Execution, or Prompting, depending on the interface and workflow.
