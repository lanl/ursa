from ursa.agents import ArxivAgent, ArxivAgentLegacy, OSTIAgent, WebSearchAgent
from ursa.observability.timing import render_session_summary

# Web search (ddgs) agent
web_agent = WebSearchAgent(
    llm="openai/o3-mini",
    max_results=20,
    database_path="web_db",
    summaries_path="web_summaries",
    enable_metrics=True,
)
tid = "run-" + __import__("uuid").uuid4().hex[:8]
web_agent.thread_id = tid
summary = web_agent.invoke({
    "query": "graph neural networks for PDEs",
    "context": "Summarize methods & benchmarks and potential for shock hydrodynamics",
})
render_session_summary(tid)
print("=" * 80)
print("=" * 80)
print(summary)
print("=" * 80)
print("=" * 80)

# OSTI agent
osti_agent = OSTIAgent(
    llm="openai/o3-mini",
    max_results=5,
    database_path="osti_db",
    summaries_path="osti_summaries",
    enable_metrics=True,
)
tid = "run-" + __import__("uuid").uuid4().hex[:8]
osti_agent.thread_id = tid

summary = osti_agent.invoke({
    "query": "quantum annealing materials",
    "context": "What are the key findings?",
})
render_session_summary(tid)
print(summary)
print("=" * 80)
print("=" * 80)
# ArXiv agent (legacy version)
arxiv_agent = ArxivAgentLegacy(
    llm="openai/o3-mini",
    max_results=3,
    database_path="arxiv_papers",
    summaries_path="arxiv_generated_summaries",
    enable_metrics=True,
)
tid = "run-" + __import__("uuid").uuid4().hex[:8]
arxiv_agent.thread_id = tid

summary = arxiv_agent.invoke({
    "query": "graph neural networks for PDEs",
    "context": "Summarize methods & benchmarks and potential for shock hydrodynamics",
})
render_session_summary(tid)
print(summary)
print("=" * 80)
print("=" * 80)
# ArXiv agent
arxiv_agent = ArxivAgent(
    llm="openai/o3-mini",
    max_results=3,
    database_path="arxiv_papers",
    summaries_path="arxiv_generated_summaries",
    enable_metrics=True,
)
tid = "run-" + __import__("uuid").uuid4().hex[:8]
arxiv_agent.thread_id = tid

summary = arxiv_agent.invoke({
    "query": "graph neural networks for PDEs",
    "context": "Summarize methods & benchmarks and potential for shock hydrodynamics",
})
render_session_summary(tid)
print(summary)
