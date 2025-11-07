from importlib.metadata import version as get_version
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

mcp_app = FastAPI(
    title="URSA Server",
    description="Micro-service for hosting URSA to integrate as an MCP tool.",
    version=get_version("ursa-ai"),
)


class QueryRequest(BaseModel):
    agent: Literal[
        "arxiv", "plan", "execute", "web", "recall", "chat", "hypothesize"
    ]
    query: Annotated[
        str,
        Field(examples=["Write the first 1000 prime numbers to a text file."]),
    ]


class QueryResponse(BaseModel):
    response: str


def get_hitl(req: Request):
    # Single, pre-created instance set by the CLI (see below)
    return req.app.state.hitl


@mcp_app.post("/run", response_model=QueryResponse)
def run_ursa(req: QueryRequest, hitl=Depends(get_hitl)):
    """
    Queries the URSA Agentic AI Workflow to request that one of the URSA Agents
    address a query. The available agents are:
        ArxivAgent: Search for papers on ArXiv and summarize them in the context of the query
        PlanningAgent: Builds a structured step-by-step plan to attempt to solve the users problem
        ExecuteAgent: Runs a ReAct agent to write/edit code and run commands to attempt to solve the user query
        WebSearchAgent: Search the web for information on a query and summarize the results given that context
        RecallAgent: Perform RAG on previous ExecutionAgent steps saved in a memory database
        HypothesizerAgent: Perform detailed reasoning to propose an approach to solve a given user problem/query
        ChatAgent: Query the hosted LLM as a straightforward chatbot.

    Arguments:
        agent: str, one of: arxiv, plan, execute, web, recall, hypothesize, or chat. Directs the query to the corresponding agent
        query: str, query to send to the requested agent for processing

    Returns:
        response: str, summary of the agent output. The Execute agent may also write code and generate artifacts in the ursa_mcp workspace
    """
    try:
        match req.agent:
            case "arxiv":
                response = hitl.run_arxiv(req.query)
            case "plan":
                response = hitl.run_planner(req.query)
            case "execute":
                response = hitl.run_executor(req.query)
            case "web":
                response = hitl.run_websearcher(req.query)
            case "recall":
                response = hitl.run_rememberer(req.query)
            case "hypothesize":
                response = hitl.run_hypothesizer(req.query)
            case "chat":
                response = hitl.run_chatter(req.query)
            case _:
                response = f"Agent '{req.agent}' not found."
        return QueryResponse(response=response)
    except Exception as exc:
        # Surface a readable error message for upstream agents
        raise HTTPException(status_code=500, detail=str(exc)) from exc
