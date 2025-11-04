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
