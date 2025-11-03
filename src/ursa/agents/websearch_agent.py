# from langchain_community.tools    import TavilySearchResults
# from langchain_core.runnables.graph import MermaidDrawMethod
from typing import Annotated, Any, Mapping

import requests
from bs4 import BeautifulSoup
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from pydantic import Field
from typing_extensions import TypedDict

from ..prompt_library.websearch_prompts import (
    reflection_prompt,
    summarize_prompt,
    websearch_prompt,
)
from .base import BaseAgent


class WebSearchState(TypedDict):
    websearch_query: str
    messages: Annotated[list, add_messages]
    urls_visited: list[str]
    max_websearch_steps: Annotated[
        int, Field(default=100, description="Maximum number of websearch steps")
    ]
    remaining_steps: int
    is_last_step: bool
    model: Any
    thread_id: Any


# Adding the model to the state clumsily so that all "read" sources arent in the
# context window. That eats a ton of tokens because each `llm.invoke` passes
# all the tokens of all the sources.


class WebSearchAgentLegacy(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel = init_chat_model("openai:gpt-5-mini"),
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.websearch_prompt = websearch_prompt
        self.reflection_prompt = reflection_prompt
        self.tools = [search_tool, process_content]  # + cb_tools
        self.has_internet = self._check_for_internet(
            kwargs.get("url", "http://www.lanl.gov")
        )
        self._build_graph()

    def _review_node(self, state: WebSearchState) -> WebSearchState:
        if not self.has_internet:
            return {
                "messages": [
                    HumanMessage(
                        content="No internet for WebSearch Agent so no research to review."
                    )
                ],
                "urls_visited": [],
            }

        translated = [SystemMessage(content=reflection_prompt)] + state[
            "messages"
        ]
        res = self.llm.invoke(
            translated, {"configurable": {"thread_id": self.thread_id}}
        )
        return {"messages": [HumanMessage(content=res.content)]}

    def _response_node(self, state: WebSearchState) -> WebSearchState:
        if not self.has_internet:
            return {
                "messages": [
                    HumanMessage(
                        content="No internet for WebSearch Agent. No research carried out."
                    )
                ],
                "urls_visited": [],
            }

        messages = state["messages"] + [SystemMessage(content=summarize_prompt)]
        response = self.llm.invoke(
            messages, {"configurable": {"thread_id": self.thread_id}}
        )

        urls_visited = []
        for message in messages:
            if message.model_dump().get("tool_calls", []):
                if "url" in message.tool_calls[0]["args"]:
                    urls_visited.append(message.tool_calls[0]["args"]["url"])
        return {"messages": [response.content], "urls_visited": urls_visited}

    def _check_for_internet(self, url, timeout=2):
        """
        Checks for internet connectivity by attempting an HTTP GET request.
        """
        try:
            requests.get(url, timeout=timeout)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _state_store_node(self, state: WebSearchState) -> WebSearchState:
        state["thread_id"] = self.thread_id
        return state
        # return dict(**state, thread_id=self.thread_id)

    def _create_react(self, state: WebSearchState) -> WebSearchState:
        react_agent = create_agent(
            self.llm,
            self.tools,
            state_schema=WebSearchState,
            system_prompt=self.websearch_prompt,
        )
        return react_agent.invoke(state)

    def _build_graph(self):
        graph = StateGraph(WebSearchState)
        self.add_node(graph, self._state_store_node)
        self.add_node(graph, self._create_react)
        self.add_node(graph, self._review_node)
        self.add_node(graph, self._response_node)

        graph.set_entry_point("_state_store_node")
        graph.add_edge("_state_store_node", "_create_react")
        graph.add_edge("_create_react", "_review_node")
        graph.set_finish_point("_response_node")

        graph.add_conditional_edges(
            "_review_node",
            should_continue,
            {
                "_create_react": "_create_react",
                "_response_node": "_response_node",
            },
        )
        self._action = graph.compile(checkpointer=self.checkpointer)
        # self._action.get_graph().draw_mermaid_png(output_file_path="./websearch_agent_graph.png", draw_method=MermaidDrawMethod.PYPPETEER)

    def _invoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 1000, **_
    ):
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )
        return self._action.invoke(inputs, config)


def process_content(
    url: str, context: str, state: Annotated[dict, InjectedState]
) -> str:
    """
    Processes content from a given webpage.

    Args:
        url: string with the url to obtain text content from.
        context: string summary of the information the agent wants from the url for summarizing salient information.
    """
    print("Parsing information from ", url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    content_prompt = f"""
    Here is the full content:
    {soup.get_text()}

    Carefully summarize the content in full detail, given the following context:
    {context}
    """
    summarized_information = (
        state["model"]
        .invoke(
            content_prompt, {"configurable": {"thread_id": state["thread_id"]}}
        )
        .content
    )
    return summarized_information


search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)
# search_tool = TavilySearchResults(max_results=10, search_depth="advanced",include_answer=True)


def should_continue(state: WebSearchState):
    if len(state["messages"]) > (state.get("max_websearch_steps", 100) + 3):
        return "_response_node"
    if "[APPROVED]" in state["messages"][-1].content:
        return "_response_node"
    return "_create_react"
