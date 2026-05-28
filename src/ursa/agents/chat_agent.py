from typing import Annotated, Literal, TypedDict

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from ursa.agents.base import AgentContext, AgentWithTools, BaseAgent
from ursa.prompt_library.chatter_prompts import get_chatter_system_prompt
from ursa.tools import (
    edit_code,
    edit_experience,
    list_experiences,
    read_experience,
    read_file,
    run_command,
    write_code,
    write_experience,
)
from ursa.tools.read_image_tool import read_image_tool
from ursa.tools.search_tools import (
    run_arxiv_search,
    run_osti_search,
    run_web_search,
)


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str


class BasicChatAgent(BaseAgent[ChatState]):
    """Basic Chat Agent"""

    state_type = ChatState

    def _response_node(
        self, state: ChatState, runtime: Runtime[AgentContext]
    ) -> ChatState:
        new_state, full_overwrite = self.prepare_messages_context(state)
        res = self.llm.invoke(new_state["messages"])
        return self.messages_update(
            new_state, [res], full_overwrite=full_overwrite
        )

    def format_query(self, prompt: str, state: ChatState | None = None):
        if state is None:
            state = ChatState(
                messages=[SystemMessage(content=get_chatter_system_prompt())]
            )
        state["messages"].append(HumanMessage(content=prompt))

        return state

    def format_result(self, result: ChatState) -> str:
        return result["messages"][-1].text

    def _build_graph(self):
        self.add_node(self._response_node)
        self.graph.set_entry_point("_response_node")
        self.graph.set_finish_point("_response_node")


def should_continue(state: ChatState) -> Literal["finish", "continue"]:
    """Return 'finish' if no tool calls in the last message, else 'continue'.

    Args:
        state: The current execution state containing messages.

    Returns:
        A literal "finish" if the last message has no tool calls,
        otherwise "continue".
    """
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "finish"
    # Otherwise if there is, we continue
    else:
        return "continue"


class ChatAgent(AgentWithTools, BasicChatAgent):
    """Chat Agent"""

    state_type = ChatState

    def __init__(
        self,
        llm: BaseChatModel,
        use_web: bool = False,
        **kwargs,
    ):
        default_tools = [
            run_command,
            write_code,
            edit_code,
            read_file,
            read_image_tool,
            list_experiences,
            write_experience,
            read_experience,
            edit_experience,
        ]
        if use_web:
            default_tools.extend([
                run_web_search,
                run_osti_search,
                run_arxiv_search,
            ])
        super().__init__(llm=llm, tools=default_tools, **kwargs)

    def _build_graph(self):
        # Bind tools to llm and context summarizer
        self.llm = self.llm.bind_tools(self.tools.values())

        self.add_node(self._response_node, "respond")
        self.add_node(self.tool_node, "tool_node")
        self.graph.set_entry_point("respond")
        self.graph.add_conditional_edges(
            "respond",
            self._wrap_cond(should_continue, "should_continue"),
            {"continue": "tool_node", "finish": END},
        )
        self.graph.add_edge("tool_node", "respond")
