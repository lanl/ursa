from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.load import dumps
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.base import BaseCheckpointSaver


class BaseAgent:
    # llm: BaseChatModel
    # llm_with_tools: Runnable[LanguageModelInput, BaseMessage]

    def __init__(
        self,
        llm: str | BaseChatModel,
        checkpointer: BaseCheckpointSaver = None,
        **kwargs,
    ):
        match llm:
            case BaseChatModel():
                self.llm = llm

            case str():
                self.llm_provider, self.llm_model = llm.split("/")
                self.llm = ChatLiteLLM(
                    model=llm,
                    max_tokens=kwargs.pop("max_tokens", 10000),
                    max_retries=kwargs.pop("max_retries", 2),
                    **kwargs,
                )

            case _:
                raise TypeError(
                    "llm argument must be a string with the provider and model, or a BaseChatModel instance."
                )

        self.checkpointer = checkpointer
        self.thread_id = self.__class__.__name__

    def write_state(self, filename, state):
        json_state = dumps(state, ensure_ascii=False)
        with open(filename, "w") as f:
            f.write(json_state)
