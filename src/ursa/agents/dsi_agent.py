import sys
import os
from time import time as now
from typing import Annotated, Any, Dict, Mapping, TypedDict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown as RichMarkdown

from langchain_core.tools import Tool
from langchain.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode

from mistune import Markdown
from networkx import display

from ursa.tools.search_tools import (
    run_arxiv_search,
    run_osti_search,
    run_web_search,
)

from .base import BaseAgent

from dsi.dsi import DSI


# define some global variables
dsi_store = None
db_schema = ""
db_description = None
master_db_folder = ""


def load_db_description(db_path: str) -> str:
    """Load the database description from a YAML file when provided with the path to a DSI database.

    Arg:
        db_path (str): the path of the DSI database
    Returns:
        str: message indicating success or failure
    """

    global db_description

    try:
        db_description_path = db_path.rsplit(".", 1)[0] + '_description.yaml'
        print(f"load_db_description :: db_description_path: {db_description_path}")
        
        with open(db_description_path, "r") as f:
            db_description = f.read()

        return "Successfully loaded database description."
    except:
        db_description = None
        return "Failed to load database description."



def load_dsi(path: str) -> str:
    """Load a DSI object from the path and add information to the context for the llm to use.

    Arg:
        path (str): the path to the DSI object
        
    Returns:
        str: message indicating success or failure
    """

    global dsi_store
    global db_schema
    global master_db_folder
    

    p = Path(path)
    if not p.is_absolute():
        db_path = str(master_db_folder + '/' + path)
    else:
        db_path = path

    data_path = db_path.strip()
    

    # Check if the path exists and there is data
    
    if not os.path.exists(data_path):
        return f"Failed to extract database information at: {data_path}"
    
    else:
        try:
            temp_store = DSI(data_path, backend_name = "Sqlite", check_same_thread=False)
            temp_tables = temp_store.list(True) # force things to fail if the table is empty
            temp_store.close()
        except Exception as e:
            return f"Failed to extract database information at: {data_path}"
            
            
    # if the above works, actually load the DSI object
    try:
        if dsi_store is not None:
            dsi_store.close()

        dsi_store = DSI(data_path, backend_name = "Sqlite", check_same_thread=False)
        tables = dsi_store.list(True)
        schema = dsi_store.schema()


        # Append the database information to the context
        print(f"""load_dsi :: Loading {dsi_store}; the DSI Object has a database that has {len(tables)} tables: {tables}.
        The schema of the database is {schema} \n""")
        
        db_schema += f"""The DSI Object has a database that has {len(tables)} tables: {tables}.
            The schema of the database is {schema} \n"""
        
        # load the database description if available
        try:
            load_db_description(data_path)
        except Exception as e:
            print("load_dsi :: No master database description to load.")
        
        
        return "Successfully loaded DSI object and extracted database information"
        
    except Exception as e:
        return "Failed to extract database information"


load_dsi_tool = Tool(
    name="load_dsi_tool",
    description="Load a DSI object from a path and update LLM context.",
    func=load_dsi,
)


class DSIState(TypedDict):
    messages: Annotated[list, add_messages]
    response: str
    metadata: Dict[str, Any]
    thread_id: str


class DSIAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        db_index_name:str="", 
        output_mode:str="jupyter",
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        
        self.store = None
        self.tables = None
        self.schema = None
        self.msg = {}
        self.master_datbase_path = ""
        self.output_mode = output_mode
        
        global master_db_folder
        
        # Get absolute path of dataset
        relative_db_path = Path(db_index_name)
        absolute_db_path = str(relative_db_path.resolve())
        master_db_folder = "/".join(absolute_db_path.split("/")[:-1]) + '/'


        self.load_master_db(absolute_db_path)

        self.tools = [
            run_web_search,
            run_osti_search,
            run_arxiv_search,
            load_dsi_tool,
        ]

        self.prompt = f"""
        You are a data-analysis agent who can write python code, SQL queries, and generate plots to answer user questions based on the data available in a DSI object.
        Use the load_dsi_tool tool to load DSI files that have a .db extension
        The currently loaded DSI object is stored in dsi_store global variable. Use query_dsi_tool to run SQL queries on it.
        When a user asks for data or dataset or ... you have, do NOT list the schema or metadata information you have about tables. Query the DSI objects for data and list the data in the tables.
        
        You can:
        - write and execute Python code,
        - compose SQL statements,
        - generate plots and diagrams,
        - analyze and summarize data.

        The dsi_explorer master dataset is avilable at {self.master_datbase_path} in case you need to reload it.

        Requirements:
        - Planning: Think carefully about the problem, but **do not show your reasoning**.
        - Data:
            - Always use the provided tools when available — never simulate results.
            - Never fabricate or assume sample data. Query or compute real values only.
            - When creating plots or files, always save them directly to disk (do not embed inline output).
            - Do not infer or assume any data beyond what is provided by the tools.
        - Keep all responses concise and focused on the requested task.
        - Do not restate the prompt or reasoning; just act and report the outcome briefly.
        """


        self.llm = self.llm.bind_tools(self.tools)
        self._build_graph()
        print(f"Dataset {db_index_name} has been loaded.\nThe DSI Data Explorer agent is ready.")


    def load_master_db(self, master_db_path: str) -> None:
        """Load the  master dataset from the given path.
        
        Arg:
            path (str): the path to the DSI object
        """
        
        if master_db_path != "":
            output_msg = load_dsi(master_db_path)
            if "Failed" in output_msg:
                print(f"[ERROR] A valid DSI file is needed. {master_db_path}. We will now exit!")
                raise ValueError("No DSI database path provided.")

            self.master_datbase_path = master_db_path
        else:        
            print("No DSI database provided. We will now exit!")
            raise RuntimeError(
            f"A valid DSI file is required. Failed to load: {master_db_path}"
        )

    # __call__ from my agent
    def _response_node(self, state):
        messages = state["messages"]

        conversation = [SystemMessage(content=self.prompt)] + messages
        response = self.llm.invoke(conversation)

        return {
            "messages": messages + [response],
            "response": response.content,
            "metadata": response.response_metadata
        }
        
    
    def _build_graph(self):
        graph = StateGraph(DSIState)
        self.add_node(graph, self._response_node)
        graph.set_entry_point("_response_node")
        graph.set_finish_point("_response_node")
        self._action = graph.compile(checkpointer=self.checkpointer)


    # matches __call__
    def _invoke(self, inputs: Mapping[str, Any], recursion_limit: int = 1000, **_):
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )
        return self._action.invoke(inputs, config)
    
    
    def craft_message(self, human_msg):
        """Craft the message with context if available."""

        global db_schema
        global db_description

        base_system_context = f"""
            The following phrases all refer to this same database:
            - "master database"
            - "master dataset"
            - "DSIExplorer master database"
            - "Diana dataset

            When the user asks to reload, refresh, reset, reinitialize, restart, or 
            the **master database**, interpret that as a request to reload the 
            DSIExplorer master database using the tool load_dsi_tool("{self.master_datbase_path}"),
            load the last dataset in the context.

            Do no reload or load the master dataset unless explicitly asked by the user.
        """

        # Build remaining dynamic system context parts
        system_parts = [base_system_context]

        if db_schema != "":
            system_parts.append("You have this dataset loaded: " + db_schema)

        if db_description is not None:
            system_parts.append("Dataset description: " + db_description)

        # Combine
        if system_parts:
            system_message = SystemMessage(content="\n\n".join(system_parts))
            messages = [system_message, HumanMessage(content=human_msg)]
        else:
            messages = [HumanMessage(content=human_msg)]

        # clear
        db_schema = ""
        db_description = None
        

        return {"messages": messages}

    
    def ask(self, user_query) -> None:
        """Ask a question to the DSI Explorer agent.

        Arg:
            user_query (str): the user question
        """

        start = now()

        msg = self.craft_message(user_query)

        result = self._invoke(
            msg,
            config={"configurable": {"thread_id": self.thread_id}}
        )

        # Get and display the cleaned output
        response_text = result["response"] 
        cleaned_output = response_text.strip()
        if self.output_mode == "jupyter":
            display(Markdown(cleaned_output))
        elif self.output_mode == "console":
            console = Console()
            md = RichMarkdown(cleaned_output)
            console.print(md)
            #print("\n" + cleaned_output)
        else:
            print(cleaned_output)


        elapsed = now() - start
        total_tokens = str(result["metadata"].get("token_usage", {}).get("total_tokens", 0)).strip()

        print(f"\nQuery took: {elapsed:.2f} seconds, total tokens used: {total_tokens}.\n")



# def main():
#     model = ChatOpenAI( model="gpt-5-mini", max_tokens=10000, timeout=None, max_retries=2)
#     dsi_explorer = DSIAgent(llm=model)
#     problem_string = "What is your name?"
#     print("Prompt: ", problem_string)
#     result = dsi_explorer.invoke(problem_string)
#     return result["messages"][-1].content


# if __name__ == "__main__":
#     print("Response: ", main())
