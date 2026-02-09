import json
import sys
import os
import random
from time import time as now
from typing import Annotated, Any, Dict, Mapping, TypedDict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown as RichMarkdown

from langchain_core.tools import tool
from langchain_core.tools import Tool
from langchain.chat_models import BaseChatModel
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph
from contextlib import redirect_stdout, redirect_stderr

from ursa.tools.search_tools import (
    run_arxiv_search,
    run_osti_search,
    run_web_search,
)

from ursa.tools.read_file_tool import (
    download_file_tool,
)

from .base import BaseAgent

from dsi.dsi import DSI






########################################################################
#### Utility functions

def extract_message_texts(result) -> str:
    """Extract readable message content from either a dict or list of LangGraph messages."""
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
    elif isinstance(result, list):
        messages = result
    else:
        messages = []

    lines = []
    for m in messages:
        if hasattr(m, "content"):
            text = m.content
        elif isinstance(m, dict) and "content" in m:
            text = m["content"]
        else:
            text = m

        # Ensure text is a string (ToolMessage content can be list/dict)
        if isinstance(text, (list, dict)):
            text = json.dumps(text, indent=2, default=str)
        else:
            text = str(text)

        text = text.replace("\\n", "\n")
        lines.append(text.strip())

    return "\n\n".join(lines)



def load_db_description(db_path: str) -> str:
    """Load the database description from a YAML file when provided with the path to a DSI database.

    Arg:
        db_path (str): the absolute path of the DSI database
        
    Returns:
        str: message indicating success or failure
    """

    try:
        # The description file is expected to be in the same directory as the database, with the same name but ending in '_description.yaml'
        db_description_path = db_path.rsplit(".", 1)[0] + '_description.yaml'
        
        with open(db_description_path, "r") as f:
            db_desc = f.read()

        return str(db_desc)
    except:
        return ""



def check_db_valid(db_path: str) -> bool:
    """Check if the provided path points to a valid DSI database.

    Arg:
        db_path (str): the absolute path of the DSI database
        
    Returns:
        bool: True if the database is valid, False otherwise
    """
    if not os.path.exists(db_path):
        return False
    else:
        try:
            with open(os.devnull, "w") as fnull:
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    temp_store = DSI(db_path, check_same_thread=False)
                    temp_tables = temp_store.list(True) # force things to fail if the table is empty
                    temp_store.close()
                    
        except Exception as e:
            return False

    return True



def get_db_info(db_path: str) -> tuple[list, dict, str]:
    """Load the database information (tables and schema) from a DSI database.

    Arg:
        db_path (str): the absolute path of the DSI database    
        
    Returns:
        list: the list of tables in the database
        dict: the schema of the database
        str: the description of the database (if available, otherwise empty string)
    """
    
    tables = []
    schema = {}
    desc = ""
    
    if check_db_valid(db_path) == False:
        return tables, schema, desc
    
    try:
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                _dsi_store = DSI(db_path, check_same_thread=False)
                tables = _dsi_store.list(True)
                schema = _dsi_store.schema()
                desc = load_db_description(db_path)
                _dsi_store.close()
             
            return tables, schema, desc

    except Exception as e:
        return tables, schema, desc
    
    
    
def get_db_abs_path(db_path: str, run_path: str) -> [str, str]:
    """Get the absolute path of a DSI database given its path.

    Arg:
        db_path (str): the path of the DSI database (can be relative or absolute)
        run_path (str): the path of the codebase to resolve relative paths against
        
    Returns:
        str: the absolute path of the DSI database
        str: the absolute path of the folder containing the DSI database
    """
    
    p = Path(db_path)
    if not p.is_absolute():
        master_database_path = str( (Path(run_path) / db_path).expanduser() )
        master_db_folder = "/".join(master_database_path.split("/")[:-1]) + '/'
    else:
        master_database_path = db_path
        master_db_folder = "/".join(master_database_path.split("/")[:-1]) + '/'
        
    return master_database_path, master_db_folder


########################################################################
#### Tools Available

@tool
def load_dsi_tool(path: str, run_path: str = "", master_db_folder: str = "") -> dict:
    """Load a DSI object from the path and add information to the context for the llm to use.

    Arg:
        path (str): the path to the DSI object to load
        run_path (str): the path this code is being run from
        master_db_folder (str): the folder containing the master database, used to resolve relative paths when loading new databases
        
    Returns:
        str: message indicating success or failure
    """

    master_database_previously_set = True
    if master_db_folder == "":
        master_database_previously_set = False
        # the ai is loading the master database for the first time
        master_database_path, master_db_folder = get_db_abs_path(path, run_path)
        data_path = master_database_path.strip()
    else:
        p = Path(path)
        if not p.is_absolute():
            db_path = str(master_db_folder + '/' + path)
        else:
            db_path = path

        data_path = db_path.strip()
        

    if not check_db_valid(data_path):
        return f"Failed to load DSI database at: {data_path}. Please check the path and ensure it points to a valid DSI .db file."
            
    try:
        _, _db_schema, _db_description = get_db_info(data_path)
        _current_db_abs_path = data_path

        if master_database_previously_set == False:
            return {
                "the current working database path (current_db_abs_path) is": _current_db_abs_path,
                "the master database path (master_database_path) is": master_database_path,
                "the master databse folder (master_db_folder) is": master_db_folder,
                "the current databse schema is": _db_schema,
                "the database description is": _db_description
            }
        else:
            return {
                "the current working database path (current_db_abs_path) is": _current_db_abs_path,
                "the current databse schema is": _db_schema,
                "the database description is": _db_description
            }
        
        
    except Exception as e:
        return "Failed to load database information"
    


@tool
def query_dsi_tool(query_str: str, db_path: str) ->dict:
    """Execute a SQL query on a DSI object

    Arg:
        query_str (str): the SQL query to run on DSI object
        db_path (str): the absolute path to the DSI database to query

    Returns:
        collection: the results of the query
    """

    #print(f"query {query_str}, db_path: {db_path}")
    
    _store = None
    try:
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                _store = DSI(db_path, check_same_thread=False)
                df = _store.query(query_str, collection=True)
                
        if df is None:
            return {}
        return df.to_dict(orient="records")
    
    except Exception as e:   
        return {}
    
    finally:
        if _store is not None:
            try:
                with open(os.devnull, "w") as fnull:
                    with redirect_stdout(fnull), redirect_stderr(fnull):
                        _store.close()
            except Exception:
                pass



class DSIState(TypedDict):
    messages: Annotated[list, add_messages]
    response: str
    metadata: Dict[str, Any]
    thread_id: str



def should_call_tools(state: DSIState) -> str:
    """Decide whether to call tools or continue.
    
    Arg:
        state (State): the current state of the graph   
        
    Returns:
        str: "call_tools" or "continue"
    """
    
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "call_tools"
    
    return "continue"





class DSIAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        database_path:str="", 
        output_mode:str="jupyter",
        run_path: str = "",
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        
        self.db_schema = ""
        self.db_description = ""
        
        self.master_db_folder = ""
        self.master_database_path = ""
        self.current_db_abs_path = ""

        self.msg = {}
        self.output_mode = output_mode
        
        # Try to load the master database if a path is provided, otherwise wait for the user to load one
        if run_path == "":
            self.run_path = os.getcwd()
        else:
            self.run_path = run_path
        self.load_master_db(database_path)

        self.tools = [
            run_web_search,
            run_osti_search,
            run_arxiv_search,
            load_dsi_tool,
            query_dsi_tool,
            download_file_tool,
        ]

        self.prompt = f"""
        You are a data-analysis agent who can write python code, SQL queries, and generate plots to answer user questions based on the data available in a DSI object.
        Use the load_dsi_tool tool to load DSI files that have a .db extension
        Use query_dsi_tool to run SQL queries on it
        When a user asks for data or dataset or ... you have, do NOT list the schema or metadata information you have about tables. Query the DSI objects for data and list the data in the tables.
        
        You can:
        - write and execute Python code,
        - compose SQL statements but ONLY select; no update or delete
        - generate plots and diagrams,
        - analyze and summarize data.

        Requirements:
        - Planning: Think carefully about the problem, but **do not show your reasoning**.
        - Data:
            - Always use the provided tools when available — never simulate results.
            - Never fabricate or assume sample data. Query or compute real values only.
            - When creating plots or files, always save them directly to disk (do not embed inline output).
            - Do not infer or assume any data beyond what is provided by the tools.
        - Keep all responses concise and focused on the requested task.
        - Only load a dataset when explicitly asked by a user
        - Do not restate the prompt or reasoning; just act and report the outcome briefly.
        """

        self.thread_id = str(random.randint(1, 20000))
        self.llm = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self._build_graph()




    def load_master_db(self, master_database: str) -> None:
        """Load the  master dataset from the given path.
        
        Arg:
            master_database (str): the path to the DSI object
        """
        
        if master_database == "":        
            print("No DSI database provided. Please load one")
            return

        _master_database_path, _master_db_folder = get_db_abs_path(master_database, self.run_path)
        absolute_db_path = _master_database_path
        
        if check_db_valid(absolute_db_path):
            self.db_tables, self.db_schema, self.db_description = get_db_info(absolute_db_path)
            
            # set the values now that we know things are correct
            self.current_db_abs_path =  absolute_db_path
            self.master_database_path = absolute_db_path
            self.master_db_folder = _master_db_folder

        else:        
            print("No valid DSI database provided. Please load one")
            #sys.exit(1)




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
        self.graph = StateGraph(DSIState)

        self.graph.add_node("response", self._response_node)
        self.graph.add_node("tools", self.tool_node)

        self.graph.add_edge(START, "response")
        self.graph.add_conditional_edges(
            "response",
            should_call_tools,
            {
                "call_tools": "tools",
                "continue": END,
            },
        )
        self.graph.add_edge("tools", "response")

        self.action = self.graph.compile(checkpointer=self.checkpointer)



    # matches __call__
    def _invoke(self, inputs: Mapping[str, Any], recursion_limit: int = 1000, **_):
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )
        return self.action.invoke(inputs, config)
    
    
    def craft_message(self, human_msg):
        """Craft the message with context if available."""

        base_system_context = f"""
            The following phrases all refer to this same database:
            - "master database"
            - "master dataset"
            - "DSIExplorer master database"
            - "Diana dataset

            When the user asks to reload, refresh, reset, reinitialize, restart, or 
            the **master database**, interpret that as a request to reload the 
            DSIExplorer master database using the tool load_dsi_tool("{self.master_database_path}"),
            load the last dataset in the context.

            Do no reload or load the master dataset unless explicitly asked by the user.
        """

        # Build remaining dynamic system context parts
        system_parts = [base_system_context]
        
        if self.current_db_abs_path != "":
            system_parts.append("The current working database path (current_db_abs_path) is: " + self.current_db_abs_path)
            
        if self.master_database_path != "":
            system_parts.append("The master database path (master_database_path) is: " + self.master_database_path)
            
        if self.db_schema != "":
            system_parts.append("The schema of the dataset loaded: " + self.db_schema)

        if self.db_description != "":
            system_parts.append("Dataset description: " + self.db_description)

        # Combine
        if system_parts:
            system_message = SystemMessage(content="\n\n".join(system_parts))
            messages = [system_message, HumanMessage(content=human_msg)]
        else:
            messages = [HumanMessage(content=human_msg)]


        # clear
        self.db_schema = ""
        self.db_description = ""
        self.master_database_path = ""
        self.current_db_abs_path = ""
        

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
            from IPython.display import display, Markdown
            display(Markdown(cleaned_output))
        elif self.output_mode == "console":
            console = Console()
            md = RichMarkdown(cleaned_output)
            console.print(md)
        else:
            print(cleaned_output)


        elapsed = now() - start
        total_tokens = str(result["metadata"].get("token_usage", {}).get("total_tokens", 0)).strip()

        print(f"\nQuery took: {elapsed:.2f} seconds, total tokens used: {total_tokens}.\n")

