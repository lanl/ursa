from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
from langchain_core.tools import tool

from dsi.dsi import DSI

_NULL = io.StringIO()  # Hides DSI outout

########################################################################
#### Utility functions

def _load_db_description(db_path: str) -> str:
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
    except Exception:
        return ""



def _check_db_valid(db_path: str) -> bool:
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
            with redirect_stdout(_NULL), redirect_stderr(_NULL):
                temp_store = DSI(db_path, check_same_thread=False)
                temp_store.list(True) # force things to fail if the table is empty
                temp_store.close()
                    
        except Exception:
            return False

    return True



def _get_db_info(db_path: str) -> tuple[list, dict, str]:
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
    
    if _check_db_valid(db_path) is False:
        return tables, schema, desc
    
    try:
        with redirect_stdout(_NULL), redirect_stderr(_NULL):
            _dsi_store = DSI(db_path, check_same_thread=False)
            tables = _dsi_store.list(True)
            schema = _dsi_store.schema()
            desc = _load_db_description(db_path)
            _dsi_store.close()
            
        return tables, schema, desc

    except Exception:
        return tables, schema, desc
    
    
    
def _get_db_abs_path(db_path: str, run_path: str) -> [str, str]:
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
        master_database_path, master_db_folder = _get_db_abs_path(path, run_path)
        data_path = master_database_path.strip()
    else:
        p = Path(path)
        if not p.is_absolute():
            db_path = str(master_db_folder + '/' + path)
        else:
            db_path = path

        data_path = db_path.strip()
        

    if not _check_db_valid(data_path):
        return f"Failed to load DSI database at: {data_path}. Please check the path and ensure it points to a valid DSI .db file."
            
    try:
        _, _db_schema, _db_description = _get_db_info(data_path)
        _current_db_abs_path = data_path

        if master_database_previously_set is False:
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
        return f"Failed to load database information: {e}"
    


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
        with redirect_stdout(_NULL), redirect_stderr(_NULL):
            _store = DSI(db_path, check_same_thread=False)
            df = _store.query(query_str, collection=True)
                
        if df is None:
            return {}
        return df.to_dict(orient="records")
    
    except Exception:   
        return {}
    
    finally:
        if _store is not None:
            try:
                with redirect_stdout(_NULL), redirect_stderr(_NULL):
                    _store.close()
            except Exception:
                pass
