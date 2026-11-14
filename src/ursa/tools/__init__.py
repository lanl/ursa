"""Load tool implementations only when requested."""

from importlib import import_module

_LAZY_TOOLS = {
    "load_dsi_tool": ".dsi_search_tools",
    "query_dsi_tool": ".dsi_search_tools",
    "edit_experience": ".experience_tools",
    "list_experiences": ".experience_tools",
    "read_experience": ".experience_tools",
    "write_experience": ".experience_tools",
    "download_file_tool": ".read_file_tool",
    "read_file": ".read_file_tool",
    "get_supported_term_tools": ".term_tool",
    "run_command": ".run_command_tool",
    "list_workspace_files": ".workspace_tools",
    "edit_code": ".write_code_tool",
    "write_code": ".write_code_tool",
    "write_code_with_repo": ".write_code_tool",
}

__all__ = list(_LAZY_TOOLS)


def __getattr__(name):
    if name not in _LAZY_TOOLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_TOOLS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
