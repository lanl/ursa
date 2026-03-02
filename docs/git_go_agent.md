# GitGoAgent Documentation

`GitGoAgent` is a specialized execution agent for git-managed Go repositories. It combines git control, Go development tools, code linting, and file operations to enable autonomous development workflows on Go projects.

## Basic Usage

```python
from pathlib import Path

from langchain.chat_models import init_chat_model
from ursa.agents import GitGoAgent

# Initialize the agent
agent = GitGoAgent(
    llm=init_chat_model("openai:gpt-5-mini"),
    workspace=Path("/path/to/your/repo"),
)

# Run a request
result = agent.invoke(
    "Run tests, lint the code, and commit any fixes."
)
print(result["messages"][-1].text)
```

## Tools Available

### Git Operations
- `git_status`: Show repository status.
- `git_diff`: Show diffs (staged or unstaged).
- `git_log`: Show recent commits.
- `git_ls_files`: List tracked files.
- `git_add`: Stage files for commit.
- `git_commit`: Create a commit.
- `git_switch`: Switch branches (optionally creating new ones).
- `git_create_branch`: Create a branch without switching.

### Go Build and Test Tools
- `go_build`: Build the module using `go build ./...`
- `go_test`: Run tests with `go test ./...` (supports verbose mode)
- `go_vet`: Run Go vet for code pattern analysis
- `go_mod_tidy`: Validate and clean module dependencies

### Code Quality Tools
- `golangci_lint`: Run golangci-lint on the repository
  - Automatically detects and uses `.golangci.yml` if present
  - Falls back to default linter configuration if config file missing
  - Provides helpful error messages if golangci-lint is not installed
- `gofmt_files`: Format .go files in-place using gofmt

### File Operations
- `read_file`: Read file contents
- `write_code`: Write new files (with path validation)
- `edit_code`: Edit existing files (with path validation)

## Configuration and Behavior

### Timeouts
Operations use differentiated, operation-specific timeouts (not a unified timeout):

| Operation | Timeout | Rationale |
|-----------|---------|-----------|
| Git commands | 30 seconds | Should be near-instant; timeout indicates hanging (waiting for input or wrong directory) |
| Code formatting (`gofmt`) | 30 seconds | Usually fast operation |
| Code analysis (`go vet`, `go mod tidy`) | 60 seconds | Analysis is typically quick |
| Go build | 5 minutes (300s) | Builds on large codebases can be slow |
| Go test | 10 minutes (600s) | Test suites can legitimately take time |
| Linting (`golangci-lint`) | 3 minutes (180s) | Comprehensive linting takes moderate time |

**Design note:** If git commands timeout, it typically indicates:
- The agent is running commands in the wrong directory (should use `repo_path` parameter)
- Git is waiting for interactive input (e.g., passphrase, editor)
- Network issues when accessing remote repos

If other operations timeout, try running them in smaller chunks or profiling the specific operation.

### Path Safety
Both `write_code` and `edit_code` validate file paths to prevent:
- **Path traversal attacks** (e.g., `../../../etc/passwd` attempts are rejected)
- **Writes outside the workspace** (all files must be within the workspace directory)
- **Writes outside the repository** (optional, when `repo_path` parameter is used)

Example: Specifying a repo boundary ensures all file modifications stay within that repository.

### Golangci-lint Integration

The agent automatically integrates with `golangci-lint` for code quality checks:

```python
# Agent detects .golangci.yml and uses it automatically
agent.invoke("Run linting and report all issues")

# Linter configuration is respected while agent iterates on fixes
```

**Install golangci-lint** if not already present:
```bash
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
```

The linter supports:
- Custom linter configurations via `.golangci.yml`
- Extensibility to additional linters in future versions
- Clear error reporting when linter is misconfigured

## Common Workflows

### 1. Build and Test Validation
```python
agent.invoke(
    "Build the module, run all tests, and report any failures."
)
```

### 2. Code Quality Check and Fix
```python
agent.invoke(
    "Run golangci-lint, identify issues, and attempt to fix them automatically."
)
```

### 3. Feature Implementation with Git Integration
```python
agent.invoke(
    "Create a new feature branch, implement the requested functionality, "
    "run tests and linting, and commit the changes."
)
```

### 4. Dependency Management
```python
agent.invoke(
    "Run go mod tidy to clean up dependencies, then commit the changes."
)
```

## Notes

- Operates only inside the configured workspace
- All file writes are validated against workspace and optionally repository boundaries
- Avoids destructive git operations by design (no force pushes, rebases, etc.)
- Supports subdirectory repositories via `repo_path` parameter on tools
- Explicit timeout handling prevents the agent from hanging on slow operations
- All tool output (stdout/stderr) is captured and returned to the agent for analysis
