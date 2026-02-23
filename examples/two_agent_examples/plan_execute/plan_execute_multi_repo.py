import argparse
import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Iterable

import randomname
import yaml
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rich import box, get_console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ursa.agents import WebSearchAgent, make_git_agent
from ursa.prompt_library.planning_prompts import reflection_prompt
from ursa.util.github_research import gather_github_context

console = get_console()


class RepoStep(BaseModel):
    repo: str = Field(description="Target repo name from the provided list")
    name: str = Field(description="Short, specific step title")
    description: str = Field(description="Detailed description of the step")
    requires_code: bool = Field(
        description="True if this step needs code to be written/run"
    )
    expected_outputs: list[str] = Field(
        description="Concrete artifacts or results produced by this step"
    )
    success_criteria: list[str] = Field(
        description="Measurable checks that indicate the step succeeded"
    )
    depends_on_repos: list[str] = Field(
        default_factory=list,
        description=(
            "Repo names that must complete ALL their steps before this step "
            "can start. Use this when this step depends on changes made in "
            "another repo (e.g. a library repo must finish before consumer "
            "repos can integrate its changes)."
        ),
    )


class RepoPlan(BaseModel):
    steps: list[RepoStep] = Field(
        description="Ordered list of steps to solve the problem"
    )


def _hash_plan(plan_steps: Iterable[RepoStep]) -> str:
    serial = json.dumps(
        [step.model_dump() for step in plan_steps],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()


def _load_yaml(path: str) -> NS:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
            if not isinstance(raw_cfg, dict):
                raise ValueError("Top-level YAML must be a mapping/object.")
            return NS(**raw_cfg)
    except FileNotFoundError:
        print(f"Config file not found: {path}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"Error loading YAML: {exc}", file=sys.stderr)
        sys.exit(2)


def _resolve_workspace(user_workspace: str | None, project: str) -> Path:
    if user_workspace:
        workspace = Path(user_workspace)
    else:
        suffix = randomname.get_name(
            adj=(
                "colors",
                "emotions",
                "character",
                "speed",
                "size",
                "weather",
                "appearance",
                "sound",
                "age",
                "taste",
                "physics",
            ),
            noun=(
                "cats",
                "dogs",
                "apex_predators",
                "birds",
                "fish",
                "fruit",
                "seasonings"
            ),
        )
        workspace = Path(f"{project}_{suffix}")

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "repos").mkdir(exist_ok=True)
    return workspace


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    base = dict(base or {})
    override = dict(override or {})
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_llm_kwargs_for_agent(
    models_cfg: dict | None, agent_name: str | None
) -> dict:
    models_cfg = models_cfg or {}
    profiles = models_cfg.get("profiles") or {}
    defaults = models_cfg.get("defaults") or {}
    agents = models_cfg.get("agents") or {}

    merged = {}
    merged = _deep_merge_dicts(merged, defaults.get("params") or {})

    default_profile_name = defaults.get("profile")
    if default_profile_name and default_profile_name in profiles:
        merged = _deep_merge_dicts(merged, profiles[default_profile_name] or {})

    if agent_name and isinstance(agents, dict) and agent_name in agents:
        a = agents.get(agent_name) or {}
        agent_profile_name = a.get("profile")
        if agent_profile_name and agent_profile_name in profiles:
            merged = _deep_merge_dicts(
                merged, profiles[agent_profile_name] or {}
            )
        merged = _deep_merge_dicts(merged, a.get("params") or {})

    return merged


def _resolve_model_choice(model_choice: str, models_cfg: dict):
    if ":" in model_choice:
        alias, pure_model = model_choice.split(":", 1)
    else:
        alias, pure_model = "openai", model_choice

    providers = (models_cfg or {}).get("providers", {})
    prov = providers.get(alias, {})

    model_provider = prov.get("model_provider", alias)

    api_key = None
    if prov.get("api_key_env"):
        api_key = os.getenv(prov["api_key_env"])
    if not api_key and prov.get("token_loader"):
        mod, fn = prov["token_loader"].rsplit(".", 1)
        api_key = getattr(__import__(mod, fromlist=[fn]), fn)()

    provider_extra = {}
    if prov.get("base_url"):
        provider_extra["base_url"] = prov["base_url"]
    if api_key:
        provider_extra["api_key"] = api_key

    return model_provider, pure_model, provider_extra


def _validate_model(llm, model_name: str, role: str) -> None:
    """Send a minimal chat completion to verify the model is reachable and
    supports the chat completions endpoint.  Raises ``RuntimeError`` with a
    clear message on failure so the user can fix their config before burning
    tokens on a full run.
    """
    from langchain_core.messages import HumanMessage as _HM

    try:
        llm.invoke([_HM(content="ping")])
    except Exception as exc:
        msg = str(exc)
        # Surface the most common issues clearly
        if "not a chat model" in msg.lower() or "404" in msg:
            raise RuntimeError(
                f"Model '{model_name}' (role: {role}) is not available as a "
                f"chat model.  Check that the model ID is correct and that it "
                f"supports the v1/chat/completions endpoint.\n"
                f"  Original error: {msg}"
            ) from exc
        if "401" in msg or "auth" in msg.lower():
            raise RuntimeError(
                f"Authentication failed for model '{model_name}' (role: {role}).  "
                f"Check your API key.\n  Original error: {msg}"
            ) from exc
        # Re-raise anything else as-is
        raise RuntimeError(
            f"Failed to reach model '{model_name}' (role: {role}).\n"
            f"  Original error: {msg}"
        ) from exc


def setup_llm(
    model_choice: str,
    models_cfg: dict | None = None,
    agent_name: str | None = None,
):
    models_cfg = models_cfg or {}

    provider, pure_model, provider_extra = _resolve_model_choice(
        model_choice, models_cfg
    )

    base_llm_kwargs = {
        "max_completion_tokens": 10000,
        "max_retries": 2,
    }

    yaml_llm_kwargs = _resolve_llm_kwargs_for_agent(models_cfg, agent_name)
    llm_kwargs = _deep_merge_dicts(base_llm_kwargs, yaml_llm_kwargs)

    return init_chat_model(
        model=pure_model,
        model_provider=provider,
        **llm_kwargs,
        **(provider_extra or {}),
    )


def _resolve_repos(
    raw_repos: list[dict], config_dir: Path, workspace: Path
) -> list[dict]:
    repos = []
    for raw in raw_repos:
        if not isinstance(raw, dict):
            raise ValueError("Each repo entry must be a mapping/object.")

        name = raw.get("name")
        if not name:
            raise ValueError("Each repo requires a 'name'.")

        path_value = raw.get("path")
        if path_value:
            path = Path(path_value)
            if not path.is_absolute():
                path = (config_dir / path).resolve()
        else:
            # Default: clone into <workspace>/repos/<name>
            path = (workspace / "repos" / name).resolve()

        repos.append({
            "name": name,
            "path": path,
            "url": raw.get("url"),
            "branch": raw.get("branch"),
            "checkout": bool(raw.get("checkout", False)),
            "checks": raw.get("checks") or [],
            "description": raw.get("description") or "",
            "language": raw.get("language", "generic"),
        })
    return repos


def _run_command(
    args: list[str], cwd: Path | None = None, timeout: int = 600
) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            args,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=cwd,
        )
    except Exception as exc:
        return 1, "", f"Error: {exc}"
    return result.returncode, result.stdout, result.stderr


def _ensure_checkout(repo: dict) -> None:
    path = repo["path"]
    url = repo.get("url")
    branch = repo.get("branch")
    if not repo.get("checkout"):
        return

    if not path.exists():
        if not url:
            raise RuntimeError(
                f"Repo {repo['name']} missing locally and no url provided."
            )
        args = ["git", "clone"]
        if branch:
            args.extend(["--branch", branch])
        args.extend([url, str(path)])
        code, stdout, stderr = _run_command(args)
        if code != 0:
            raise RuntimeError(
                f"git clone failed for {repo['name']}\n{stdout}\n{stderr}"
            )
        return

    if not branch:
        return

    # Check current branch -- skip checkout if already on the right one
    code, current, _ = _run_command(
        ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"]
    )
    current = current.strip()
    if code == 0 and current == branch:
        console.print(
            f"  [dim]{repo['name']}:[/dim] already on [cyan]{branch}[/cyan]"
        )
        return

    # Attempt checkout
    code, stdout, stderr = _run_command(
        ["git", "-C", str(path), "checkout", branch]
    )
    if code == 0:
        return

    # Checkout failed -- likely dirty working tree.  Warn and continue on
    # the current branch rather than crashing the entire run.
    console.print(
        Panel(
            f"[bold yellow]{repo['name']}:[/bold yellow] "
            f"Could not checkout [cyan]{branch}[/cyan] "
            f"(staying on [cyan]{current}[/cyan]).\n\n"
            f"[dim]{stderr.strip()}[/dim]\n\n"
            "Tip: commit or stash local changes, or set "
            "[bold]checkout: false[/bold] in the config.",
            border_style="yellow",
            expand=False,
        )
    )


def _ensure_repo_symlink(workspace: Path, repo: dict) -> Path:
    repos_dir = workspace / "repos"
    repos_dir.mkdir(exist_ok=True)
    target = repos_dir / repo["name"]
    source = repo["path"]

    # If the repo was cloned directly into workspace/repos/<name>,
    # it's already in the right place -- no symlink needed.
    if target.resolve() == source.resolve():
        return target

    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return target
        raise RuntimeError(
            f"Repo link target already exists: {target}"
        )

    target.symlink_to(source, target_is_directory=True)
    return target


def _format_repo_list(repos: list[dict]) -> str:
    lines = []
    for repo in repos:
        desc = repo.get("description")
        extra = f" - {desc}" if desc else ""
        branch = repo.get("branch")
        branch_note = f" (branch: {branch})" if branch else ""
        lines.append(
            f"- {repo['name']}: repos/{repo['name']}{branch_note}{extra}"
        )
    return "\n".join(lines)


def _planner_prompt(problem: str, repos: list[dict], research: str | None) -> str:
    repo_block = _format_repo_list(repos)
    research_block = (
        f"\n\nResearch notes:\n{research}\n" if research else ""
    )
    repo_names = ", ".join([repo["name"] for repo in repos])
    return (
        "You are planning changes across multiple git repositories.\n"
        "Create a step-by-step plan that can be executed independently per repo.\n\n"
        f"Available repos (use repo field from this list only): {repo_names}\n"
        f"Repo details:\n{repo_block}\n\n"
        f"Problem:\n{problem}\n"
        f"{research_block}\n"
        "Rules:\n"
        "- Each step MUST include a 'repo' field matching one of the repo names.\n"
        "- If a task affects multiple repos, split it into separate steps per repo.\n"
        "- Prefer small, reviewable steps that can run in parallel across repos.\n"
        "- Include expected outputs and success criteria for each step.\n"
        "- Use 'depends_on_repos' to declare cross-repo dependencies.\n"
        "  When a step in repo B depends on changes made in repo A (e.g. repo B\n"
        "  consumes a library from repo A), set depends_on_repos: [A].\n"
        "  This means ALL steps in repo A must complete before this step starts.\n"
        "  Steps with no dependencies run in parallel. Only add dependencies\n"
        "  when there is a real build/import dependency, not just logical ordering.\n"
        "  CRITICAL RULES for depends_on_repos:\n"
        "  - NEVER create circular dependencies (A depends on B AND B depends on A).\n"
        "    Dependencies must form a DAG (directed acyclic graph).\n"
        "  - NEVER list a repo as depending on itself — steps within a repo are\n"
        "    already sequential and self-deps will be stripped.\n"
        "  - Dependencies flow one direction: libraries -> consumers, never back.\n"
    )


async def _gather_research(
    llm,
    workspace: Path,
    research_cfg: dict | None,
    problem: str,
    repos: list[dict] | None = None,
) -> str | None:
    sections: list[str] = []

    # -- GitHub context: auto-fetch issues/PRs from repo URLs --
    if repos:
        gh_cfg = (research_cfg or {}).get("github", {}) or {}
        if gh_cfg.get("enabled", True):
            max_issues = int(gh_cfg.get("max_issues", 10))
            max_prs = int(gh_cfg.get("max_prs", 10))
            gh_context = gather_github_context(
                repos, max_issues=max_issues, max_prs=max_prs
            )
            if gh_context:
                sections.append(
                    "# GitHub Repository Context\n\n" + gh_context
                )
                console.print(
                    "[green]Fetched GitHub issues/PRs for repos with GitHub URLs.[/green]"
                )
            else:
                console.print(
                    "[dim]No GitHub context available "
                    "(gh CLI missing or no GitHub URLs).[/dim]"
                )

    # -- Explicit web search queries (optional) --
    queries = (research_cfg or {}).get("queries") or []
    if queries:
        try:
            agent = WebSearchAgent(
                llm=llm,
                workspace=workspace,
                summarize=True,
                max_results=int((research_cfg or {}).get("max_results", 3)),
            )
        except Exception as exc:
            console.print(
                f"[bold yellow]WebSearchAgent unavailable:[/bold yellow] {exc}"
            )
            agent = None

        if agent:
            for query in queries:
                context = f"{problem}\n\nResearch focus: {query}"
                result = await agent.ainvoke(
                    {"query": query, "context": context}
                )
                summary = result.get("final_summary") or ""
                sections.append(f"Query: {query}\n{summary}")

    return "\n\n".join(sections) if sections else None


async def _plan(
    llm,
    problem: str,
    repos: list[dict],
    research: str | None,
    reflection_steps: int,
) -> RepoPlan:
    prompt = _planner_prompt(problem, repos, research)
    messages = [SystemMessage(content=prompt)]
    structured_llm = llm.with_structured_output(RepoPlan)
    plan = structured_llm.invoke(messages)

    for _ in range(max(0, reflection_steps)):
        review = llm.invoke(
            [
                SystemMessage(content=reflection_prompt),
                HumanMessage(content=plan.model_dump_json()),
            ]
        )
        review_text = (review.text or "").strip()
        if "[APPROVED]" in review_text:
            break
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"Reviewer notes:\n{review_text}\n\nRevise the plan."
            ),
        ]
        plan = structured_llm.invoke(messages)

    return plan


def _write_plan(workspace: Path, plan: RepoPlan) -> None:
    plan_path = workspace / "plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2))


def _render_repo_plan(plan: RepoPlan) -> None:
    """Display the multi-repo plan as a Rich table."""
    table = Table(
        title="[bold]Multi-Repo Plan[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#", style="bold cyan", no_wrap=True, width=3)
    table.add_column("Repo", style="bold yellow", no_wrap=True)
    table.add_column("Step", overflow="fold")
    table.add_column("Description", overflow="fold")
    table.add_column("Code?", justify="center", no_wrap=True)

    for i, step in enumerate(plan.steps, 1):
        code_badge = Text.from_markup(
            "[bold green]Yes[/]" if step.requires_code else "[dim]No[/dim]"
        )
        table.add_row(
            str(i),
            step.repo,
            step.name,
            step.description,
            code_badge,
        )

    console.print(table)

    # Summary: steps per repo
    repo_counts: dict[str, int] = {}
    for step in plan.steps:
        repo_counts[step.repo] = repo_counts.get(step.repo, 0) + 1
    summary_parts = [f"[bold]{name}[/bold]: {count}" for name, count in sorted(repo_counts.items())]
    console.print(
        Panel(
            "  ".join(summary_parts),
            title="[bold]Steps per repo[/bold]",
            border_style="cyan",
            expand=False,
        )
    )


def _group_steps_by_repo(plan: RepoPlan) -> dict[str, list[RepoStep]]:
    grouped: dict[str, list[RepoStep]] = {}
    for step in plan.steps:
        grouped.setdefault(step.repo, []).append(step)
    return grouped


def _validate_plan_repos(plan: RepoPlan, repos: list[dict]) -> None:
    repo_names = {repo["name"] for repo in repos}
    invalid = sorted({step.repo for step in plan.steps} - repo_names)
    if invalid:
        raise RuntimeError(
            "Plan referenced unknown repos: " + ", ".join(invalid)
        )
    # Validate and sanitize depends_on_repos
    plan_repos = {step.repo for step in plan.steps}
    for step in plan.steps:
        bad_deps = sorted(set(step.depends_on_repos) - plan_repos)
        if bad_deps:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] step '{step.name}' in "
                f"repo '{step.repo}' depends on repos not in the plan: "
                + ", ".join(bad_deps)
            )
        # Strip self-dependencies — steps within a repo are already sequential
        if step.repo in step.depends_on_repos:
            step.depends_on_repos = [
                d for d in step.depends_on_repos if d != step.repo
            ]
            console.print(
                f"[dim]Stripped self-dependency from step '{step.name}' "
                f"in repo '{step.repo}'[/dim]"
            )

    # Build repo-level dependency graph and break any cycles.
    # A repo A depends on repo B if ANY step in A lists B in depends_on_repos.
    dep_graph: dict[str, set[str]] = {name: set() for name in plan_repos}
    for step in plan.steps:
        for dep in step.depends_on_repos:
            if dep in plan_repos and dep != step.repo:
                dep_graph[step.repo].add(dep)

    # Detect and break cycles via DFS
    UNVISITED, IN_PROGRESS, DONE = 0, 1, 2
    state: dict[str, int] = {name: UNVISITED for name in dep_graph}
    broken_edges: list[tuple[str, str]] = []

    def _visit(node: str, stack: set[str]) -> None:
        state[node] = IN_PROGRESS
        stack.add(node)
        for dep in list(dep_graph.get(node, set())):
            if dep in stack:
                # Back-edge found — break the cycle by removing this edge
                dep_graph[node].discard(dep)
                broken_edges.append((node, dep))
            elif state[dep] == UNVISITED:
                _visit(dep, stack)
        stack.discard(node)
        state[node] = DONE

    for repo in list(dep_graph):
        if state[repo] == UNVISITED:
            _visit(repo, set())

    # Remove broken edges from the actual step data
    if broken_edges:
        broken_set = set(broken_edges)
        for step in plan.steps:
            removed = [d for d in step.depends_on_repos if (step.repo, d) in broken_set]
            if removed:
                step.depends_on_repos = [
                    d for d in step.depends_on_repos if (step.repo, d) not in broken_set
                ]
        edge_strs = [f"{a} -> {b}" for a, b in broken_edges]
        console.print(
            Panel(
                "Circular repo dependencies detected and broken:\n"
                + "\n".join(f"  {e}" for e in edge_strs)
                + "\n\nThese dependency edges were removed to prevent deadlock. "
                "Steps that lost dependencies will run without waiting.",
                title="[bold yellow]Cycle detected[/bold yellow]",
                border_style="yellow",
                expand=False,
            )
        )


def _progress_path(
    workspace: Path,
    repo_name: str,
    resume_dir: Path | None,
    resume_files: dict[str, Path],
) -> Path:
    if repo_name in resume_files:
        return resume_files[repo_name]

    progress_dir = resume_dir or (workspace / "progress")
    progress_dir.mkdir(exist_ok=True, parents=True)
    return progress_dir / f"{repo_name}.json"


def _load_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_progress(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _parse_resume_overrides(
    paths: list[str] | None, config_dir: Path
) -> tuple[Path | None, dict[str, Path]]:
    resume_dir: Path | None = None
    resume_files: dict[str, Path] = {}

    for raw in paths or []:
        path = Path(raw)
        if not path.is_absolute():
            path = (config_dir / path).resolve()

        if path.is_dir():
            if resume_dir and resume_dir != path:
                raise ValueError(
                    "Only one resume directory may be provided."
                )
            resume_dir = path
            continue

        if not path.exists():
            raise ValueError(f"Resume checkpoint not found: {path}")
        resume_files[path.stem] = path

    return resume_dir, resume_files


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as compact h:mm:ss or m:ss."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _init_progress(repo_steps: dict[str, list[RepoStep]]) -> dict[str, dict]:
    now = time.time()
    return {
        name: {
            "state": "queued",
            "step": 0,
            "total": len(steps),
            "current": None,
            "error": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "started": now,
            "step_started": None,
            "updated": now,
        }
        for name, steps in repo_steps.items()
    }


def _extract_agent_tokens(agent) -> dict[str, int]:
    """Read token usage from agent telemetry samples accumulated since last begin_run."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        for sample in agent.telemetry.llm.samples:
            rollup = (sample.get("metrics") or {}).get("usage_rollup") or {}
            totals["input_tokens"] += int(rollup.get("input_tokens", 0))
            totals["output_tokens"] += int(rollup.get("output_tokens", 0))
            totals["total_tokens"] += int(rollup.get("total_tokens", 0))
    except Exception:
        pass
    return totals


def _fmt_tokens(n: int) -> str:
    """Format token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


_STATE_STYLES = {
    "queued": ("dim", "..."),
    "blocked": ("bold yellow", "||"),
    "running": ("bold cyan", ">>"),
    "done": ("bold green", "ok"),
    "failed": ("bold red", "!!"),
}


def _build_progress_table(
    snapshot: dict[str, dict], max_parallel: int
) -> Table:
    counts: dict[str, int] = {"queued": 0, "blocked": 0, "running": 0, "done": 0, "failed": 0}
    grand_total_tokens = 0
    for info in snapshot.values():
        state = info.get("state", "queued")
        counts[state] = counts.get(state, 0) + 1
        grand_total_tokens += info.get("total_tokens", 0) + info.get("_live_total", 0)

    title = (
        f"[bold]active [cyan]{counts['running']}/{max_parallel}[/cyan]  "
        + (f"blocked [yellow]{counts['blocked']}[/yellow]  " if counts["blocked"] else "")
        + f"queued [dim]{counts['queued']}[/dim]  "
        f"done [green]{counts['done']}[/green]  "
        f"failed [red]{counts['failed']}[/red]  "
        f"tokens [yellow]{_fmt_tokens(grand_total_tokens)}[/yellow][/bold]"
    )

    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        header_style="bold magenta",
        expand=False,
        padding=(0, 1),
    )
    table.add_column("", no_wrap=True, width=2)
    table.add_column("Repo", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Progress", no_wrap=True)
    table.add_column("Elapsed", no_wrap=True, justify="right")
    table.add_column("Tokens", no_wrap=True, justify="right")
    table.add_column("Step", overflow="fold")

    now = time.time()
    for name in sorted(snapshot):
        info = snapshot[name]
        state = info.get("state", "queued")
        step = info.get("step", 0)
        total = info.get("total", 0)
        current = info.get("current") or ""
        error = info.get("error")
        in_tok = info.get("input_tokens", 0) + info.get("_live_input", 0)
        out_tok = info.get("output_tokens", 0) + info.get("_live_output", 0)

        style, icon = _STATE_STYLES.get(state, ("", "?"))

        progress_bar = ""
        if total:
            filled = int((step / total) * 10)
            progress_bar = f"{'█' * filled}{'░' * (10 - filled)} {step}/{total}"

        # Elapsed time: total for repo + current step duration
        elapsed_text = ""
        if state not in ("queued",):
            started = info.get("started") or now
            total_elapsed = now - started
            elapsed_text = f"[dim]{_fmt_elapsed(total_elapsed)}[/dim]"
            step_started = info.get("step_started")
            if step_started and state in ("running", "blocked"):
                step_elapsed = now - step_started
                elapsed_text += f" [bold]({_fmt_elapsed(step_elapsed)})[/bold]"

        token_text = ""
        if in_tok or out_tok:
            token_text = f"[dim]{_fmt_tokens(in_tok)}[/dim]/[bold]{_fmt_tokens(out_tok)}[/bold]"

        step_text = current
        if error:
            step_text = f"[red]{error[:60]}[/red]"

        table.add_row(
            Text(icon, style=style),
            name,
            Text(state, style=style),
            progress_bar,
            elapsed_text,
            token_text,
            step_text,
        )

    return table


async def _snapshot_progress(
    progress: dict[str, dict], lock: asyncio.Lock
) -> dict[str, dict]:
    async with lock:
        return {name: dict(info) for name, info in progress.items()}


async def _emit_progress(
    progress: dict[str, dict], lock: asyncio.Lock, max_parallel: int
) -> None:
    snapshot = await _snapshot_progress(progress, lock)
    console.print(_build_progress_table(snapshot, max_parallel))


def _executor_prompt(
    problem: str,
    repo: dict,
    step: RepoStep,
    step_index: int,
    total_steps: int,
    previous_summary: str | None,
) -> str:
    prev = previous_summary or "None"
    checks = repo.get("checks") or []
    checks_block = ""
    if checks:
        checks_list = "\n".join(f"  - {c}" for c in checks)
        checks_block = (
            f"\nVerification commands (will be run automatically after this step):\n"
            f"{checks_list}\n"
            f"You MUST ensure these commands pass before considering the step complete.\n"
            f"Run the verification commands yourself using language-specific tools "
            f"and fix any failures before finishing.\n"
        )
    return (
        f"Working repo: {repo['name']} (path: repos/{repo['name']}).\n"
        f"Overall goal:\n{problem}\n\n"
        f"Step {step_index + 1} of {total_steps}: {step.name}\n"
        f"Description: {step.description}\n\n"
        f"Expected outputs:\n- "
        + "\n- ".join(step.expected_outputs)
        + "\n\n"
        f"Success criteria:\n- "
        + "\n- ".join(step.success_criteria)
        + "\n\n"
        f"Previous step summary:\n{prev}\n"
        f"{checks_block}\n"
        "Use git tools with repo_path='repos/{repo_name}'.\n"
        "Use language-specific tools to validate your changes.\n"
        "Report the changes you made and the git status/diff summary."
    ).replace("{repo_name}", repo["name"])


def _fix_prompt(
    repo: dict,
    step: RepoStep,
    check_output: str,
    attempt: int,
    max_attempts: int,
) -> str:
    return (
        f"The verification checks FAILED after completing step '{step.name}' "
        f"in repo {repo['name']} (attempt {attempt}/{max_attempts}).\n\n"
        f"Failure output:\n{check_output}\n\n"
        f"Fix the failing tests. Do NOT move on to other work -- focus entirely "
        f"on making the checks pass. Run the tests again after your fix to confirm."
    )


async def _run_checks(repo: dict, workspace: Path) -> list[dict]:
    results = []
    checks = repo.get("checks") or []
    if not checks:
        return results

    log_dir = workspace / "checks"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{repo['name']}.log"

    with open(log_path, "w", encoding="utf-8") as log:
        for command in checks:
            args = shlex.split(command)
            code, stdout, stderr = _run_command(args, cwd=repo["path"])
            log.write(f"$ {command}\n")
            log.write(stdout)
            if stderr:
                log.write("\nSTDERR:\n")
                log.write(stderr)
            log.write("\n\n")
            results.append({
                "command": command,
                "exit_code": code,
                "stdout": stdout,
                "stderr": stderr,
                "log": str(log_path),
            })
    return results


def _checks_passed(check_results: list[dict]) -> bool:
    return all(c.get("exit_code", 1) == 0 for c in check_results)


def _format_check_failures(check_results: list[dict], max_output: int = 2000) -> str:
    """Format failed check output for inclusion in a retry prompt."""
    lines = []
    for cr in check_results:
        if cr.get("exit_code", 1) != 0:
            lines.append(f"FAILED: {cr['command']} (exit code {cr['exit_code']})")
            out = (cr.get("stdout") or "").strip()
            err = (cr.get("stderr") or "").strip()
            combined = f"{out}\n{err}".strip()
            if len(combined) > max_output:
                combined = combined[:max_output] + "\n... (truncated)"
            lines.append(combined)
    return "\n\n".join(lines)


async def _accumulate_tokens(
    agent, progress_state: dict, repo_name: str, lock: asyncio.Lock
) -> dict[str, int]:
    """Extract tokens from agent telemetry and add to progress state.

    Clears any live-preview counters set by the heartbeat so they aren't
    double-counted in the progress table.
    """
    step_tokens = _extract_agent_tokens(agent)
    async with lock:
        info = progress_state[repo_name]
        info["input_tokens"] = info.get("input_tokens", 0) + step_tokens["input_tokens"]
        info["output_tokens"] = info.get("output_tokens", 0) + step_tokens["output_tokens"]
        info["total_tokens"] = info.get("total_tokens", 0) + step_tokens["total_tokens"]
        # Clear live counters — the real totals are now in the main fields
        info.pop("_live_input", None)
        info.pop("_live_output", None)
        info.pop("_live_total", None)
    return step_tokens


async def _ainvoke_with_heartbeat(
    agent,
    prompt: str,
    recursion_limit: int,
    repo_name: str,
    step_name: str,
    timeout_sec: int = 0,
    heartbeat_sec: int = 60,
    progress_state: dict[str, dict] | None = None,
    progress_lock: asyncio.Lock | None = None,
) -> dict:
    """Run agent.ainvoke with periodic heartbeat logs and optional timeout.

    Heartbeat logs print every *heartbeat_sec* seconds so the user can see
    the agent is still alive even when no progress-table fields change.
    If *timeout_sec* > 0, the call is cancelled after that many seconds.
    Token counts in *progress_state* are updated live during heartbeats.
    """

    async def _heartbeat(stop: asyncio.Event):
        elapsed = 0
        while not stop.is_set():
            await asyncio.sleep(heartbeat_sec)
            if stop.is_set():
                break
            elapsed += heartbeat_sec
            # Read live token snapshot and update progress for display
            if progress_state is not None and progress_lock is not None:
                live_tokens = _extract_agent_tokens(agent)
                async with progress_lock:
                    info = progress_state.get(repo_name, {})
                    # Store live step tokens separately so _accumulate_tokens
                    # can do the final authoritative add without double-counting
                    info["_live_input"] = live_tokens["input_tokens"]
                    info["_live_output"] = live_tokens["output_tokens"]
                    info["_live_total"] = live_tokens["total_tokens"]
            tok_str = ""
            if progress_state and repo_name in progress_state:
                info = progress_state[repo_name]
                total = (
                    info.get("total_tokens", 0)
                    + info.get("_live_total", 0)
                )
                if total:
                    tok_str = f" [{_fmt_tokens(total)} tokens]"
            console.log(
                f"[dim]{repo_name}[/dim] step [bold]{step_name}[/bold] "
                f"still running ({_fmt_elapsed(elapsed)}){tok_str}"
            )

    stop = asyncio.Event()
    hb_task = asyncio.create_task(_heartbeat(stop))

    try:
        coro = agent.ainvoke(prompt, config={"recursion_limit": recursion_limit})
        if timeout_sec > 0:
            result = await asyncio.wait_for(coro, timeout=timeout_sec)
        else:
            result = await coro
        return result
    except asyncio.TimeoutError:
        console.log(
            f"[bold red]{repo_name}[/bold red] step [bold]{step_name}[/bold] "
            f"timed out after {_fmt_elapsed(timeout_sec)}"
        )
        raise
    finally:
        stop.set()
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass


async def _wait_for_repos(
    deps: list[str],
    repo_done_events: dict[str, asyncio.Event],
    repo_name: str,
    progress_state: dict[str, dict],
    progress_lock: asyncio.Lock,
    max_parallel: int,
) -> None:
    """Block until all repos in *deps* have finished their steps."""
    # Filter out self-dependencies — a repo's own steps are already sequential
    pending = [
        d for d in deps
        if d != repo_name and d in repo_done_events and not repo_done_events[d].is_set()
    ]
    if not pending:
        return

    async with progress_lock:
        info = progress_state[repo_name]
        info["state"] = "blocked"
        info["current"] = f"waiting on {', '.join(pending)}"
        info["updated"] = time.time()
    await _emit_progress(progress_state, progress_lock, max_parallel)

    await asyncio.gather(*(repo_done_events[d].wait() for d in pending))

    async with progress_lock:
        info = progress_state[repo_name]
        info["state"] = "running"
        info["updated"] = time.time()
    await _emit_progress(progress_state, progress_lock, max_parallel)


async def _run_repo_steps(
    repo: dict,
    steps: list[RepoStep],
    problem: str,
    workspace: Path,
    llm,
    recursion_limit: int,
    resume: bool,
    progress_state: dict[str, dict],
    progress_lock: asyncio.Lock,
    max_parallel: int,
    resume_dir: Path | None,
    resume_files: dict[str, Path],
    max_check_retries: int = 2,
    repo_done_events: dict[str, asyncio.Event] | None = None,
    step_timeout_sec: int = 0,
) -> dict:
    agent = make_git_agent(llm=llm, language=repo.get("language", "generic"), workspace=workspace)
    progress_path = _progress_path(
        workspace, repo["name"], resume_dir, resume_files
    )
    resume_progress = _load_progress(progress_path) if resume else {}
    start_index = int(resume_progress.get("next_index", 0)) if resume else 0
    plan_hash = _hash_plan(steps)
    has_checks = bool(repo.get("checks"))

    if resume and resume_progress.get("plan_hash") != plan_hash:
        start_index = 0

    last_summary = resume_progress.get("last_summary") if resume else None
    step_outputs_dir = workspace / "step_outputs" / repo["name"]
    step_outputs_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()
    async with progress_lock:
        info = progress_state[repo["name"]]
        info.update({
            "state": "running",
            "step": start_index,
            "total": len(steps),
            "current": None,
            "started": now,
            "step_started": None,
            "updated": now,
        })
    await _emit_progress(progress_state, progress_lock, max_parallel)

    all_check_results: list[dict] = []

    for idx in range(start_index, len(steps)):
        step = steps[idx]

        # -- Wait for cross-repo dependencies --
        if step.depends_on_repos and repo_done_events:
            await _wait_for_repos(
                deps=step.depends_on_repos,
                repo_done_events=repo_done_events,
                repo_name=repo["name"],
                progress_state=progress_state,
                progress_lock=progress_lock,
                max_parallel=max_parallel,
            )

        step_start = time.time()
        async with progress_lock:
            info = progress_state[repo["name"]]
            info.update({
                "state": "running",
                "step": idx + 1,
                "current": step.name,
                "step_started": step_start,
                "updated": step_start,
            })
        await _emit_progress(progress_state, progress_lock, max_parallel)
        prompt = _executor_prompt(
            problem=problem,
            repo=repo,
            step=step,
            step_index=idx,
            total_steps=len(steps),
            previous_summary=last_summary,
        )
        result = await _ainvoke_with_heartbeat(
            agent=agent,
            prompt=prompt,
            recursion_limit=recursion_limit,
            repo_name=repo["name"],
            step_name=step.name,
            timeout_sec=step_timeout_sec,
            progress_state=progress_state,
            progress_lock=progress_lock,
        )
        step_tokens = await _accumulate_tokens(
            agent, progress_state, repo["name"], progress_lock
        )

        summary = result["messages"][-1].text
        last_summary = summary

        # -- Run checks after each step --
        if has_checks:
            check_results = await _run_checks(repo, workspace)
            passed = _checks_passed(check_results)

            for cr in check_results:
                status = (
                    "[green]pass[/green]" if cr["exit_code"] == 0
                    else "[red]FAIL[/red]"
                )
                console.log(
                    f"[bold]{repo['name']}[/bold] step {idx + 1} check {status}: {cr['command']}"
                )

            # Retry loop: give the agent a chance to fix failures
            attempt = 0
            while not passed and attempt < max_check_retries:
                attempt += 1
                failure_output = _format_check_failures(check_results)
                console.log(
                    f"[bold yellow]{repo['name']}[/bold yellow] step {idx + 1} "
                    f"checks failed, retry {attempt}/{max_check_retries}"
                )
                async with progress_lock:
                    info = progress_state[repo["name"]]
                    info["current"] = f"{step.name} (fix {attempt}/{max_check_retries})"
                    info["updated"] = time.time()
                await _emit_progress(progress_state, progress_lock, max_parallel)

                fix = _fix_prompt(
                    repo=repo,
                    step=step,
                    check_output=failure_output,
                    attempt=attempt,
                    max_attempts=max_check_retries,
                )
                result = await _ainvoke_with_heartbeat(
                    agent=agent,
                    prompt=fix,
                    recursion_limit=recursion_limit,
                    repo_name=repo["name"],
                    step_name=f"{step.name} (fix {attempt})",
                    timeout_sec=step_timeout_sec,
                    progress_state=progress_state,
                    progress_lock=progress_lock,
                )
                await _accumulate_tokens(
                    agent, progress_state, repo["name"], progress_lock
                )
                summary = result["messages"][-1].text
                last_summary = summary

                check_results = await _run_checks(repo, workspace)
                passed = _checks_passed(check_results)
                for cr in check_results:
                    status = (
                        "[green]pass[/green]" if cr["exit_code"] == 0
                        else "[red]FAIL[/red]"
                    )
                    console.log(
                        f"[bold]{repo['name']}[/bold] step {idx + 1} retry {attempt} "
                        f"check {status}: {cr['command']}"
                    )

            all_check_results = check_results  # keep latest results

            if not passed:
                console.log(
                    f"[bold red]{repo['name']}[/bold red] step {idx + 1} "
                    f"checks still failing after {max_check_retries} retries, "
                    f"continuing to next step"
                )

        (step_outputs_dir / f"step_{idx + 1}.md").write_text(
            summary, encoding="utf-8"
        )
        _save_progress(
            progress_path,
            {
                "next_index": idx + 1,
                "plan_hash": plan_hash,
                "last_summary": summary,
            },
        )
        step_tok_str = _fmt_tokens(step_tokens["total_tokens"]) if step_tokens["total_tokens"] else ""
        console.log(
            f"[bold]{repo['name']}[/bold] step {idx + 1}/{len(steps)} "
            f"[green]complete[/green]: {step.name}"
            + (f"  [dim]({step_tok_str} tokens)[/dim]" if step_tok_str else "")
        )

    async with progress_lock:
        info = progress_state[repo["name"]]
        info.update({
            "state": "done",
            "step": len(steps),
            "current": None,
            "updated": time.time(),
        })
    # Signal that this repo is done so dependent repos can proceed
    if repo_done_events and repo["name"] in repo_done_events:
        repo_done_events[repo["name"]].set()
    await _emit_progress(progress_state, progress_lock, max_parallel)

    # Final check run (catches anything the last step may have missed)
    if has_checks:
        all_check_results = await _run_checks(repo, workspace)
        for cr in all_check_results:
            status = (
                "[green]pass[/green]" if cr["exit_code"] == 0
                else "[red]FAIL[/red]"
            )
            console.log(
                f"[bold]{repo['name']}[/bold] final check {status}: {cr['command']}"
            )
    else:
        console.log(f"[bold]{repo['name']}[/bold] [dim]no checks configured[/dim]")

    async with progress_lock:
        info = progress_state[repo["name"]]
        repo_tokens = {
            "input_tokens": info.get("input_tokens", 0),
            "output_tokens": info.get("output_tokens", 0),
            "total_tokens": info.get("total_tokens", 0),
        }

    return {
        "repo": repo["name"],
        "steps": len(steps),
        "checks": all_check_results,
        "tokens": repo_tokens,
    }


async def _run_parallel(
    repo_steps: dict[str, list[RepoStep]],
    repos: list[dict],
    problem: str,
    workspace: Path,
    models_cfg: dict,
    executor_model: str,
    recursion_limit: int,
    max_parallel: int,
    resume: bool,
    status_interval_sec: int,
    resume_dir: Path | None,
    resume_files: dict[str, Path],
    max_check_retries: int = 2,
    step_timeout_sec: int = 0,
) -> list[dict]:
    sem = asyncio.Semaphore(max(1, max_parallel))
    repo_lookup = {repo["name"]: repo for repo in repos}
    progress_state = _init_progress(repo_steps)
    progress_lock = asyncio.Lock()

    # Create an event per repo so dependents can wait for completion
    repo_done_events: dict[str, asyncio.Event] = {
        name: asyncio.Event() for name in repo_steps
    }

    # Log dependency info
    dep_pairs: list[str] = []
    for name, steps in repo_steps.items():
        for step in steps:
            for dep in step.depends_on_repos:
                dep_pairs.append(f"{name} -> {dep}")
    if dep_pairs:
        console.print(
            Panel(
                "\n".join(sorted(set(dep_pairs))),
                title="[bold]Repo dependencies[/bold]",
                border_style="cyan",
                expand=False,
            )
        )

    async def status_loop(stop_event: asyncio.Event):
        if status_interval_sec <= 0:
            return
        while not stop_event.is_set():
            await asyncio.sleep(status_interval_sec)
            await _emit_progress(progress_state, progress_lock, max_parallel)

    async def run_one(repo_name: str, steps: list[RepoStep]) -> dict:
        async with sem:
            repo = repo_lookup[repo_name]
            llm = setup_llm(
                model_choice=executor_model,
                models_cfg=models_cfg,
                agent_name="executor",
            )
            try:
                return await _run_repo_steps(
                    repo=repo,
                    steps=steps,
                    problem=problem,
                    workspace=workspace,
                    llm=llm,
                    recursion_limit=recursion_limit,
                    resume=resume,
                    progress_state=progress_state,
                    progress_lock=progress_lock,
                    max_parallel=max_parallel,
                    resume_dir=resume_dir,
                    resume_files=resume_files,
                    max_check_retries=max_check_retries,
                    repo_done_events=repo_done_events,
                    step_timeout_sec=step_timeout_sec,
                )
            except Exception as exc:
                async with progress_lock:
                    info = progress_state[repo_name]
                    info.update({
                        "state": "failed",
                        "error": str(exc),
                        "updated": time.time(),
                    })
                # Signal done even on failure so dependents don't deadlock
                if repo_name in repo_done_events:
                    repo_done_events[repo_name].set()
                await _emit_progress(progress_state, progress_lock, max_parallel)
                raise

    await _emit_progress(progress_state, progress_lock, max_parallel)
    stop_event = asyncio.Event()
    reporter = asyncio.create_task(status_loop(stop_event))
    try:
        tasks = [run_one(name, steps) for name, steps in repo_steps.items()]
        return await asyncio.gather(*tasks)
    finally:
        stop_event.set()
        await reporter


def main():
    parser = argparse.ArgumentParser(
        description="Multi-repo plan/execute runner"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config for planning and multi-repo execution",
    )
    parser.add_argument(
        "--workspace",
        required=False,
        help="Workspace directory for artifacts and logs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress files in the workspace",
    )
    parser.add_argument(
        "--resume-from",
        action="append",
        dest="resume_from",
        help=(
            "Path to a repo progress file (repeatable) or a directory containing"
            " progress files"
        ),
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    config_dir = Path(args.config).parent.resolve()

    project = getattr(cfg, "project", "multi_repo_run")
    problem = getattr(cfg, "problem", "").strip()
    if not problem:
        console.print("[bold red]Config must include a non-empty 'problem' field.[/bold red]")
        sys.exit(2)

    raw_repos = getattr(cfg, "repos", None)
    if not raw_repos:
        console.print("[bold red]Config must include a 'repos' list.[/bold red]")
        sys.exit(2)

    workspace = _resolve_workspace(args.workspace, project)
    repos = _resolve_repos(raw_repos, config_dir, workspace)

    # -- Workspace banner --
    repo_lines = "\n".join(
        f"  [bold]{r['name']}[/bold] ({r.get('language', 'generic')})"
        + (f" - {r['description']}" if r.get("description") else "")
        for r in repos
    )
    console.print(
        Panel(
            f"[bold bright_blue]{workspace}[/bold bright_blue]\n\n"
            f"[bold]Repos:[/bold]\n{repo_lines}",
            title="[bold green]MULTI-REPO WORKSPACE[/bold green]",
            border_style="bright_magenta",
            padding=(1, 2),
        )
    )

    with console.status("[bold green]Checking out repos...", spinner="point"):
        for repo in repos:
            _ensure_checkout(repo)
            _ensure_repo_symlink(workspace, repo)

    models_cfg = getattr(cfg, "models", {}) or {}
    default_model = (
        (models_cfg.get("default") or None)
        or (models_cfg.get("choices") or ["openai:gpt-5-mini"])[0]
    )
    planner_model = models_cfg.get("planner") or default_model
    executor_model = models_cfg.get("executor") or default_model
    console.print(
        Panel(
            f"[bold]Planner model:[/bold] [cyan]{planner_model}[/cyan]  "
            f"[bold]Executor model:[/bold] [cyan]{executor_model}[/cyan]",
            border_style="cyan",
            expand=False,
        )
    )

    # -- Validate models before starting real work --
    with console.status("[bold green]Validating models...", spinner="point"):
        planner_llm = setup_llm(
            model_choice=planner_model,
            models_cfg=models_cfg,
            agent_name="planner",
        )
        _validate_model(planner_llm, planner_model, "planner")
        console.log(f"[green]✓[/green] planner model [cyan]{planner_model}[/cyan]")

        if executor_model != planner_model:
            executor_test_llm = setup_llm(
                model_choice=executor_model,
                models_cfg=models_cfg,
                agent_name="executor",
            )
            _validate_model(executor_test_llm, executor_model, "executor")
            console.log(f"[green]✓[/green] executor model [cyan]{executor_model}[/cyan]")
        else:
            console.log(f"[green]✓[/green] executor model (same as planner)")

    planner_cfg = getattr(cfg, "planner", {}) or {}
    reflection_steps = int(planner_cfg.get("reflection_steps", 0))
    research_cfg = planner_cfg.get("research") or {}

    # -- Problem statement --
    console.print(
        Panel(
            Text.from_markup(f"[bold cyan]Problem:[/bold cyan]\n{problem}"),
            border_style="cyan",
        )
    )

    # -- Research phase --
    with console.status(
        "[bold green]Gathering research...", spinner="point"
    ):
        research = asyncio.run(
            _gather_research(
                llm=planner_llm,
                workspace=workspace,
                research_cfg=research_cfg,
                problem=problem,
                repos=repos,
            )
        )
    if research:
        console.print("[green]Research complete.[/green]")
    else:
        console.print("[dim]No research context gathered.[/dim]")

    # -- Planning phase --
    with console.status(
        f"[bold green]Planning across {len(repos)} repos "
        f"(reflection steps: {reflection_steps})...",
        spinner="point",
    ):
        plan = asyncio.run(
            _plan(
                llm=planner_llm,
                problem=problem,
                repos=repos,
                research=research,
                reflection_steps=reflection_steps,
            )
        )

    _validate_plan_repos(plan, repos)
    _write_plan(workspace, plan)
    _render_repo_plan(plan)
    repo_steps = _group_steps_by_repo(plan)

    missing = sorted({repo["name"] for repo in repos} - set(repo_steps))
    if missing:
        console.print(
            Panel(
                "[bold yellow]Plan includes no steps for:[/bold yellow] "
                + ", ".join(missing),
                border_style="yellow",
                expand=False,
            )
        )

    exec_cfg = getattr(cfg, "execution", {}) or {}
    max_parallel = int(exec_cfg.get("max_parallel", len(repo_steps)))
    recursion_limit = int(exec_cfg.get("recursion_limit", 2000))
    resume = bool(exec_cfg.get("resume", False))
    status_interval_sec = int(exec_cfg.get("status_interval_sec", 5))
    max_check_retries = int(exec_cfg.get("max_check_retries", 2))
    step_timeout_sec = int(exec_cfg.get("step_timeout_sec", 0))  # 0 = no limit
    resume_dir = None
    resume_files: dict[str, Path] = {}

    if args.resume_from:
        resume = True
        resume_dir, resume_files = _parse_resume_overrides(
            args.resume_from, config_dir
        )

    if args.resume:
        resume = True

    unknown_resume = sorted(
        set(resume_files) - {repo["name"] for repo in repos}
    )
    if unknown_resume:
        raise RuntimeError(
            "Resume checkpoints do not match repos: "
            + ", ".join(unknown_resume)
        )

    # -- Execution banner --
    console.rule("[bold cyan]Execution")
    if resume:
        console.print(
            Panel(
                f"[bold yellow]Resuming[/bold yellow] from saved progress"
                + (f" (dir: {resume_dir})" if resume_dir else ""),
                border_style="yellow",
                expand=False,
            )
        )
    timeout_str = f"{step_timeout_sec}s" if step_timeout_sec else "none"
    console.print(
        f"[bold]Parallel workers:[/bold] {max_parallel}  "
        f"[bold]Status interval:[/bold] {status_interval_sec}s  "
        f"[bold]Check retries:[/bold] {max_check_retries}  "
        f"[bold]Step timeout:[/bold] {timeout_str}  "
        f"[bold]Repos:[/bold] {len(repo_steps)}"
    )

    results = asyncio.run(
        _run_parallel(
            repo_steps=repo_steps,
            repos=repos,
            problem=problem,
            workspace=workspace,
            models_cfg=models_cfg,
            executor_model=executor_model,
            recursion_limit=recursion_limit,
            max_parallel=max_parallel,
            resume=resume,
            status_interval_sec=status_interval_sec,
            resume_dir=resume_dir,
            resume_files=resume_files,
            max_check_retries=max_check_retries,
            step_timeout_sec=step_timeout_sec,
        )
    )

    summary_path = workspace / "run_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    # -- Final summary --
    console.rule("[bold cyan]Run complete")
    result_lines = []
    grand_in = 0
    grand_out = 0
    grand_total = 0
    for r in results:
        name = r.get("repo", "?")
        steps = r.get("steps", 0)
        checks = r.get("checks") or []
        tokens = r.get("tokens") or {}
        r_in = tokens.get("input_tokens", 0)
        r_out = tokens.get("output_tokens", 0)
        r_total = tokens.get("total_tokens", 0)
        grand_in += r_in
        grand_out += r_out
        grand_total += r_total
        check_ok = all(c.get("exit_code", 1) == 0 for c in checks)
        check_text = (
            f"[green]passed[/green]" if check_ok and checks
            else f"[red]failures[/red]" if checks
            else "[dim]none[/dim]"
        )
        tok_text = f"tokens: {_fmt_tokens(r_in)} in / {_fmt_tokens(r_out)} out" if r_total else "tokens: [dim]0[/dim]"
        result_lines.append(
            f"  [bold]{name}[/bold]: {steps} steps, checks: {check_text}, {tok_text}"
        )
    result_lines.append("")
    result_lines.append(
        f"  [bold]Total tokens:[/bold] [yellow]{_fmt_tokens(grand_total)}[/yellow] "
        f"({_fmt_tokens(grand_in)} in / {_fmt_tokens(grand_out)} out)"
    )
    console.print(
        Panel(
            "\n".join(result_lines)
            + f"\n\n[dim]Summary written to {summary_path}[/dim]",
            title="[bold green]RESULTS[/bold green]",
            border_style="bright_magenta",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    main()
