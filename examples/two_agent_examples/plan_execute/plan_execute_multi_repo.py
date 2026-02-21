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

from ursa.agents import WebSearchAgent, make_git_agent
from ursa.prompt_library.planning_prompts import reflection_prompt


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
            ),
            noun=(
                "cats",
                "dogs",
                "apex_predators",
                "birds",
                "fish",
                "fruit",
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


def _resolve_repos(raw_repos: list[dict], config_dir: Path) -> list[dict]:
    repos = []
    for raw in raw_repos:
        if not isinstance(raw, dict):
            raise ValueError("Each repo entry must be a mapping/object.")

        name = raw.get("name")
        path_value = raw.get("path")
        if not name or not path_value:
            raise ValueError("Each repo requires 'name' and 'path'.")

        path = Path(path_value)
        if not path.is_absolute():
            path = (config_dir / path).resolve()

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

    if branch:
        code, stdout, stderr = _run_command(
            ["git", "-C", str(path), "checkout", branch]
        )
        if code != 0:
            raise RuntimeError(
                f"git checkout failed for {repo['name']}\n{stdout}\n{stderr}"
            )


def _ensure_repo_symlink(workspace: Path, repo: dict) -> Path:
    repos_dir = workspace / "repos"
    repos_dir.mkdir(exist_ok=True)
    target = repos_dir / repo["name"]
    source = repo["path"]

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
    )


async def _gather_research(
    llm,
    workspace: Path,
    research_cfg: dict | None,
    problem: str,
) -> str | None:
    if not research_cfg:
        return None

    queries = research_cfg.get("queries") or []
    if not queries:
        return None

    try:
        agent = WebSearchAgent(
            llm=llm,
            workspace=workspace,
            summarize=True,
            max_results=int(research_cfg.get("max_results", 3)),
        )
    except Exception as exc:
        print(f"[warn] WebSearchAgent unavailable: {exc}")
        return None

    summaries: list[str] = []
    for query in queries:
        context = f"{problem}\n\nResearch focus: {query}"
        result = await agent.ainvoke({"query": query, "context": context})
        summary = result.get("final_summary") or ""
        summaries.append(f"Query: {query}\n{summary}")

    return "\n\n".join(summaries)


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


def _init_progress(repo_steps: dict[str, list[RepoStep]]) -> dict[str, dict]:
    return {
        name: {
            "state": "queued",
            "step": 0,
            "total": len(steps),
            "current": None,
            "error": None,
            "updated": time.time(),
        }
        for name, steps in repo_steps.items()
    }


def _summarize_progress(snapshot: dict[str, dict], max_parallel: int) -> str:
    counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
    for info in snapshot.values():
        state = info.get("state", "queued")
        counts[state] = counts.get(state, 0) + 1

    header = (
        "[status] "
        f"active {counts.get('running', 0)}/{max_parallel} | "
        f"queued {counts.get('queued', 0)} | "
        f"done {counts.get('done', 0)} | "
        f"failed {counts.get('failed', 0)}"
    )
    lines = [header]
    for name in sorted(snapshot):
        info = snapshot[name]
        state = info.get("state", "queued")
        step = info.get("step", 0)
        total = info.get("total", 0)
        current = info.get("current")
        detail = ""
        if total:
            detail = f" step {step}/{total}"
        if current:
            detail += f" - {current}"
        if info.get("error"):
            detail += " (error)"
        lines.append(f"  - {name}: {state}{detail}")
    return "\n".join(lines)


async def _snapshot_progress(
    progress: dict[str, dict], lock: asyncio.Lock
) -> dict[str, dict]:
    async with lock:
        return {name: dict(info) for name, info in progress.items()}


async def _emit_progress(
    progress: dict[str, dict], lock: asyncio.Lock, max_parallel: int
) -> None:
    snapshot = await _snapshot_progress(progress, lock)
    print(_summarize_progress(snapshot, max_parallel))


def _executor_prompt(
    problem: str,
    repo: dict,
    step: RepoStep,
    step_index: int,
    total_steps: int,
    previous_summary: str | None,
) -> str:
    prev = previous_summary or "None"
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
        f"Previous step summary:\n{prev}\n\n"
        "Use git tools with repo_path='repos/{repo_name}'.\n"
        "Use language-specific tools to validate your changes.\n"
        "Report the changes you made and the git status/diff summary."
    ).replace("{repo_name}", repo["name"])


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
                "log": str(log_path),
            })
    return results


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
) -> dict:
    agent = make_git_agent(llm=llm, language=repo.get("language", "generic"), workspace=workspace)
    progress_path = _progress_path(
        workspace, repo["name"], resume_dir, resume_files
    )
    resume_progress = _load_progress(progress_path) if resume else {}
    start_index = int(resume_progress.get("next_index", 0)) if resume else 0
    plan_hash = _hash_plan(steps)

    if resume and resume_progress.get("plan_hash") != plan_hash:
        start_index = 0

    last_summary = resume_progress.get("last_summary") if resume else None
    step_outputs_dir = workspace / "step_outputs" / repo["name"]
    step_outputs_dir.mkdir(parents=True, exist_ok=True)

    async with progress_lock:
        info = progress_state[repo["name"]]
        info.update({
            "state": "running",
            "step": start_index,
            "total": len(steps),
            "current": None,
            "updated": time.time(),
        })
    await _emit_progress(progress_state, progress_lock, max_parallel)
    for idx in range(start_index, len(steps)):
        step = steps[idx]
        async with progress_lock:
            info = progress_state[repo["name"]]
            info.update({
                "state": "running",
                "step": idx + 1,
                "current": step.name,
                "updated": time.time(),
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
        result = await agent.ainvoke(
            prompt,
            config={"recursion_limit": recursion_limit},
        )
        summary = result["messages"][-1].text
        last_summary = summary
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

    async with progress_lock:
        info = progress_state[repo["name"]]
        info.update({
            "state": "done",
            "step": len(steps),
            "current": None,
            "updated": time.time(),
        })
    await _emit_progress(progress_state, progress_lock, max_parallel)

    check_results = await _run_checks(repo, workspace)
    return {
        "repo": repo["name"],
        "steps": len(steps),
        "checks": check_results,
    }


async def _run_parallel(
    repo_steps: dict[str, list[RepoStep]],
    repos: list[dict],
    problem: str,
    workspace: Path,
    models_cfg: dict,
    model_choice: str,
    recursion_limit: int,
    max_parallel: int,
    resume: bool,
    status_interval_sec: int,
    resume_dir: Path | None,
    resume_files: dict[str, Path],
) -> list[dict]:
    sem = asyncio.Semaphore(max(1, max_parallel))
    repo_lookup = {repo["name"]: repo for repo in repos}
    progress_state = _init_progress(repo_steps)
    progress_lock = asyncio.Lock()

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
                model_choice=model_choice,
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
                )
            except Exception as exc:
                async with progress_lock:
                    info = progress_state[repo_name]
                    info.update({
                        "state": "failed",
                        "error": str(exc),
                        "updated": time.time(),
                    })
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
        print("Config must include a non-empty 'problem' field.")
        sys.exit(2)

    raw_repos = getattr(cfg, "repos", None)
    if not raw_repos:
        print("Config must include a 'repos' list.")
        sys.exit(2)

    repos = _resolve_repos(raw_repos, config_dir)
    workspace = _resolve_workspace(args.workspace, project)

    for repo in repos:
        _ensure_checkout(repo)
        _ensure_repo_symlink(workspace, repo)

    models_cfg = getattr(cfg, "models", {}) or {}
    model_choice = (
        (models_cfg.get("default") or None)
        or (models_cfg.get("choices") or ["openai:gpt-5-mini"])[0]
    )

    planner_cfg = getattr(cfg, "planner", {}) or {}
    reflection_steps = int(planner_cfg.get("reflection_steps", 0))
    research_cfg = planner_cfg.get("research") or {}

    planner_llm = setup_llm(
        model_choice=model_choice,
        models_cfg=models_cfg,
        agent_name="planner",
    )

    research = asyncio.run(
        _gather_research(
            llm=planner_llm,
            workspace=workspace,
            research_cfg=research_cfg,
            problem=problem,
        )
    )

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
    repo_steps = _group_steps_by_repo(plan)

    missing = sorted({repo["name"] for repo in repos} - set(repo_steps))
    if missing:
        print(
            "[warn] Plan includes no steps for: " + ", ".join(missing)
        )

    exec_cfg = getattr(cfg, "execution", {}) or {}
    max_parallel = int(exec_cfg.get("max_parallel", len(repo_steps)))
    recursion_limit = int(exec_cfg.get("recursion_limit", 2000))
    resume = bool(exec_cfg.get("resume", False))
    status_interval_sec = int(exec_cfg.get("status_interval_sec", 5))
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

    results = asyncio.run(
        _run_parallel(
            repo_steps=repo_steps,
            repos=repos,
            problem=problem,
            workspace=workspace,
            models_cfg=models_cfg,
            model_choice=model_choice,
            recursion_limit=recursion_limit,
            max_parallel=max_parallel,
            resume=resume,
            status_interval_sec=status_interval_sec,
            resume_dir=resume_dir,
            resume_files=resume_files,
        )
    )

    summary_path = workspace / "run_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Run complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
