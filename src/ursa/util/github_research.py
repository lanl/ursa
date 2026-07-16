"""Fetch recent issues & PRs from GitHub repos for planning context.

Uses the ``gh`` CLI (https://cli.github.com/) which handles authentication
transparently.  Falls back gracefully when ``gh`` is not installed or when a
repo URL does not point at GitHub.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any

_GH_URL_RE = re.compile(
    r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"
)


def parse_github_owner_repo(url: str) -> tuple[str, str] | None:
    """Extract ``(owner, repo)`` from a GitHub clone URL.

    Supports both HTTPS and SSH URLs.  Returns ``None`` for non-GitHub URLs.
    """
    m = _GH_URL_RE.search(url or "")
    if m:
        return m.group("owner"), m.group("repo")
    return None


def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _gh_api(endpoint: str, timeout: int = 30) -> Any:
    """Call ``gh api`` and return parsed JSON."""
    result = subprocess.run(
        ["gh", "api", endpoint, "--paginate"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def _format_issue(item: dict) -> str:
    number = item.get("number", "?")
    title = item.get("title", "")
    state = item.get("state", "")
    labels = ", ".join(
        label.get("name", "") for label in (item.get("labels") or [])
    )
    created = (item.get("created_at") or "")[:10]
    body = (item.get("body") or "")[:300]
    parts = [f"  #{number} [{state}] {title}"]
    if labels:
        parts.append(f"    Labels: {labels}")
    if created:
        parts.append(f"    Created: {created}")
    if body:
        parts.append(f"    {body}")
    return "\n".join(parts)


def _format_pr(item: dict) -> str:
    number = item.get("number", "?")
    title = item.get("title", "")
    state = item.get("state", "")
    created = (item.get("created_at") or "")[:10]
    body = (item.get("body") or "")[:300]
    parts = [f"  #{number} [{state}] {title}"]
    if created:
        parts.append(f"    Created: {created}")
    if body:
        parts.append(f"    {body}")
    return "\n".join(parts)


def fetch_repo_context(
    owner: str,
    repo: str,
    *,
    max_issues: int = 10,
    max_prs: int = 10,
    issue_state: str = "all",
    pr_state: str = "all",
) -> str:
    """Fetch recent issues and PRs for a single GitHub repo.

    Returns a formatted text block suitable for inclusion in a planner prompt.
    """
    sections: list[str] = []
    sections.append(f"## {owner}/{repo}")

    # Recent issues
    try:
        issues = _gh_api(
            f"/repos/{owner}/{repo}/issues?state={issue_state}"
            f"&per_page={max_issues}&sort=updated&direction=desc"
        )
        # gh api may return PRs mixed with issues; filter them out
        pure_issues = [i for i in issues if "pull_request" not in i][
            :max_issues
        ]
        if pure_issues:
            sections.append(f"### Recent issues ({len(pure_issues)})")
            sections.extend(_format_issue(issue) for issue in pure_issues)
        else:
            sections.append("### Recent issues: none")
    except Exception as exc:  # noqa: BLE001
        sections.append(f"### Issues: could not fetch ({exc})")

    # Recent PRs
    try:
        prs = _gh_api(
            f"/repos/{owner}/{repo}/pulls?state={pr_state}"
            f"&per_page={max_prs}&sort=updated&direction=desc"
        )
        if prs:
            sections.append(f"### Recent pull requests ({len(prs[:max_prs])})")
            sections.extend(_format_pr(pr) for pr in prs[:max_prs])
        else:
            sections.append("### Recent pull requests: none")
    except Exception as exc:  # noqa: BLE001
        sections.append(f"### PRs: could not fetch ({exc})")

    return "\n".join(sections)


def gather_github_context(
    repos: list[dict],
    *,
    max_issues: int = 10,
    max_prs: int = 10,
) -> str | None:
    """Gather GitHub context for all repos that have GitHub URLs.

    Parameters
    ----------
    repos:
        List of repo config dicts (each must have at least ``url`` and ``name``).
    max_issues:
        Maximum recent issues to fetch per repo.
    max_prs:
        Maximum recent PRs to fetch per repo.

    Returns
    -------
    Formatted text block with issues/PRs across repos, or ``None`` if nothing
    was fetched (e.g. no GitHub URLs, ``gh`` not installed).
    """
    if not _gh_available():
        return None

    blocks: list[str] = []
    for repo in repos:
        parsed = parse_github_owner_repo(repo.get("url", ""))
        if not parsed:
            continue
        owner, name = parsed
        try:
            block = fetch_repo_context(
                owner, name, max_issues=max_issues, max_prs=max_prs
            )
            blocks.append(block)
        except Exception:  # noqa: BLE001, S112
            # Network issue, auth issue, etc. -- skip silently
            continue

    if not blocks:
        return None
    return "\n\n".join(blocks)
