"""Dependency-free persistent Chainlit data layer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chainlit.data.base import BaseDataLayer
from chainlit.types import PageInfo, PaginatedResponse
from chainlit.user import PersistedUser


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JsonDataLayer(BaseDataLayer):
    """Persist Chainlit threads and messages in a workspace-local JSON file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {"threads": {}, "users": {}}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2, default=str) + "\n")

    async def get_user(self, identifier: str):
        user = self._load()["users"].get(identifier)
        return PersistedUser(**user) if user else None

    async def create_user(self, user):
        data = self._load()
        persisted = {
            "id": user.identifier,
            "identifier": user.identifier,
            "createdAt": _now(),
            "metadata": user.metadata,
        }
        data["users"][user.identifier] = persisted
        self._save(data)
        return PersistedUser(**persisted)

    async def delete_feedback(self, feedback_id: str) -> bool:
        return False

    async def upsert_feedback(self, feedback) -> str:
        return feedback.id

    async def create_element(self, element) -> None:
        return None

    async def get_element(self, thread_id: str, element_id: str):
        thread = await self.get_thread(thread_id)
        if thread is None:
            return None
        return next(
            (
                element
                for element in thread.get("elements") or []
                if element["id"] == element_id
            ),
            None,
        )

    async def delete_element(
        self, element_id: str, thread_id: str | None = None
    ):
        if thread_id is None:
            return
        data = self._load()
        thread = data["threads"].get(thread_id)
        if thread is not None:
            thread["elements"] = [
                element
                for element in thread.get("elements", [])
                if element.get("id") != element_id
            ]
            self._save(data)
        return

    async def create_step(self, step_dict: dict[str, Any]) -> None:
        data = self._load()
        thread_id = step_dict.get("threadId")
        if not thread_id:
            return
        thread = data["threads"].setdefault(
            thread_id,
            {
                "id": thread_id,
                "createdAt": _now(),
                "name": None,
                "userId": "local",
                "userIdentifier": "local",
                "tags": [],
                "metadata": {},
                "steps": [],
                "elements": [],
            },
        )
        steps = thread["steps"]
        step_id = step_dict.get("id")
        for index, step in enumerate(steps):
            if step.get("id") == step_id:
                steps[index] = step_dict
                break
        else:
            steps.append(step_dict)
        if thread["name"] is None and step_dict.get("type") == "user_message":
            prompt = str(
                step_dict.get("output") or step_dict.get("input") or ""
            )
            thread["name"] = prompt[:80] or "New conversation"
        self._save(data)

    async def update_step(self, step_dict: dict[str, Any]) -> None:
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str):
        data = self._load()
        for thread in data["threads"].values():
            thread["steps"] = [
                step for step in thread["steps"] if step.get("id") != step_id
            ]
        self._save(data)

    async def get_thread_author(self, thread_id: str) -> str:
        thread = await self.get_thread(thread_id)
        return "" if thread is None else thread.get("userIdentifier") or ""

    async def delete_thread(self, thread_id: str):
        data = self._load()
        data["threads"].pop(thread_id, None)
        self._save(data)

    async def list_threads(self, pagination, filters):
        threads = list(self._load()["threads"].values())
        if filters.search:
            query = filters.search.lower()
            threads = [
                thread
                for thread in threads
                if query in (thread.get("name") or "").lower()
            ]
        threads.sort(key=lambda thread: thread["createdAt"], reverse=True)
        page = threads[: pagination.first]
        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=len(threads) > len(page),
                startCursor=page[0]["id"] if page else None,
                endCursor=page[-1]["id"] if page else None,
            ),
            data=page,
        )

    async def get_thread(self, thread_id: str):
        return self._load()["threads"].get(thread_id)

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ):
        data = self._load()
        thread = data["threads"].setdefault(
            thread_id,
            {
                "id": thread_id,
                "createdAt": _now(),
                "name": name,
                "userId": user_id,
                "userIdentifier": None,
                "tags": tags or [],
                "metadata": metadata or {},
                "steps": [],
                "elements": [],
            },
        )
        if name is not None:
            thread["name"] = name
        if user_id is not None:
            thread["userId"] = user_id
        if metadata is not None:
            thread["metadata"] = metadata
        if tags is not None:
            thread["tags"] = tags
        self._save(data)

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        return None

    async def get_favorite_steps(self, user_id: str) -> list[dict[str, Any]]:
        return []
