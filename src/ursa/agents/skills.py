from ursa.agents.base import AgentWithTools
from ursa.skills import (
    annotate_skill_requests,
    build_skill_tool,
    get_skill_catalog,
)


class AgentWithSkills(AgentWithTools):
    """Add skill tools and explicit skill requests to a tool-using agent."""

    def __init__(
        self,
        *args,
        use_skills: bool = True,
        workspace=None,
        tools=None,
        **kwargs,
    ):
        self.use_skills = use_skills
        self.skill_catalog = (
            get_skill_catalog(workspace) if use_skills else None
        )
        tools = (
            dict(tools)
            if isinstance(tools, dict)
            else {tool.name: tool for tool in tools or []}
        )
        if self.skill_catalog is not None:
            skill_tool = build_skill_tool(skills=self.skill_catalog.load())
            if skill_tool is not None:
                tools[skill_tool.name] = skill_tool
        super().__init__(*args, workspace=workspace, tools=tools, **kwargs)

    def format_query(self, prompt: str, state=None):
        if self.skill_catalog is not None:
            prompt = annotate_skill_requests(
                prompt, skills=self.skill_catalog.load()
            )
        return super().format_query(prompt, state=state)
