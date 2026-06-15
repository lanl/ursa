from __future__ import annotations

import re
from datetime import UTC, datetime
from operator import add, or_
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, TypedDict, cast

from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from ursa.agents.base import AgentContext, AgentWithTools, BaseAgent
from ursa.prompt_library.hypothesizer_prompts import (
    competitor_prompt,
    critic_prompt,
    hypothesizer_prompt,
)
from ursa.tools import list_workspace_files, read_file
from ursa.tools.search_tools import (
    run_arxiv_search,
    run_osti_search,
    run_web_search,
)


class DeepReviewState(TypedDict, total=False):
    question: str
    question_search_query: str
    current_iteration: int
    max_iterations: int
    agent1_solution: Annotated[list[str], add]
    agent2_critiques: Annotated[list[str], add]
    agent3_perspectives: Annotated[list[str], add]
    solution: str
    summary_report: str
    visited_sites: Annotated[set[str], or_]
    messages: Annotated[list[BaseMessage], add_messages]
    active_phase: str


class DeepReviewAgent(AgentWithTools, BaseAgent[DeepReviewState]):
    """Iterative adversarial review agent with explicit autonomous tool use.

    The old Deep Review implementation performed hidden DuckDuckGo searches inside
    each phase. This implementation routes all external information access through
    configured LangChain tools:

    - workspace tools are always available (`list_workspace_files`, `read_file`),
      and operate on the user-selected workspace rather than the agent den;
    - web/search tools are only available when `use_web=True`;
    - persisted RAG tools are merged automatically by `AgentWithTools` when
      `rag_tools=...` is supplied to the base-agent constructor;
    - callers may add further tools via `extra_tools` or the normal
      `AgentWithTools.add_tool` / MCP mechanisms.
    """

    state_type = DeepReviewState

    def __init__(
        self,
        llm: BaseChatModel,
        max_iterations: int = 2,
        use_web: bool = False,
        extra_tools: list[BaseTool] | None = None,
        **kwargs,
    ):
        default_tools: list[BaseTool] = [list_workspace_files, read_file]
        if use_web:
            default_tools.extend([
                run_web_search,
                run_osti_search,
                run_arxiv_search,
            ])
        if extra_tools:
            default_tools.extend(extra_tools)

        super().__init__(llm=llm, tools=default_tools, **kwargs)
        self.hypothesizer_prompt = hypothesizer_prompt
        self.critic_prompt = critic_prompt
        self.competitor_prompt = competitor_prompt
        self.strllm = self.llm | StrOutputParser()
        self.max_iterations = max_iterations
        self.use_web = use_web
        self.extra_tools = extra_tools or []

    def _normalize_inputs(self, inputs) -> DeepReviewState:
        if isinstance(inputs, str):
            return DeepReviewState(
                question=inputs,
                question_search_query=self._normalize_search_query(inputs),
                max_iterations=self.max_iterations,
                current_iteration=0,
                agent1_solution=[],
                agent2_critiques=[],
                agent3_perspectives=[],
                visited_sites=set(),
                messages=[],
                active_phase="",
            )

        state = dict(cast(Mapping[str, Any], inputs))
        question = str(state.get("question") or state.get("query") or "")
        state.setdefault("question", question)
        state.setdefault(
            "question_search_query", self._normalize_search_query(question)
        )
        state.setdefault("max_iterations", self.max_iterations)
        state.setdefault("current_iteration", 0)
        state.setdefault("agent1_solution", [])
        state.setdefault("agent2_critiques", [])
        state.setdefault("agent3_perspectives", [])
        state.setdefault("visited_sites", set())
        state.setdefault("messages", [])
        state.setdefault("active_phase", "")
        return cast(DeepReviewState, state)

    def format_result(self, result: DeepReviewState) -> str:
        return result.get("solution", "Deep review failed to return a solution")

    def _normalize_search_query(self, query: str | None) -> str:
        words = " ".join(str(query or "").replace("\n", " ").split())
        words = words.strip(" \"'")
        return " ".join(words.split()[:8])

    @staticmethod
    def _message_text(message: Any) -> str:
        text = getattr(message, "text", None)
        if isinstance(text, str):
            return text
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _extract_urls(text: str) -> set[str]:
        return set(re.findall(r"https?://[^\s)\]}>,]+", text or ""))

    def _collect_visited_sites(self, state: DeepReviewState) -> set[str]:
        visited: set[str] = set(state.get("visited_sites") or set())
        for message in state.get("messages") or []:
            if message.__class__.__name__ == "ToolMessage":
                visited.update(self._extract_urls(self._message_text(message)))
        return visited

    def _base_phase_instructions(self, phase_label: str) -> str:
        tool_names = sorted(self.tools)
        web_instruction = (
            "Web/search tools are available because use_web=True. Use them only "
            "when they would materially improve the review."
            if self.use_web
            else "No web/search tools are available because use_web=False. Do not "
            "claim that you searched the web; rely on the prompt, workspace files, "
            "and any configured RAG tools."
        )
        return (
            f"You are the {phase_label} in an iterative deep-review process.\n"
            "You may autonomously call the configured tools when useful. Workspace "
            "documents live in the current workspace, not in the agent den. Use "
            "list_workspace_files to discover relevant files and read_file to read "
            "specific workspace documents. If RAG tools are available, use them for "
            "focused retrieval from configured document collections.\n"
            f"{web_instruction}\n"
            f"Available tools: {', '.join(tool_names) if tool_names else 'none'}.\n"
            "When you have enough information, answer directly without tool calls."
        )

    def _phase_prompt(self, phase: str, state: DeepReviewState) -> str:
        iteration = state["current_iteration"] + 1
        question = state["question"]
        if phase == "agent1":
            user_content = (
                f"Question: {question}\n"
                f"Deep-review iteration: {iteration}/{state['max_iterations']}\n"
            )
            if state.get("current_iteration", 0) > 0:
                user_content += (
                    f"\nPrevious solution: {state['agent1_solution'][-1]}"
                    f"\nCritique: {state['agent2_critiques'][-1]}"
                    "\nCompetitor/stakeholder perspective: "
                    f"{state['agent3_perspectives'][-1]}"
                    "\n\nExplicitly list how the new solution differs from the "
                    "previous solution, point by point, explaining what changes "
                    "were made in response to the critique and stakeholder "
                    "perspective. Then provide the updated solution."
                )
            else:
                user_content += (
                    "Generate an initial solution/hypothesis. Use available "
                    "workspace, RAG, or web tools only if they are relevant and "
                    "configured."
                )
            return user_content

        if phase == "agent2":
            solution = state["agent1_solution"][-1]
            return (
                f"Question: {question}\n"
                f"Proposed solution: {solution}\n"
                "Provide a detailed critique of this solution. Identify "
                "potential flaws, assumptions, missing evidence, and areas for "
                "improvement. Use available tools for targeted verification if "
                "useful and configured."
            )

        if phase == "agent3":
            solution = state["agent1_solution"][-1]
            critique = state["agent2_critiques"][-1]
            return (
                f"Question: {question}\n"
                f"Proposed solution: {solution}\n"
                f"Critique: {critique}\n"
                "Simulate how a competitor, government agency, stakeholder, or "
                "other adversarial party might respond to this solution. Use "
                "available tools for targeted context if useful and configured."
            )

        raise ValueError(f"Unknown deep-review phase: {phase}")

    def _role_system_prompt(self, phase: str) -> str:
        if phase == "agent1":
            role_prompt = self.hypothesizer_prompt
            label = "solution generator"
        elif phase == "agent2":
            role_prompt = self.critic_prompt
            label = "critic"
        elif phase == "agent3":
            role_prompt = self.competitor_prompt
            label = "competitor/stakeholder simulator"
        else:
            raise ValueError(f"Unknown deep-review phase: {phase}")
        return f"{role_prompt}\n\n{self._base_phase_instructions(label)}"

    def _role_output_key(self, phase: str) -> str:
        return {
            "agent1": "agent1_solution",
            "agent2": "agent2_critiques",
            "agent3": "agent3_perspectives",
        }[phase]

    def _run_role_phase(
        self,
        phase: str,
        state: DeepReviewState,
        config: RunnableConfig | None = None,
    ) -> DeepReviewState:
        events = self.events(config)
        iteration = state["current_iteration"] + 1
        events.emit(
            f"Deep-review {phase} pass {iteration}/{state['max_iterations']}",
            stage=phase,
            iteration=iteration,
            max_iterations=state["max_iterations"],
            tools=sorted(self.tools),
        )

        messages = list(state.get("messages") or [])
        new_phase_messages: list[BaseMessage] = []
        if state.get("active_phase") != phase:
            new_phase_messages = [
                SystemMessage(content=self._role_system_prompt(phase)),
                HumanMessage(content=self._phase_prompt(phase, state)),
            ]
            messages.extend(new_phase_messages)

        try:
            response = self.tool_llm.invoke(
                messages,
                self.build_config(tags=["deep_review", phase]),
            )
        except Exception as exc:  # noqa: BLE001
            response = AIMessage(content=f"Deep-review phase error: {exc}")
            events.emit(
                "Deep-review phase failed",
                stage=f"{phase}_error",
                error_type=type(exc).__name__,
                error=str(exc),
            )

        update: dict[str, Any] = {
            "messages": [*new_phase_messages, response],
            "active_phase": phase,
        }
        if getattr(response, "tool_calls", None):
            events.emit(
                "Deep-review phase requested tools",
                stage=f"{phase}_tool_request",
                tool_calls=[call.get("name") for call in response.tool_calls],
            )
            return cast(DeepReviewState, update)

        text = self._message_text(response)
        update[self._role_output_key(phase)] = [text]
        update["active_phase"] = ""
        update["visited_sites"] = self._collect_visited_sites(state)
        if phase == "agent1":
            update["question_search_query"] = self._normalize_search_query(
                state.get("question_search_query") or state.get("question")
            )
        events.emit(
            "Deep-review phase complete",
            stage=f"{phase}_result",
            preview=text,
        )
        return cast(DeepReviewState, update)

    def agent1_generate_solution(
        self,
        state: DeepReviewState,
        runtime: Runtime[AgentContext] | None = None,
        config: RunnableConfig | None = None,
    ) -> DeepReviewState:
        """Agent 1: autonomous solution generator."""
        return self._run_role_phase("agent1", state, config)

    def agent2_critique(
        self,
        state: DeepReviewState,
        runtime: Runtime[AgentContext] | None = None,
        config: RunnableConfig | None = None,
    ) -> DeepReviewState:
        """Agent 2: autonomous critic."""
        return self._run_role_phase("agent2", state, config)

    def agent3_competitor_perspective(
        self,
        state: DeepReviewState,
        runtime: Runtime[AgentContext] | None = None,
        config: RunnableConfig | None = None,
    ) -> DeepReviewState:
        """Agent 3: autonomous competitor/stakeholder simulator."""
        return self._run_role_phase("agent3", state, config)

    def increment_iteration(self, state: DeepReviewState) -> DeepReviewState:
        current_iteration = state["current_iteration"] + 1
        return {"current_iteration": current_iteration}

    def generate_solution(
        self,
        state: DeepReviewState,
        config: RunnableConfig | None = None,
    ) -> DeepReviewState:
        """Generate the overall, refined solution based on all iterations."""
        events = self.events(config)
        events.emit(
            "Synthesizing final hypothesis",
            stage="finalize",
            iteration=state["current_iteration"],
        )
        prompt = f"Original question: {state['question']}\n\n"
        prompt += "Evolution of solutions:\n"

        for i, (solution_text, critique_text, perspective_text) in enumerate(
            zip(
                state["agent1_solution"],
                state["agent2_critiques"],
                state["agent3_perspectives"],
            ),
            start=1,
        ):
            prompt += f"\nIteration {i}:\n"
            prompt += f"Solution: {solution_text}\n"
            prompt += f"Critique: {critique_text}\n"
            prompt += f"Competitor perspective: {perspective_text}\n"

        prompt += "\nBased on this iterative process, provide the overall, refined solution."
        solution = self.strllm.invoke(prompt)
        events.emit(
            "Final hypothesis ready",
            stage="finalize_result",
            preview=solution,
        )
        return {"solution": solution}

    def print_visited_sites(
        self,
        state: DeepReviewState,
        config: RunnableConfig | None = None,
    ) -> DeepReviewState:
        return state.copy()

    @staticmethod
    def _escape_latex(text: Any) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        escaped = str(text or "")
        for char, replacement in replacements.items():
            escaped = escaped.replace(char, replacement)
        return escaped

    @staticmethod
    def _strip_latex_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```latex"):
            stripped = stripped.removeprefix("```latex").strip()
        elif stripped.startswith("```tex"):
            stripped = stripped.removeprefix("```tex").strip()
        elif stripped.startswith("```"):
            stripped = stripped.removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped.removesuffix("```").strip()
        return stripped

    def _format_websites_latex(self, visited_sites: set[str] | None) -> str:
        if not visited_sites:
            return ""
        items = "\n".join(
            f"  \\item \\url{{{self._escape_latex(site)}}}"
            for site in sorted(visited_sites)
        )
        return (
            "\\section*{Websites Used in Research}\n"
            "\\begin{itemize}\n"
            f"{items}\n"
            "\\end{itemize}\n"
        )

    def _ensure_complete_latex_document(
        self,
        latex_response: str,
        state: DeepReviewState,
    ) -> str:
        latex_doc = self._strip_latex_fences(latex_response)
        if r"\documentclass" in latex_doc:
            if r"\end{document}" not in latex_doc:
                latex_doc = latex_doc.rstrip() + "\n\\end{document}"
            return latex_doc

        solution = self._escape_latex(state.get("solution", ""))
        question = self._escape_latex(state.get("question", ""))
        model_response = self._escape_latex(latex_response)
        return f"""\\documentclass{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{hyperref}}
\\begin{{document}}
\\section*{{Executive Summary}}
This report summarizes the deep-review process for the following question:

{question}

\\section*{{Final Solution}}
{solution}

\\section*{{Raw Model Report}}
{model_response}

\\end{{document}}"""

    @staticmethod
    def _inject_before_end_document(original_tex: str, injection: str) -> str:
        if not injection.strip():
            return original_tex
        injection_index = original_tex.rfind(r"\end{document}")
        if injection_index == -1:
            return original_tex.rstrip() + "\n" + injection
        return (
            original_tex[:injection_index]
            + "\n"
            + injection
            + "\n"
            + original_tex[injection_index:]
        )

    def summarize_process_as_latex(
        self,
        state: DeepReviewState,
        config: RunnableConfig | None = None,
    ) -> DeepReviewState:
        """Summarize the iterative review process as a valid LaTeX document."""
        events = self.events(config)
        events.emit(
            "Writing LaTeX report",
            stage="summarize",
        )
        iteration_details = ""
        for i, (sol, crit, comp) in enumerate(
            zip(
                state["agent1_solution"],
                state["agent2_critiques"],
                state["agent3_perspectives"],
            ),
            start=1,
        ):
            iteration_details += (
                f"\\subsection*{{Iteration {i}}}\n\n"
                f"\\textbf{{Solution:}}\\\\\n{sol}\n\n"
                f"\\textbf{{Critique:}}\\\\\n{crit}\n\n"
                f"\\textbf{{Competitor Perspective:}}\\\\\n{comp}\n\n"
            )

        timestamp_str = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        txt_filename = Path(
            self.den,
            f"iteration_details_{timestamp_str}_chat_history.txt",
        )
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(iteration_details)

        prompt = f"""\
            You are a system that produces a FULL LaTeX document.
            Here is information about a multi-iteration process:

            Original question: {state["question"]}

            Below are the solutions, critiques, and competitor perspectives from each iteration:

            {iteration_details}

            The solution we arrived at was:

            {state["solution"]}

            Now produce a valid LaTeX document. Be sure to use a table of contents.
            It must start with an Executive Summary (that may be multiple pages) which summarizes
            the entire iterative process. Following that, include the solution in full,
            not summarized, but reformatted for appropriate LaTeX. Finally, include all
            steps - solutions, critiques, and competitor perspectives - in an Appendix,
            and include a listing of all websites used in research if any were used.

            You must ONLY RETURN LaTeX, nothing else. It must be valid LaTeX syntax!

            Your output should start with:
            \\documentclass{{article}}
            \\usepackage[margin=1in]{{geometry}}
            etc.

            It must compile without errors under pdflatex.
        """

        websites_latex = self._format_websites_latex(state.get("visited_sites"))
        latex_response = self.strllm.invoke(prompt)
        latex_doc = self._ensure_complete_latex_document(latex_response, state)
        final_latex = self._inject_before_end_document(
            latex_doc, websites_latex
        )
        events.emit(
            "Report ready",
            stage="summarize_result",
            preview=latex_response,
            output_path=str(txt_filename),
        )
        return {"summary_report": final_latex}

    def _build_graph(self):
        self.tool_llm = self.tool_llm.bind_tools(self.tools.values())

        self.add_node(self.agent1_generate_solution, "agent1")
        self.add_node(self.agent2_critique, "agent2")
        self.add_node(self.agent3_competitor_perspective, "agent3")
        self.add_node(self.tool_node, "tool_node")
        self.add_node(self.increment_iteration, "increment_iteration")
        self.add_node(self.generate_solution, "finalize")
        self.add_node(self.print_visited_sites, "print_sites")
        self.add_node(self.summarize_process_as_latex, "summarize_as_latex")

        self.graph.set_entry_point("agent1")

        self.graph.add_conditional_edges(
            "agent1",
            self._wrap_cond(
                phase_should_continue, "agent1_continue", "deep_review"
            ),
            {"tools": "tool_node", "done": "agent2"},
        )
        self.graph.add_conditional_edges(
            "agent2",
            self._wrap_cond(
                phase_should_continue, "agent2_continue", "deep_review"
            ),
            {"tools": "tool_node", "done": "agent3"},
        )
        self.graph.add_conditional_edges(
            "agent3",
            self._wrap_cond(
                phase_should_continue, "agent3_continue", "deep_review"
            ),
            {"tools": "tool_node", "done": "increment_iteration"},
        )
        self.graph.add_conditional_edges(
            "tool_node",
            self._wrap_cond(
                route_after_tools, "route_after_tools", "deep_review"
            ),
            {"agent1": "agent1", "agent2": "agent2", "agent3": "agent3"},
        )

        self.graph.add_conditional_edges(
            "increment_iteration",
            self._wrap_cond(should_continue, "should_continue", "deep_review"),
            {"continue": "agent1", "finish": "finalize"},
        )

        self.graph.add_edge("finalize", "summarize_as_latex")
        self.graph.add_edge("summarize_as_latex", "print_sites")
        self.graph.set_finish_point("print_sites")


def phase_should_continue(state: DeepReviewState) -> Literal["tools", "done"]:
    messages = state.get("messages") or []
    if messages and getattr(messages[-1], "tool_calls", None):
        return "tools"
    return "done"


def route_after_tools(
    state: DeepReviewState,
) -> Literal["agent1", "agent2", "agent3"]:
    phase = state.get("active_phase")
    if phase in {"agent1", "agent2", "agent3"}:
        return cast(Literal["agent1", "agent2", "agent3"], phase)
    return "agent1"


def should_continue(state: DeepReviewState) -> Literal["continue", "finish"]:
    if state["current_iteration"] >= state["max_iterations"]:
        return "finish"
    return "continue"
