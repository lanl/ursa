import ast
from datetime import datetime
from operator import add, or_
from pathlib import Path
from typing import (
    Annotated,
    Literal,
    Optional,
    TypedDict,
    cast,
)

from langchain.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as lc_tool
from langgraph.prebuilt import InjectedState, ToolNode

from ursa.tools.read_file_tool import read_file

try:
    from ddgs import DDGS  # pip install duckduckgo-search
except Exception:
    DDGS = None


# from langchain_core.runnables.graph import MermaidDrawMethod
from ursa.agents.base import BaseAgent
from ursa.prompt_library.hypothesizer_prompts import (
    competitor_prompt,
    critic_prompt,
    hypothesizer_prompt,
)

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"


# add this as a LangChain (lc) tool so the LLM can figure out what docs there are if it wants to
@lc_tool(
    description="List filenames (and sizes) in input_docs_dir (or workspace)."
)
def list_input_docs(state: Annotated[dict, InjectedState]) -> str:
    from pathlib import Path

    root = state.get("input_docs_dir") or state.get("workspace")
    if not root:
        return "[Error]: no input_docs_dir or workspace set in state."

    p = Path(root).resolve()
    if not p.exists() or not p.is_dir():
        return f"[Error]: input_docs_dir not a directory: {p}"

    rows = []
    for f in sorted([x for x in p.iterdir() if x.is_file()]):
        try:
            rows.append(f"{f.name} ({f.stat().st_size} bytes)")
        except Exception:
            rows.append(f"{f.name} (size unknown)")
    ret = "\n".join(rows) if rows else "No files found."

    # print(f"@lc_tool:list_input_docs() ->\n{ret}")
    return ret


# Define our state schema
class HypothesizerState(TypedDict, total=False):
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
    input_docs_dir: str
    use_search: bool


class HypothesizerAgent(BaseAgent[HypothesizerState]):
    state_type = HypothesizerState

    def __init__(
        self,
        llm: BaseChatModel,
        max_iterations: int = 3,
        extra_tools: Optional[list[BaseTool] | None] = None,
        **kwargs,
    ):
        super().__init__(llm=llm, **kwargs)
        default_tools = [read_file, list_input_docs]
        if extra_tools:
            default_tools.extend(extra_tools)

        # ---- coerce any plain callables into BaseTool via lc_tool ----
        coerced_tools: list[BaseTool] = []
        for t in default_tools:
            if isinstance(t, BaseTool):
                coerced_tools.append(t)
            elif callable(t):
                # wrap plain function into a LangChain-style tool
                coerced_tools.append(lc_tool(t))
            else:
                raise TypeError(f"Unsupported tool type: {type(t)} for {t}")

        # pass tools to BaseAgent like ExecutionAgent does
        self.tools = coerced_tools

        # Bind LLM to tools so it can emit tool calls
        # (this returns a tool-capable llm wrapper)
        try:
            self.llm = self.llm.bind_tools(self.tools)
        except Exception:
            # fallback: some LLMs/versions want an iterable of tool objects
            self.llm = self.llm.bind_tools(self.tools)

        self.tool_node = ToolNode(self.tools)

        # bind tools to the LLM for function/tool calling
        self.llm_with_tools = llm.bind_tools(self.tools)

        # debug print tools enabled
        print("[HypothesizerAgent] Tools enabled:")
        for t in self.tools:
            try:
                print(f"  - {t.name}")
            except Exception:
                print(f"  - (unnamed tool) {t}")

        # keep existing setup
        self.hypothesizer_prompt = hypothesizer_prompt
        self.critic_prompt = critic_prompt
        self.competitor_prompt = competitor_prompt
        # Only create DDGS if the import worked - this helps w/ offline mode too
        self.search_tool = DDGS() if DDGS else None
        self.strllm = self.llm | StrOutputParser()
        self.max_iterations = max_iterations

    def _content_to_text(self, content) -> str:
        """OpenAI Responses models may return list-of-blocks; normalize to plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                # common shapes: {"type": "output_text", "text": "..."} or {"type": "...", ...}
                if isinstance(block, dict):
                    t = block.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t)
                elif isinstance(block, str):
                    if block.strip():
                        parts.append(block)
                else:
                    # last resort
                    s = str(block)
                    if s.strip():
                        parts.append(s)
            return "\n".join(parts)
        # fallback
        return str(content)

    async def _ainvoke_text_with_tools(
        self,
        messages: list[BaseMessage],
        state: HypothesizerState,
        max_rounds: int = 8,
    ) -> str:
        tool_state = dict(state)

        if state.get("input_docs_dir"):
            tool_state["workspace"] = state["input_docs_dir"]
        elif "workspace" not in tool_state and getattr(self, "workspace", None):
            tool_state["workspace"] = str(self.workspace)

        for _ in range(max_rounds):
            ai_msg = await self.llm_with_tools.ainvoke(messages)
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None)
            print("[HypothesizerAgent] tool_calls raw:", tool_calls)
            if not tool_calls:
                return self._content_to_text(
                    getattr(ai_msg, "content", None)
                ).strip()

            def _tc_name(tc):
                if isinstance(tc, dict):
                    return tc.get("name")
                return (
                    getattr(tc, "name", None)
                    or getattr(tc, "tool", None)
                    or str(tc)
                )

            print(
                f"[HypothesizerAgent] Tool calls requested: {[_tc_name(tc) for tc in tool_calls]}"
            )

            invoke_state = dict(tool_state)
            invoke_state["messages"] = list(messages)

            tool_result = await self.tool_node.ainvoke(invoke_state)

            # ToolNode returns ToolMessages that must be appended verbatim
            if isinstance(tool_result, dict) and "messages" in tool_result:
                returned_msgs = tool_result["messages"]
            elif isinstance(tool_result, list):
                returned_msgs = tool_result
            else:
                returned_msgs = []

            print(
                "[HypothesizerAgent] ToolNode returned:",
                [type(m).__name__ for m in returned_msgs],
            )

            for m in returned_msgs:
                if isinstance(m, ToolMessage):
                    messages.append(m)
                else:
                    # fallback only (should be rare)
                    messages.append(HumanMessage(content=f"[Tool output]\n{m}"))

        print(
            "[HypothesizerAgent] Tool loop max rounds reached without final text."
        )
        return ""

    def _normalize_inputs(self, inputs) -> HypothesizerState:
        if isinstance(inputs, str):
            return HypothesizerState(
                question=inputs,
                max_iterations=self.max_iterations,
                current_iteration=0,
            )
        return cast(HypothesizerState, inputs)

    def format_result(self, result: HypothesizerState) -> str:
        return result.get(
            "solution", "Hypothesizer failed to return a solution"
        )

    def parse_visited_sites(self, raw_search_results) -> set[str]:
        visited_sites = set()
        try:
            if isinstance(raw_search_results, str):
                results_list = ast.literal_eval(raw_search_results)
            else:
                results_list = raw_search_results
            # Each item typically might have "link", "title", "snippet"
            for item in results_list:
                link = item.get("link")
                if link:
                    visited_sites.add(link)
        except (ValueError, SyntaxError, TypeError):
            # If it's not valid Python syntax or something else goes wrong
            print("[DEBUG] Could not parse search results as Python list.")
            print("[DEBUG] raw_search_results:", raw_search_results)

        return visited_sites

    def _describe_input_docs(self, input_dir: str | Path) -> str:
        """
        Return fingerprints for ALL files in input_dir (no content).
        This nudges the LLM to use tools to fetch content selectively.
        """
        import hashlib
        from pathlib import Path

        p = Path(input_dir)
        if not p.exists() or not p.is_dir():
            return ""

        lines = []
        for fp in sorted([f for f in p.iterdir() if f.is_file()]):
            try:
                b = fp.read_bytes()
                sha = hashlib.sha256(b).hexdigest()[:8]
                size = fp.stat().st_size
            except Exception:
                sha = "????????"
                size = -1
            lines.append(f"- {fp.name} | bytes={size} | sha256_8={sha}")

        return (
            "DOC_FINGERPRINTS (files available via tools; content not preloaded):\n"
            + "\n".join(lines)
        )

    async def agent1_generate_solution(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """Agent 1: Hypothesizer. Can read local input docs if state['input_docs_dir'] is provided."""
        print(
            f"[iteration {state['current_iteration']}] Entering agent1_generate_solution. Iteration: {state['current_iteration']}"
        )

        current_iter = state["current_iteration"]
        user_content = f"Question: {state['question']}\n"

        # Include previous iteration data if present
        if current_iter > 0:
            if state.get("agent1_solution"):
                user_content += (
                    f"\nPrevious solution: {state['agent1_solution'][-1]}"
                )
            if state.get("agent2_critiques"):
                user_content += f"\nCritique: {state['agent2_critiques'][-1]}"
            if state.get("agent3_perspectives"):
                user_content += f"\nCompetitor perspective: {state['agent3_perspectives'][-1]}"

            user_content += (
                "\n\n**You must explicitly list how this new solution differs from the previous solution,** "
                "point by point, explaining what changes were made in response to the critique and competitor perspective."
                "\nAfterward, provide your updated solution."
            )
        else:
            user_content += "Research this problem and generate a solution."

        user_content += (
            "\n\nTOOLING RULES:\n"
            "- You have tools: list_input_docs and read_file.\n"
            "- Before doing anything, be sure to call list_input_docs to figure out what is there.  Only after that, use read_file"
            "  on relevant docs."
            "- Before citing or relying on ANY local file content, you MUST call read_file on that file.\n"
            "- If you are unsure what files exist, call list_input_docs first.\n"
            "- Read only the minimum set of files needed (start with README if present).\n"
            "- If a PDF looks relevant, explicitly read it with read_file.\n"
        )

        # Option A: include local documents if provided
        input_docs_dir = state.get("input_docs_dir")
        docs_text = ""
        if input_docs_dir:
            fp_text = self._describe_input_docs(input_docs_dir)
            if fp_text:
                user_content += f"\n\nLOCAL FILES AVAILABLE (use tools to read):\n{fp_text}\n"

        # Option B: optionally run a short web search if requested and search tool available
        use_search = state.get(
            "use_search", False
        )  # default False for local-doc workflows
        search_query = ""
        visited_sites = set()
        if use_search and getattr(self, "search_tool", None):
            # Ask the LLM to craft a compact search query
            search_query = await self.strllm.ainvoke(
                f"Here is a problem description: {state['question']}. Turn it into a short query to be fed into a search engine."
            )
            if '"' in (search_query or ""):
                # strip quoted part if model returns something like: "query here"
                search_query = (
                    search_query.split('"')[1]
                    if '"' in search_query
                    else search_query
                )
            raw_search_results = self.search_tool.text(
                search_query or state["question"]
            )
            user_content += f"\n\nSearch results: {raw_search_results}"
            visited_sites = self.parse_visited_sites(raw_search_results)
        else:
            # If not searching, tack on a short instruction letting the LLM know it should rely on local docs
            if docs_text:
                user_content += "\n\nNOTE: Use ONLY the local documents provided above for evidence and context unless asked otherwise."
            else:
                user_content += "\n\nNOTE: No local documents provided and web search disabled; rely on model knowledge."

        # Provide a system message to define this agent's role (same as before)
        messages = [
            SystemMessage(content=self.hypothesizer_prompt),
            HumanMessage(content=user_content),
        ]
        # solution = await self.strllm.ainvoke(messages)
        solution = await self._ainvoke_text_with_tools(messages, state)

        # Print the entire solution in green
        print(f"{GREEN}[Agent1 - Hypothesizer solution]\n{solution}{RESET}")
        print(
            f"[iteration {state['current_iteration']}] Exiting agent1_generate_solution."
        )

        out = {
            "agent1_solution": [solution],
            "question_search_query": search_query,
            "visited_sites": visited_sites,
        }

        return out

    async def agent2_critique(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """Agent 2: Critic."""
        print(
            f"[iteration {state['current_iteration']}] Entering agent2_critique."
        )

        solution = state["agent1_solution"][-1]
        user_content = (
            f"Question: {state['question']}\n"
            f"Proposed solution: {solution}\n"
            "Provide a detailed critique of this solution. Identify potential flaws, assumptions, and areas for improvement."
        )

        user_content += (
            "\n\nYou have tools to list and read local documents. "
            "If any claim in the proposed solution depends on the documents, "
            "you MUST verify by reading the relevant file sections and cite them."
        )

        use_search = bool(state.get("use_search", False))
        visited_sites = set()

        if use_search and getattr(self, "search_tool", None):
            try:
                fact_check_query = f"fact check {state.get('question_search_query', '')} solution effectiveness"
                fact_check_results = self.search_tool.text(fact_check_query)
                visited_sites = self.parse_visited_sites(fact_check_results)
                user_content += f"\nFact check results: {fact_check_results}"
            except Exception as e:
                user_content += (
                    f"\nNOTE: Web search failed ({type(e).__name__}: {e}). "
                    "Proceeding without web results."
                )
        else:
            user_content += "\nNOTE: Web search disabled; critique must rely on local docs + reasoning only."

        messages = [
            SystemMessage(content=self.critic_prompt),
            HumanMessage(content=user_content),
        ]
        critique = await self._ainvoke_text_with_tools(messages, state)

        # Print the entire critique in blue
        print(f"{BLUE}[Agent2 - Critic]\n{critique}{RESET}")
        print(
            f"[iteration {state['current_iteration']}] Exiting agent2_critique."
        )
        return {
            "agent2_critiques": [critique],
            "visited_sites": visited_sites,
        }

    async def agent3_competitor_perspective(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """Agent 3: Competitor/Stakeholder Simulator."""
        print(
            f"[iteration {state['current_iteration']}] Entering agent3_competitor_perspective."
        )

        solution = state["agent1_solution"][-1]
        critique = state["agent2_critiques"][-1]

        user_content = (
            f"Question: {state['question']}\n"
            f"Proposed solution: {solution}\n"
            f"Critique: {critique}\n"
            "Simulate how a competitor, government agency, or other stakeholder might respond to this solution."
        )

        user_content += (
            "\n\nYou have tools to list and read local documents. "
            "Use them to ground the stakeholder response in the provided materials."
        )

        competitor_search_query = (
            f"competitor responses to {state['question_search_query']}"
        )

        use_search = bool(state.get("use_search", False))
        visited_sites = set()

        if use_search and getattr(self, "search_tool", None):
            try:
                competitor_search_query = f"competitor responses to {state.get('question_search_query', '')}"
                competitor_info = self.search_tool.text(competitor_search_query)
                visited_sites = self.parse_visited_sites(competitor_info)
                user_content += f"\nCompetitor information: {competitor_info}"
            except Exception as e:
                user_content += (
                    f"\nNOTE: Web search failed ({type(e).__name__}: {e}). "
                    "Proceeding without web info."
                )
        else:
            user_content += "\nNOTE: Web search disabled; simulate stakeholder reaction without external web info."

        messages = [
            SystemMessage(content=self.competitor_prompt),
            HumanMessage(content=user_content),
        ]
        perspective = await self._ainvoke_text_with_tools(messages, state)

        # Print the entire perspective in red
        print(
            f"{RED}[Agent3 - Competitor/Stakeholder Perspective]\n{perspective}{RESET}"
        )
        print(
            f"[iteration {state['current_iteration']}] Exiting agent3_competitor_perspective."
        )
        return {
            "agent3_perspectives": [perspective],
            "visited_sites": visited_sites,
        }

    def increment_iteration(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        current_iteration = state["current_iteration"] + 1
        print(
            f"[iteration {state['current_iteration']}] Iteration incremented to {current_iteration}"
        )
        return {"current_iteration": current_iteration}

    async def generate_solution(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """Generate the overall, refined solution based on all iterations."""
        print(
            f"[iteration {state['current_iteration']}] Entering generate_solution."
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

        print(
            f"[iteration {state['current_iteration']}] Generating overall solution with LLM..."
        )
        solution = await self.strllm.ainvoke(prompt)
        print(
            f"[iteration {state['current_iteration']}] Overall solution obtained. Preview:",
            solution[:200],
            "...",
        )

        print(
            f"[iteration {state['current_iteration']}] Exiting generate_solution."
        )
        return {"solution": solution}

    def print_visited_sites(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        new_state = state.copy()
        # all_sites = list(new_state["visited_sites"])
        # print("[DEBUG] Visited Sites:")
        # for s in all_sites:
        #     print("  ", s)
        return new_state

    async def summarize_process_as_latex(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """
        Summarize how the solution changed over time, referencing
        each iteration's critique and competitor perspective,
        then produce a final LaTeX document.
        """
        print("Entering summarize_process_as_latex.")
        # Build a single string describing the entire iterative process
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

        # -----------------------------
        # Write iteration_details to disk as .txt
        # -----------------------------
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        txt_filename = Path(
            self.workspace,
            f"iteration_details_{timestamp_str}_chat_history.txt",
        )
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(iteration_details)

        print(f"Wrote iteration details to {txt_filename}.")

        # Prompt the LLM to produce a LaTeX doc
        # We'll just pass it as a single string to the LLM;
        # you could also do system+human messages if you prefer.
        prompt = f"""\
            You are a system that produces a FULL LaTeX document.
            Here is information about a multi-iteration process:

            Original question: {state["question"]}

            Below are the solutions, critiques, and competitor perspectives from each iteration:

            {iteration_details}

            The solution we arrived at was:

            {state["solution"]}

            Now produce a valid LaTeX document.  Be sure to use a table of contents.
            It must start with an Executive Summary (that may be multiple pages) which summarizes
            the entire iterative process.  Following that, we should include the solution in full,
            not summarized, but reformatted for appropriate LaTeX.  And then, finally (and this will be
            quite long), we must take all the steps - solutions, critiques, and competitor perspectives
            and *NOT SUMMARIZE THEM* but merely reformat them for the reader.  This will be in an Appendix
            of the full content of the steps.  Finally, include a listing of all of the websites we
            used in our research.

            You must ONLY RETURN LaTeX, nothing else.  It must be valid LaTeX syntax!

            Your output should start with:
            \\documentclass{{article}}
            \\usepackage[margin=1in]{{geometry}}
            etc.

            It must compile without errors under pdflatex.
        """

        # Now produce a valid LaTeX document that nicely summarizes this entire iterative process.
        # It must include the overall solution in full, not summarized, but reformatted for appropriate
        # LaTeX. The summarization is for the other steps.

        # all_visited_sites = list(state["visited_sites"])
        # (Optional) remove duplicates by converting to a set, then back to a list
        # visited_sites_unique = list(set(all_visited_sites))
        # if visited_sites_unique:
        #     websites_latex = "\\section*{Websites Visited}\\begin{itemize}\n"
        #     for url in visited_sites_unique:
        #         print(f"We visited: {url}")
        #         # Use \url{} to handle special characters in URLs
        #         websites_latex += f"\\item \\url{{{url}}}\n"
        #     websites_latex += "\\end{itemize}\n\n"
        # else:
        #     # If no sites visited, or the list is empty
        #     websites_latex = (
        #         "\\section*{Websites Visited}\nNo sites were visited.\n\n"
        #     )
        # print(websites_latex)
        websites_latex = ""

        # Ask the LLM to produce *only* LaTeX content
        latex_response = await self.strllm.ainvoke(prompt)

        latex_doc = latex_response

        def inject_into_latex(original_tex: str, injection: str) -> str:
            """
            Find the last occurrence of '\\end{document}' in 'original_tex'
            and insert 'injection' right before it.
            If '\\end{document}' is not found, just append the injection at the end.
            """
            injection_index = original_tex.rfind(r"\end{document}")
            if injection_index == -1:
                # If the LLM didn't include \end{document}, just append
                return original_tex + "\n" + injection
            else:
                # Insert right before \end{document}
                return (
                    original_tex[:injection_index]
                    + "\n"
                    + injection
                    + "\n"
                    + original_tex[injection_index:]
                )

        final_latex = inject_into_latex(latex_doc, websites_latex)

        print(
            f"[iteration {state['current_iteration']}] Received LaTeX from LLM. Preview:"
        )
        print(latex_response[:300], "...")
        print(
            f"[iteration {state['current_iteration']}] Exiting summarize_process_as_latex."
        )
        return {"summary_report": final_latex}

    def _build_graph(self):
        # Add nodes
        self.add_node(self.agent1_generate_solution, "agent1")
        self.add_node(self.agent2_critique, "agent2")
        self.add_node(self.agent3_competitor_perspective, "agent3")
        self.add_node(self.increment_iteration, "increment_iteration")
        self.add_node(self.generate_solution, "finalize")
        self.add_node(self.print_visited_sites, "print_sites")
        self.add_node(self.summarize_process_as_latex, "summarize_as_latex")

        # Add simple edges for the known flow
        self.graph.add_edge("agent1", "agent2")
        self.graph.add_edge("agent2", "agent3")
        self.graph.add_edge("agent3", "increment_iteration")

        # Then from increment_iteration, we have a conditional:
        # If we 'continue', we go back to agent1
        # If we 'finish', we jump to the finalize node
        self.graph.add_conditional_edges(
            "increment_iteration",
            should_continue,
            {"continue": "agent1", "finish": "finalize"},
        )

        self.graph.add_edge("finalize", "summarize_as_latex")
        self.graph.add_edge("summarize_as_latex", "print_sites")
        # self.graph.add_edge("summarize_as_latex", "compile_pdf")
        # self.graph.add_edge("compile_pdf", "print_sites")

        # Set the entry point
        self.graph.set_entry_point("agent1")
        self.graph.set_finish_point("print_sites")


def should_continue(state: HypothesizerState) -> Literal["continue", "finish"]:
    if state["current_iteration"] >= state["max_iterations"]:
        print(
            f"[iteration {state['current_iteration']}] Reached max_iterations; finishing."
        )
        return "finish"
    else:
        print(
            f"[iteration {state['current_iteration']}] Still under max_iterations; continuing."
        )
        return "continue"


# def compile_summary_to_pdf(state: AgentState) -> AgentState:
#     """
#     Takes the LaTeX in state["summary_report"] and tries to compile it to a PDF
#     named with the model and timestamp, e.g.:
#     summary_report_gpt-5-mini_Mar_15_2025_8:59am.pdf
#     """
#     print(f"[DEBUG] Entering compile_summary_to_pdf.")

#     llm_model = state["llm_model"]


#     latex_code = state.get("summary_report", "")
#     if not latex_code:
#         print("[DEBUG] No LaTeX code found in summary_report.")
#         return state

#     # Create a dynamic filename using the LLM model name & a timestamp
#     # e.g. "summary_report_gpt-5-mini_Mar_15_2025_08:59AM.pdf"
#     # timestamp_str = datetime.now().strftime("%b_%d_%Y_%I:%M%p")
#     timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#     pdf_filename = f"summary_report_{llm_model}_{timestamp_str}.pdf"

#     tex_filename = "summary_report.tex"
#     with open(tex_filename, "w", encoding="utf-8") as f:
#         f.write(latex_code)

#     try:
#         subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_filename], check=True)
#         subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_filename], check=True)
#     except subprocess.CalledProcessError as e:
#         print("Error compiling LaTeX:", e)

#     if os.path.exists("summary_report.pdf"):
#         os.rename("summary_report.pdf", pdf_filename)
#         print(f"[DEBUG] Successfully compiled PDF -> {pdf_filename}")
#     else:
#         print("[DEBUG] PDF compilation failed; no summary_report.pdf found.")

#     print("[DEBUG] Exiting compile_summary_to_pdf.")
#     return state


if __name__ == "__main__":
    # Create the graph
    hypothesizer_agent = HypothesizerAgent()

    question = "Find a city with as least 10 vowels in its name."

    # Initialize the state
    initial_state: HypothesizerState = {
        "question": question,
        "question_search_query": "",
        "current_iteration": 0,
        "max_iterations": 3,
        "agent1_solution": [],
        "agent2_critiques": [],
        "agent3_perspectives": [],
        "solution": "",
        "summary_report": "",
        "visited_sites": set(),
    }

    print("Invoking the graph...")
    # Run the graph
    result = hypothesizer_agent.invoke(
        initial_state,
        {
            "recursion_limit": 999999,
            "configurable": {"thread_id": 42},
        },
    )
    summary_text = result["summary_report"]

    print("Graph invocation complete.")

    # Print the overall solution
    print("Overall Solution:")
    print(result["solution"])

    # print("Summarized Report:")
    # print(summary_text)
