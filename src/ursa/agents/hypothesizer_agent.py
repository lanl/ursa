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
)
from langchain_core.tools import BaseTool, tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Overwrite

from ursa.tools import read_file

try:
    from ddgs import DDGS  # pip install duckduckgo-search
except Exception:
    DDGS = None


# from langchain_core.runnables.graph import MermaidDrawMethod
from ursa.agents.base import AgentWithTools, BaseAgent
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


# add this as a LangChain tool so the LLM can figure out what docs there are if it wants to
@tool(
    description="List filenames (and sizes) in input_docs_dir (or workspace)."
)
def list_input_docs(state: Annotated[dict, InjectedState]) -> str:
    from pathlib import Path

    print(f"{BLUE}[list_input_docs] called{RESET}")

    root = state.get("input_docs_dir")
    if not root:
        msg = "[Error]: no input_docs_dir set in state."
        print(f"{RED}[list_input_docs] {msg}{RESET}")
        return msg

    print(f"{BLUE}[list_input_docs] root:{RESET} {root}")

    p = Path(root).resolve()
    if not p.exists() or not p.is_dir():
        msg = f"[Error]: input_docs_dir not a directory: {p}"
        print(f"{RED}[list_input_docs] {msg}{RESET}")
        return msg

    rows = []
    for f in sorted([x for x in p.iterdir() if x.is_file()]):
        try:
            rows.append(f"{f.name} ({f.stat().st_size} bytes)")
        except Exception:
            rows.append(f"{f.name} (size unknown)")

    ret = "\n".join(rows) if rows else "No files found."

    print(f"{GREEN}[list_input_docs] found {len(rows)} file(s){RESET}")
    if rows:
        preview = "\n".join(rows[:5])
        print(f"{GREEN}[list_input_docs] preview:{RESET}\n{preview}")

    return ret


@tool(
    description="Search the web for relevant information and return concise results."
)
def web_search(query: str) -> str:
    print(f"{BLUE}[web_search] called{RESET}")
    print(f"{BLUE}[web_search] query:{RESET} {query}")

    if DDGS is None:
        msg = (
            "[Error]: web search is unavailable because ddgs is not installed."
        )
        print(f"{RED}[web_search] {msg}{RESET}")
        return msg

    try:
        results = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=5), start=1):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                results.append(f"Title: {title}\nSnippet: {body}\nURL: {href}")

                if i == 1:
                    preview = (
                        f"Title: {title}\nSnippet: {body[:200]}\nURL: {href}"
                    )
                    print(
                        f"{GREEN}[web_search] first result preview:{RESET}\n{preview}"
                    )

        if results:
            print(
                f"{GREEN}[web_search] returned {len(results)} result(s){RESET}"
            )
            return "\n\n".join(results)

        print(f"{BLUE}[web_search] no results found{RESET}")
        return "No results found."

    except Exception as e:
        msg = f"[Error performing web search: {type(e).__name__}: {e}]"
        print(f"{RED}[web_search] {msg}{RESET}")
        return msg


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
    messages: Annotated[list[BaseMessage], add_messages]


class HypothesizerAgent(AgentWithTools, BaseAgent[HypothesizerState]):
    state_type = HypothesizerState

    def __init__(
        self,
        llm: BaseChatModel,
        max_iterations: int = 3,
        extra_tools: Optional[list[BaseTool] | None] = None,
        enable_web_search: bool = False,
        **kwargs,
    ):
        default_tools = [read_file, list_input_docs]

        # add web search tool if 'enable_web_search' is on
        if enable_web_search:
            default_tools.append(web_search)

        if extra_tools:
            default_tools.extend(extra_tools)

        super().__init__(llm=llm, tools=default_tools, **kwargs)

        # debug print tools enabled
        print("[HypothesizerAgent] Tools enabled:")
        for t in self.tools.values():
            try:
                print(f"  - {t.name}")
            except Exception:
                print(f"  - (unnamed tool) {t}")

        # keep existing setup
        self.hypothesizer_prompt = hypothesizer_prompt
        self.critic_prompt = critic_prompt
        self.competitor_prompt = competitor_prompt
        self.max_iterations = max_iterations

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

    async def agent1_generate_solution(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """Agent 1: Hypothesizer. One LLM step only; tool routing is handled by the graph."""
        print(
            f"[iteration {state['current_iteration'] + 1}] Entering agent1_generate_solution. Iteration: {state['current_iteration'] + 1}"
        )

        messages = list(state.get("messages", []))

        # If messages already exist, we are returning from a tool call.
        # Invoke once more using the accumulated state messages, but return only the new AI message.
        if messages:
            ai_msg = await self.llm_with_tools.ainvoke(messages)
            return {"messages": [ai_msg]}

        current_iter = state["current_iteration"]
        user_content = f"Question: {state['question']}\n"

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
            "- First call list_input_docs to discover what local files are available.\n"
            "- Then use read_file only on the files you think are relevant.\n"
            "- Before citing or relying on ANY local file content, you MUST call read_file on that file.\n"
            "- Read only the minimum set of files needed (start with README if present).\n"
            "- If a PDF looks relevant, explicitly read it with read_file.\n"
        )

        search_query = ""
        visited_sites = set()

        if state.get("input_docs_dir"):
            user_content += "\n\nNOTE: Use the local documents in input_docs_dir for evidence and context unless asked otherwise."
        elif "web_search" in self.tools:
            user_content += "\n\nNOTE: Web search is enabled if needed. Use the web_search tool only when local documents are insufficient."
        else:
            user_content += "\n\nNOTE: No local documents provided and web search disabled; rely on model knowledge."

        prompt_messages = [
            SystemMessage(content=self.hypothesizer_prompt),
            HumanMessage(content=user_content),
        ]

        ai_msg = await self.llm_with_tools.ainvoke(prompt_messages)

        print(
            f"[iteration {state['current_iteration'] + 1}] Exiting agent1_generate_solution."
        )

        return {
            "messages": prompt_messages + [ai_msg],
            "question_search_query": search_query,
            "visited_sites": visited_sites,
        }

    def finalize_agent1_solution(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        messages = list(state.get("messages", []))

        last_ai = None
        for m in reversed(messages):
            if getattr(m, "type", None) == "ai":
                last_ai = m
                break

        solution = (
            (getattr(last_ai, "text", "") or "").strip() if last_ai else ""
        )

        print(f"{GREEN}[Agent1 - Hypothesizer solution]\n{solution}{RESET}")
        print(
            f"[iteration {state['current_iteration'] + 1}] Finalized agent1 solution."
        )

        return {
            "agent1_solution": [solution],
            "messages": Overwrite([]),
        }

    async def agent2_critique(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """Agent 2: Critic. One LLM step only; tool routing is handled by the graph."""
        print(
            f"[iteration {state['current_iteration'] + 1}] Entering agent2_critique."
        )

        messages = list(state.get("messages", []))

        # If messages already exist, we are returning from a tool call.
        if messages:
            ai_msg = await self.llm_with_tools.ainvoke(messages)
            return {"messages": [ai_msg]}

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

        visited_sites = set()

        if "web_search" in self.tools:
            user_content += "\nNOTE: Web search is enabled if needed. Use the web_search tool if local documents are insufficient for fact-checking."
        else:
            user_content += "\nNOTE: Web search disabled; critique must rely on local docs + reasoning only."

        messages = [
            SystemMessage(content=self.critic_prompt),
            HumanMessage(content=user_content),
        ]

        ai_msg = await self.llm_with_tools.ainvoke(messages)

        print(
            f"[iteration {state['current_iteration'] + 1}] Exiting agent2_critique."
        )

        return {
            "messages": messages + [ai_msg],
            "visited_sites": visited_sites,
        }

    def finalize_agent2_critique(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        messages = list(state.get("messages", []))

        last_ai = None
        for m in reversed(messages):
            if getattr(m, "type", None) == "ai":
                last_ai = m
                break

        critique = (
            (getattr(last_ai, "text", "") or "").strip() if last_ai else ""
        )

        print(f"{BLUE}[Agent2 - Critic]\n{critique}{RESET}")
        print(
            f"[iteration {state['current_iteration'] + 1}] Finalized agent2 critique."
        )

        return {
            "agent2_critiques": [critique],
            "messages": Overwrite([]),
        }

    async def agent3_competitor_perspective(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """Agent 3: Competitor/Stakeholder Simulator. One LLM step only; tool routing is handled by the graph."""
        print(
            f"[iteration {state['current_iteration'] + 1}] Entering agent3_competitor_perspective."
        )

        messages = list(state.get("messages", []))

        # If messages already exist, we are returning from a tool call.
        if messages:
            ai_msg = await self.llm_with_tools.ainvoke(messages)
            return {"messages": [ai_msg]}

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

        visited_sites = set()

        if "web_search" in self.tools:
            user_content += "\nNOTE: Web search is enabled if needed. Use the web_search tool if outside information would improve the stakeholder simulation."
        else:
            user_content += "\nNOTE: Web search disabled; simulate stakeholder reaction without external web info."

        messages = [
            SystemMessage(content=self.competitor_prompt),
            HumanMessage(content=user_content),
        ]

        ai_msg = await self.llm_with_tools.ainvoke(messages)

        print(
            f"[iteration {state['current_iteration'] + 1}] Exiting agent3_competitor_perspective."
        )

        return {
            "messages": messages + [ai_msg],
            "visited_sites": visited_sites,
        }

    def finalize_agent3_perspective(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        messages = list(state.get("messages", []))

        last_ai = None
        for m in reversed(messages):
            if getattr(m, "type", None) == "ai":
                last_ai = m
                break

        perspective = (
            (getattr(last_ai, "text", "") or "").strip() if last_ai else ""
        )

        print(
            f"{RED}[Agent3 - Competitor/Stakeholder Perspective]\n{perspective}{RESET}"
        )
        print(
            f"[iteration {state['current_iteration'] + 1}] Finalized agent3 perspective."
        )

        return {
            "agent3_perspectives": [perspective],
            "messages": Overwrite([]),
        }

    def increment_iteration(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        current_iteration = state["current_iteration"] + 1
        return {"current_iteration": current_iteration}

    def generate_solution(self, state: HypothesizerState) -> HypothesizerState:
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

        response = self.llm.invoke(prompt)
        solution = (getattr(response, "text", None) or str(response)).strip()

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

    def _strip_code_fences(self, text: str) -> str:
        """
        Normalize model output so it is ready to write as a .tex file.
        Removes markdown fences and trims any leading junk before \\documentclass.
        """
        if not text:
            return text

        s = text.strip()

        if s.startswith("```"):
            lines = s.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()

        doc_idx = s.find(r"\documentclass")
        if doc_idx != -1:
            s = s[doc_idx:]

        return s.strip()

    def _latex_looks_complete(self, text: str) -> bool:
        s = text or ""
        return r"\documentclass" in s and r"\end{document}" in s

    def _complete_latex_document(
        self,
        prompt: str,
        max_rounds: int = 3,
    ) -> str:
        """
        Generate LaTeX, and if the model truncates before \\end{document},
        ask it to continue from where it left off.
        """
        latex_llm = self.llm.bind(max_completion_tokens=20000)
        response = latex_llm.invoke(prompt)
        text = self._strip_code_fences(
            (getattr(response, "text", None) or str(response)).strip()
        )

        if self._latex_looks_complete(text):
            return text

        current = text

        for _ in range(max_rounds):
            continuation_prompt = (
                "You previously started a LaTeX document but stopped before finishing.\n\n"
                "Continue exactly from where the document stopped.\n"
                "Do not restart from \\documentclass.\n"
                "Do not use markdown code fences.\n"
                "Finish the document and include \\end{document}.\n\n"
                "Current partial document:\n\n"
                f"{current}\n"
            )

            cont_response = latex_llm.invoke(continuation_prompt)
            cont_text = self._strip_code_fences(
                (
                    getattr(cont_response, "text", None) or str(cont_response)
                ).strip()
            )

            if cont_text.startswith(r"\documentclass"):
                # Model restarted instead of continuing. Keep the longer version.
                if len(cont_text) > len(current):
                    current = cont_text
            else:
                current = current.rstrip() + "\n" + cont_text.lstrip()

            if self._latex_looks_complete(current):
                return current

        return current

    def summarize_process_as_latex(
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

            You must return only raw LaTeX source.
            Do not use markdown code fences.
            Your output must be a complete LaTeX document that ends with \\end{{document}}.
            If the document would be long, still finish it completely.

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
        latex_doc = self._complete_latex_document(prompt)
        latex_response = latex_doc

        if not isinstance(latex_response, str):
            latex_response = str(latex_response)

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
        # Bind tools to the LLM used by the three agent stages
        self.llm_with_tools = self.llm.bind_tools(self.tools.values())

        # Add nodes
        self.add_node(self.agent1_generate_solution, "agent1")
        self.add_node(self.finalize_agent1_solution, "agent1_finalize")

        self.add_node(self.agent2_critique, "agent2")
        self.add_node(self.finalize_agent2_critique, "agent2_finalize")

        self.add_node(self.agent3_competitor_perspective, "agent3")
        self.add_node(self.finalize_agent3_perspective, "agent3_finalize")

        self.add_node(self.increment_iteration, "increment_iteration")
        self.add_node(self.generate_solution, "finalize")
        self.add_node(self.print_visited_sites, "print_sites")
        self.add_node(self.summarize_process_as_latex, "summarize_as_latex")

        # Separate tool nodes for each stage
        self.graph.add_node("agent1_tools", ToolNode(self.tools.values()))
        self.graph.add_node("agent2_tools", ToolNode(self.tools.values()))
        self.graph.add_node("agent3_tools", ToolNode(self.tools.values()))

        # agent1 -> tools loop or finalize -> agent2
        self.graph.add_conditional_edges(
            "agent1",
            tools_condition,
            {"tools": "agent1_tools", "__end__": "agent1_finalize"},
        )
        self.graph.add_edge("agent1_tools", "agent1")
        self.graph.add_edge("agent1_finalize", "agent2")

        # agent2 -> tools loop or finalize -> agent3
        self.graph.add_conditional_edges(
            "agent2",
            tools_condition,
            {"tools": "agent2_tools", "__end__": "agent2_finalize"},
        )
        self.graph.add_edge("agent2_tools", "agent2")
        self.graph.add_edge("agent2_finalize", "agent3")

        # agent3 -> tools loop or finalize -> increment_iteration
        self.graph.add_conditional_edges(
            "agent3",
            tools_condition,
            {"tools": "agent3_tools", "__end__": "agent3_finalize"},
        )
        self.graph.add_edge("agent3_tools", "agent3")
        self.graph.add_edge("agent3_finalize", "increment_iteration")

        # Existing iteration control flow
        self.graph.add_conditional_edges(
            "increment_iteration",
            should_continue,
            {"continue": "agent1", "finish": "finalize"},
        )

        self.graph.add_edge("finalize", "summarize_as_latex")
        self.graph.add_edge("summarize_as_latex", "print_sites")

        # Entry / exit
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
        "max_iterations": 1,
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
