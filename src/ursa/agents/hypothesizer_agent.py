import ast
from datetime import datetime
from operator import add, or_
from pathlib import Path
from typing import Annotated, Literal, TypedDict, cast

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

try:
    from ddgs import DDGS  # pip install duckduckgo-search
except Exception:
    DDGS = None


from ursa.prompt_library.hypothesizer_prompts import (
    competitor_prompt,
    critic_prompt,
    hypothesizer_prompt,
)

# from langchain_core.runnables.graph import MermaidDrawMethod
from .base import BaseAgent

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"


# Define our state schema
class HypothesizerState(TypedDict, total=False):
    question: str
    question_search_query: str
    current_iteration: int
    max_iterations: int
    agent1_solution: Annotated[list[str], add]
    agent2_critiques: list[str]
    agent3_perspectives: list[str]
    solution: str
    summary_report: str
    visited_sites: Annotated[set[str], or_]


class HypothesizerAgent(BaseAgent[HypothesizerState]):
    state_type = HypothesizerState

    def __init__(self, llm: BaseChatModel, max_iterations: int = 3, **kwargs):
        super().__init__(llm, **kwargs)
        self.hypothesizer_prompt = hypothesizer_prompt
        self.critic_prompt = critic_prompt
        self.competitor_prompt = competitor_prompt
        self.search_tool = DDGS()
        self.strllm = self.llm | StrOutputParser()
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
        """Agent 1: Hypothesizer."""
        print(
            f"[iteration {state['current_iteration']}] Entering agent1_generate_solution. Iteration: {state['current_iteration']}"
        )

        current_iter = state["current_iteration"]
        user_content = f"Question: {state['question']}\n"

        if current_iter > 0:
            user_content += (
                f"\nPrevious solution: {state['agent1_solution'][-1]}"
            )
            user_content += f"\nCritique: {state['agent2_critiques'][-1]}"
            user_content += (
                f"\nCompetitor perspective: {state['agent3_perspectives'][-1]}"
            )

            user_content += (
                "\n\n**You must explicitly list how this new solution differs from the previous solution,** "
                "point by point, explaining what changes were made in response to the critique and competitor perspective."
                "\nAfterward, provide your updated solution."
            )
        else:
            user_content += "Research this problem and generate a solution."

        search_query = await self.strllm.ainvoke(
            f"Here is a problem description: {state['question']}. Turn it into a short query to be fed into a search engine."
        )
        if '"' in search_query:
            search_query = search_query.split('"')[1]
        raw_search_results = self.search_tool.text(
            search_query or state["question"]
        )
        user_content += f"\nSearch results: {raw_search_results}"

        # Parse the results if possible, so we can collect URLs
        visited_sites = self.parse_visited_sites(raw_search_results)

        # Provide a system message to define this agent's role
        messages = [
            SystemMessage(content=self.hypothesizer_prompt),
            HumanMessage(content=user_content),
        ]
        solution = await self.strllm.ainvoke(messages)

        # Print the entire solution in green
        print(f"{GREEN}[Agent1 - Hypothesizer solution]\n{solution}{RESET}")
        print(
            f"[iteration {state['current_iteration']}] Exiting agent1_generate_solution."
        )
        return {
            "agent1_solution": [solution],
            "question_search_query": search_query,
            "visited_sites": visited_sites,
        }

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

        fact_check_query = f"fact check {state['question_search_query']} solution effectiveness"

        fact_check_results = self.search_tool.text(fact_check_query)
        visited_sites = self.parse_visited_sites(fact_check_results)
        user_content += f"\nFact check results: {fact_check_results}"

        messages = [
            SystemMessage(content=self.critic_prompt),
            HumanMessage(content=user_content),
        ]
        critique = await self.strllm.ainvoke(messages)

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

        competitor_search_query = (
            f"competitor responses to {state['question_search_query']}"
        )

        competitor_info = self.search_tool.text(competitor_search_query)
        visited_sites = self.parse_visited_sites(competitor_info)
        user_content += f"\nCompetitor information: {competitor_info}"

        messages = [
            SystemMessage(content=self.competitor_prompt),
            HumanMessage(content=user_content),
        ]
        perspective = await self.strllm.ainvoke(messages)

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
        "visited_sites": [],
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
