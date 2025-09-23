import ast

# from langchain_community.tools import TavilySearchResults
# from textwrap                  import dedent
from datetime import datetime
from typing import List, Literal, TypedDict

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from ..prompt_library.hypothesizer_prompts import (
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
class HypothesizerState(TypedDict):
    question: str
    question_search_query: str
    current_iteration: int
    max_iterations: int
    agent1_solution: List[str]  # List to store each iteration of solutions
    agent2_critiques: List[str]  # List to store critiques
    agent3_perspectives: List[str]  # List to store competitor perspectives
    solution: str  # Refined solution
    summary_report: str  # the final summarized report
    visited_sites: List[str]


class HypothesizerAgent(BaseAgent):
    def __init__(self, llm: str | BaseChatModel = "openai/o3-mini", **kwargs):
        super().__init__(llm, **kwargs)
        self.hypothesizer_prompt = hypothesizer_prompt
        self.critic_prompt = critic_prompt
        self.competitor_prompt = competitor_prompt
        self.search_tool = DuckDuckGoSearchResults(
            output_format="json", num_results=10
        )
        # self.search_tool = TavilySearchResults(
        #     max_results=10, search_depth="advanced", include_answer=False
        # )

        self._initialize_agent()

    def agent1_generate_solution(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """Agent 1: Hypothesizer."""
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Entering agent1_generate_solution. Iteration: {state['current_iteration']}"
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

        search_query = self.llm.invoke(
            f"Here is a problem description: {state['question']}. Turn it into a short query to be fed into a search engine."
        ).content
        if '"' in search_query:
            search_query = search_query.split('"')[1]
        raw_search_results = self.search_tool.invoke(search_query)

        # Parse the results if possible, so we can collect URLs
        new_state = state.copy()
        new_state["question_search_query"] = search_query
        if "visited_sites" not in new_state:
            new_state["visited_sites"] = []

        try:
            if isinstance(raw_search_results, str):
                results_list = ast.literal_eval(raw_search_results)
            else:
                results_list = raw_search_results
            # Each item typically might have "link", "title", "snippet"
            for item in results_list:
                link = item.get("link")
                if link:
                    print(f"[DEBUG] Appending visited link: {link}")
                    new_state["visited_sites"].append(link)
        except (ValueError, SyntaxError, TypeError):
            # If it's not valid Python syntax or something else goes wrong
            print("[DEBUG] Could not parse search results as Python list.")
            print("[DEBUG] raw_search_results:", raw_search_results)

        user_content += f"\nSearch results: {raw_search_results}"

        # Provide a system message to define this agent's role
        messages = [
            SystemMessage(content=self.hypothesizer_prompt),
            HumanMessage(content=user_content),
        ]
        solution = self.llm.invoke(messages)

        new_state["agent1_solution"].append(solution.content)

        # Print the entire solution in green
        print(
            f"{GREEN}[Agent1 - Hypothesizer solution]\n{solution.content}{RESET}"
        )
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Exiting agent1_generate_solution."
        )
        return new_state

    def agent2_critique(self, state: HypothesizerState) -> HypothesizerState:
        """Agent 2: Critic."""
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Entering agent2_critique."
        )

        solution = state["agent1_solution"][-1]
        user_content = (
            f"Question: {state['question']}\n"
            f"Proposed solution: {solution}\n"
            "Provide a detailed critique of this solution. Identify potential flaws, assumptions, and areas for improvement."
        )

        fact_check_query = f"fact check {state['question_search_query']} solution effectiveness"

        raw_search_results = self.search_tool.invoke(fact_check_query)

        # Parse the results if possible, so we can collect URLs
        new_state = state.copy()
        if "visited_sites" not in new_state:
            new_state["visited_sites"] = []

        try:
            if isinstance(raw_search_results, str):
                results_list = ast.literal_eval(raw_search_results)
            else:
                results_list = raw_search_results
            # Each item typically might have "link", "title", "snippet"
            for item in results_list:
                link = item.get("link")
                if link:
                    print(f"[DEBUG] Appending visited link: {link}")
                    new_state["visited_sites"].append(link)
        except (ValueError, SyntaxError, TypeError):
            # If it's not valid Python syntax or something else goes wrong
            print("[DEBUG] Could not parse search results as Python list.")
            print("[DEBUG] raw_search_results:", raw_search_results)

        fact_check_results = raw_search_results
        user_content += f"\nFact check results: {fact_check_results}"

        messages = [
            SystemMessage(content=self.critic_prompt),
            HumanMessage(content=user_content),
        ]
        critique = self.llm.invoke(messages)

        new_state["agent2_critiques"].append(critique.content)

        # Print the entire critique in blue
        print(f"{BLUE}[Agent2 - Critic]\n{critique.content}{RESET}")
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Exiting agent2_critique."
        )
        return new_state

    def agent3_competitor_perspective(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """Agent 3: Competitor/Stakeholder Simulator."""
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Entering agent3_competitor_perspective."
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

        raw_search_results = self.search_tool.invoke(competitor_search_query)

        # Parse the results if possible, so we can collect URLs
        new_state = state.copy()
        if "visited_sites" not in new_state:
            new_state["visited_sites"] = []

        try:
            if isinstance(raw_search_results, str):
                results_list = ast.literal_eval(raw_search_results)
            else:
                results_list = raw_search_results
            # Each item typically might have "link", "title", "snippet"
            for item in results_list:
                link = item.get("link")
                if link:
                    print(f"[DEBUG] Appending visited link: {link}")
                    new_state["visited_sites"].append(link)
        except (ValueError, SyntaxError, TypeError):
            # If it's not valid Python syntax or something else goes wrong
            print("[DEBUG] Could not parse search results as Python list.")
            print("[DEBUG] raw_search_results:", raw_search_results)

        competitor_info = raw_search_results
        user_content += f"\nCompetitor information: {competitor_info}"

        messages = [
            SystemMessage(content=self.competitor_prompt),
            HumanMessage(content=user_content),
        ]
        perspective = self.llm.invoke(messages)

        new_state["agent3_perspectives"].append(perspective.content)

        # Print the entire perspective in red
        print(
            f"{RED}[Agent3 - Competitor/Stakeholder Perspective]\n{perspective.content}{RESET}"
        )
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Exiting agent3_competitor_perspective."
        )
        return new_state

    def increment_iteration(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        new_state = state.copy()
        new_state["current_iteration"] += 1
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Iteration incremented to {new_state['current_iteration']}"
        )
        return new_state

    def generate_solution(self, state: HypothesizerState) -> HypothesizerState:
        """Generate the overall, refined solution based on all iterations."""
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Entering generate_solution."
        )
        prompt = f"Original question: {state['question']}\n\n"
        prompt += "Evolution of solutions:\n"

        for i in range(state["max_iterations"]):
            prompt += f"\nIteration {i + 1}:\n"
            prompt += f"Solution: {state['agent1_solution'][i]}\n"
            prompt += f"Critique: {state['agent2_critiques'][i]}\n"
            prompt += (
                f"Competitor perspective: {state['agent3_perspectives'][i]}\n"
            )

        prompt += "\nBased on this iterative process, provide the overall, refined solution."

        print(
            f"[iteration {state['current_iteration']} - DEBUG] Generating overall solution with LLM..."
        )
        solution = self.llm.invoke(prompt)
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Overall solution obtained. Preview:",
            solution.content[:200],
            "...",
        )

        new_state = state.copy()
        new_state["solution"] = solution.content

        print(
            f"[iteration {state['current_iteration']} - DEBUG] Exiting generate_solution."
        )
        return new_state

    def print_visited_sites(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        new_state = state.copy()
        all_sites = new_state.get("visited_sites", [])
        print("[DEBUG] Visited Sites:")
        for s in all_sites:
            print("  ", s)
        return new_state

    def summarize_process_as_latex(
        self, state: HypothesizerState
    ) -> HypothesizerState:
        """
        Summarize how the solution changed over time, referencing
        each iteration's critique and competitor perspective,
        then produce a final LaTeX document.
        """
        print("[DEBUG] Entering summarize_process_as_latex.")
        llm_model = state.get("llm_model", "unknown_model")

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
        txt_filename = (
            f"iteration_details_{llm_model}_{timestamp_str}_chat_history.txt"
        )
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(iteration_details)

        print(f"[DEBUG] Wrote iteration details to {txt_filename}.")

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

        all_visited_sites = state.get("visited_sites", [])
        # (Optional) remove duplicates by converting to a set, then back to a list
        visited_sites_unique = list(set(all_visited_sites))
        if visited_sites_unique:
            websites_latex = "\\section*{Websites Visited}\\begin{itemize}\n"
            for url in visited_sites_unique:
                print(f"We visited: {url}")
                # Use \url{} to handle special characters in URLs
                websites_latex += f"\\item \\url{{{url}}}\n"
            websites_latex += "\\end{itemize}\n\n"
        else:
            # If no sites visited, or the list is empty
            websites_latex = (
                "\\section*{Websites Visited}\nNo sites were visited.\n\n"
            )
        print(websites_latex)

        # Ask the LLM to produce *only* LaTeX content
        latex_response = self.llm.invoke(prompt)

        latex_doc = latex_response.content

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

        new_state = state.copy()
        new_state["summary_report"] = final_latex

        print(
            f"[iteration {state['current_iteration']} - DEBUG] Received LaTeX from LLM. Preview:"
        )
        print(latex_response.content[:300], "...")
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Exiting summarize_process_as_latex."
        )
        return new_state

    def _initialize_agent(self):
        # Initialize the graph
        self.graph = StateGraph(HypothesizerState)

        # Add nodes
        self.graph.add_node("agent1", self.agent1_generate_solution)
        self.graph.add_node("agent2", self.agent2_critique)
        self.graph.add_node("agent3", self.agent3_competitor_perspective)
        self.graph.add_node("increment_iteration", self.increment_iteration)
        self.graph.add_node("finalize", self.generate_solution)
        self.graph.add_node("print_sites", self.print_visited_sites)
        self.graph.add_node(
            "summarize_as_latex", self.summarize_process_as_latex
        )
        # self.graph.add_node("compile_pdf",                compile_summary_to_pdf)

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
        self.graph.add_edge("print_sites", END)
        # self.graph.add_edge("summarize_as_latex", "compile_pdf")
        # self.graph.add_edge("compile_pdf", "print_sites")

        # Set the entry point
        self.graph.set_entry_point("agent1")

        self.action = self.graph.compile(checkpointer=self.checkpointer)
        # self.action.get_graph().draw_mermaid_png(output_file_path="hypothesizer_agent_graph.png", draw_method=MermaidDrawMethod.PYPPETEER)

    def run(self, prompt, max_iter=3, recursion_limit=99999):
        # Initialize the state
        initial_state = HypothesizerState(
            question=prompt,
            current_iteration=0,
            max_iterations=max_iter,
            agent1_solution=[],
            agent2_critiques=[],
            agent3_perspectives=[],
            solution="",
        )
        # Run the graph
        result = self.action.invoke(
            initial_state,
            {
                "recursion_limit": recursion_limit,
                "configurable": {"thread_id": self.thread_id},
            },
        )
        return result["solution"]


def should_continue(state: HypothesizerState) -> Literal["continue", "finish"]:
    if state["current_iteration"] >= state["max_iterations"]:
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Reached max_iterations; finishing."
        )
        return "finish"
    else:
        print(
            f"[iteration {state['current_iteration']} - DEBUG] Still under max_iterations; continuing."
        )
        return "continue"


# def compile_summary_to_pdf(state: AgentState) -> AgentState:
#     """
#     Takes the LaTeX in state["summary_report"] and tries to compile it to a PDF
#     named with the model and timestamp, e.g.:
#     summary_report_gpt-4o-mini_Mar_15_2025_8:59am.pdf
#     """
#     print(f"[DEBUG] Entering compile_summary_to_pdf.")

#     llm_model = state["llm_model"]


#     latex_code = state.get("summary_report", "")
#     if not latex_code:
#         print("[DEBUG] No LaTeX code found in summary_report.")
#         return state

#     # Create a dynamic filename using the LLM model name & a timestamp
#     # e.g. "summary_report_gpt-4o-mini_Mar_15_2025_08:59AM.pdf"
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
    initial_state = HypothesizerState(
        question=question,
        current_iteration=0,
        max_iterations=3,
        agent1_solution=[],
        agent2_critiques=[],
        agent3_perspectives=[],
        solution="",
    )

    print("[DEBUG] Invoking the graph...")
    # Run the graph
    result = hypothesizer_agent.action.invoke(
        initial_state,
        {
            "recursion_limit": 999999,
            "configurable": {"thread_id": 42},
        },
    )
    summary_text = result["summary_report"]

    print("[DEBUG] Graph invocation complete.")

    # Print the overall solution
    print("Overall Solution:")
    print(result["solution"])

    # print("Summarized Report:")
    # print(summary_text)
