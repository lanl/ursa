import os
import pprint
import subprocess
from typing import Annotated, Any, Literal, Mapping, TypedDict

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState

from ..prompt_library.optimization_prompts import (
    code_generator_prompt,
    discretizer_prompt,
    explainer_prompt,
    extractor_prompt,
    feasibility_prompt,
    math_formulator_prompt,
    solver_selector_prompt,
    verifier_prompt,
)
from ..tools.feasibility_tools import feasibility_check_auto as fca
from ..util.helperFunctions import extract_tool_calls, run_tool_calls
from ..util.optimization_schema import ProblemSpec, SolverSpec
from .base import BaseAgent

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class OptimizerState(TypedDict):
    user_input: str
    problem: str
    problem_spec: ProblemSpec
    solver: SolverSpec
    code: str
    problem_diagnostic: list[dict]
    summary: str


class OptimizationAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel = init_chat_model("openai:gpt-5-mini"),
        *args,
        **kwargs,
    ):
        super().__init__(llm, *args, **kwargs)
        self.extractor_prompt = extractor_prompt
        self.explainer_prompt = explainer_prompt
        self.verifier_prompt = verifier_prompt
        self.code_generator_prompt = code_generator_prompt
        self.solver_selector_prompt = solver_selector_prompt
        self.math_formulator_prompt = math_formulator_prompt
        self.discretizer_prompt = discretizer_prompt
        self.feasibility_prompt = feasibility_prompt
        self.tools = [fca]  # [run_cmd, write_code, search_tool, fca]
        self.llm = self.llm.bind_tools(self.tools)
        self.tool_maps = {
            (getattr(t, "name", None) or getattr(t, "__name__", None)): t
            for i, t in enumerate(self.tools)
        }

        self._action = self._build_graph()

    # Define the function that calls the model
    def extractor(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()
        new_state["problem"] = self.llm.invoke([
            SystemMessage(content=self.extractor_prompt),
            HumanMessage(content=new_state["user_input"]),
        ]).content

        new_state["problem_diagnostic"] = []

        print("Extractor:\n")
        pprint.pprint(new_state["problem"])
        return new_state

    def formulator(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(
            ProblemSpec, include_raw=True
        ).invoke([
            SystemMessage(content=self.math_formulator_prompt),
            HumanMessage(content=state["problem"]),
        ])
        new_state["problem_spec"] = llm_out["parsed"]
        new_state["problem_diagnostic"].extend(
            extract_tool_calls(llm_out["raw"])
        )

        print("Formulator:\n")
        pprint.pprint(new_state["problem_spec"])
        return new_state

    def discretizer(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(
            ProblemSpec, include_raw=True
        ).invoke([
            SystemMessage(content=self.discretizer_prompt),
            HumanMessage(content=state["problem"]),
        ])
        new_state["problem_spec"] = llm_out["parsed"]
        new_state["problem_diagnostic"].extend(
            extract_tool_calls(llm_out["raw"])
        )

        print("Discretizer:\n")
        pprint.pprint(new_state["problem_spec"])

        return new_state

    def tester(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.bind(tool_choice="required").invoke([
            SystemMessage(content=self.feasibility_prompt),
            HumanMessage(content=str(state["code"])),
        ])

        tool_log = run_tool_calls(llm_out, self.tool_maps)
        new_state["problem_diagnostic"].extend(tool_log)

        print("Feasibility Tester:\n")
        for msg in new_state["problem_diagnostic"]:
            msg.pretty_print()
        return new_state

    def selector(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(
            SolverSpec, include_raw=True
        ).invoke([
            SystemMessage(content=self.solver_selector_prompt),
            HumanMessage(content=str(state["problem_spec"])),
        ])
        new_state["solver"] = llm_out["parsed"]

        print("Selector:\n ")
        pprint.pprint(new_state["solver"])
        return new_state

    def generator(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        new_state["code"] = self.llm.invoke([
            SystemMessage(content=self.code_generator_prompt),
            HumanMessage(content=str(state["problem_spec"])),
        ]).content

        print("Generator:\n")
        pprint.pprint(new_state["code"])
        return new_state

    def verifier(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(
            ProblemSpec, include_raw=True
        ).invoke([
            SystemMessage(content=self.verifier_prompt),
            HumanMessage(content=str(state["problem_spec"]) + state["code"]),
        ])
        new_state["problem_spec"] = llm_out["parsed"]
        if hasattr(llm_out, "tool_calls"):
            tool_log = run_tool_calls(llm_out, self.tool_maps)
            new_state["problem_diagnostic"].extend(tool_log)

        print("Verifier:\n ")
        pprint.pprint(new_state["problem_spec"])
        return new_state

    def explainer(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        new_state["summary"] = self.llm.invoke([
            SystemMessage(content=self.explainer_prompt),
            HumanMessage(content=state["problem"] + str(state["problem_spec"])),
            *state["problem_diagnostic"],
        ]).content

        print("Summary:\n")
        pprint.pprint(new_state["summary"])
        return new_state

    def _build_graph(self):
        graph = StateGraph(OptimizerState)

        self.add_node(graph, self.extractor, "Problem Extractor")
        self.add_node(graph, self.formulator, "Math Formulator")
        self.add_node(graph, self.selector, "Solver Selector")
        self.add_node(graph, self.generator, "Code Generator")
        self.add_node(graph, self.verifier, "Verifier")
        self.add_node(graph, self.explainer, "Explainer")
        self.add_node(graph, self.tester, "Feasibility Tester")
        self.add_node(graph, self.discretizer, "Discretizer")

        graph.add_edge(START, "Problem Extractor")
        graph.add_edge("Problem Extractor", "Math Formulator")
        graph.add_conditional_edges(
            "Math Formulator",
            should_discretize,
            {"discretize": "Discretizer", "continue": "Solver Selector"},
        )
        graph.add_edge("Discretizer", "Solver Selector")
        graph.add_edge("Solver Selector", "Code Generator")
        graph.add_edge("Code Generator", "Feasibility Tester")
        graph.add_edge("Feasibility Tester", "Verifier")
        graph.add_conditional_edges(
            "Verifier",
            should_continue,
            {"continue": "Explainer", "error": "Problem Extractor"},
        )
        graph.add_edge("Explainer", END)

        return graph.compile()

    def _invoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 100000, **_
    ):
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )
        if "user_input" not in inputs:
            try:
                inputs["user_input"] = inputs["messages"][0].content
            except KeyError:
                raise ("'user_input' is a required argument")

        return self._action.invoke(inputs, config)


#########  try:
#########      png_bytes = compiled_graph.get_graph().draw_mermaid_png()
#########      img = mpimg.imread(io.BytesIO(png_bytes), format='png')  # decode bytes -> array

#########      plt.imshow(img)
#########      plt.axis('off')
#########      plt.show()
#########  except Exception as e:
#########      # This requires some extra dependencies and is optional
#########      print(e)
#########      pass


@tool
def run_cmd(query: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Run a commandline command from using the subprocess package in python

    Args:
        query: commandline command to be run as a string given to the subprocess.run command.
    """
    workspace_dir = state["workspace"]
    print("RUNNING: ", query)
    try:
        process = subprocess.Popen(
            query.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workspace_dir,
        )

        stdout, stderr = process.communicate(timeout=60000)
    except KeyboardInterrupt:
        print("Keyboard Interrupt of command: ", query)
        stdout, stderr = "", "KeyboardInterrupt:"

    print("STDOUT: ", stdout)
    print("STDERR: ", stderr)

    return f"STDOUT: {stdout} and STDERR: {stderr}"


@tool
def write_code(
    code: str, filename: str, state: Annotated[dict, InjectedState]
) -> str:
    """
    Writes python or Julia code to a file in the given workspace as requested.

    Args:
        code: The code to write
        filename: the filename with an appropriate extension for programming language (.py for python, .jl for Julia, etc.)

    Returns:
        Execution results
    """
    workspace_dir = state["workspace"]
    print("Writing filename ", filename)
    try:
        # Extract code if wrapped in markdown code blocks
        if "```" in code:
            code_parts = code.split("```")
            if len(code_parts) >= 3:
                # Extract the actual code
                if "\n" in code_parts[1]:
                    code = "\n".join(code_parts[1].strip().split("\n")[1:])
                else:
                    code = code_parts[2].strip()

        # Write code to a file
        code_file = os.path.join(workspace_dir, filename)

        with open(code_file, "w") as f:
            f.write(code)
        print(f"Written code to file: {code_file}")

        return f"File {filename} written successfully."

    except Exception as e:
        print(f"Error generating code: {str(e)}")
        # Return minimal code that prints the error
        return f"Failed to write {filename} successfully."


search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)
# search_tool = TavilySearchResults(max_results=10, search_depth="advanced", include_answer=True)


# A function to test if discretization is needed
def should_discretize(
    state: OptimizerState,
) -> Literal["Discretize", "continue"]:
    cons = state["problem_spec"]["constraints"]
    decs = state["problem_spec"]["decision_variables"]

    if any("infinite-dimensional" in t["tags"] for t in cons) or any(
        "infinite-dimensional" in t["type"] for t in decs
    ):
        # print(f"Problem has infinite-dimensional constraints/decision variables. Needs to be discretized")
        return "discretize"

    return "continue"


# Define the function that determines whether to continue or not
def should_continue(state: OptimizerState) -> Literal["error", "continue"]:
    spec = state["problem_spec"]
    try:
        status = spec["status"].lower()
    except KeyError:
        status = spec["spec"]["status"].lower()
    if "VERIFIED".lower() in status:
        return "continue"
    # Otherwise if there is, we continue
    else:
        return "error"


def main():
    model = ChatOpenAI(
        model="gpt-5-mini", max_tokens=10000, timeout=None, max_retries=2
    )
    execution_agent = OptimizationAgent(llm=model)
    # execution_agent = execution_agent.bind_tools(feasibility_checker)
    problem_string = """
    Solve the following optimal power flow problem
    System topology and data:
        - Three buses (nodes) labeled 1, 2 and 3.
        - One generator at each bus; each can only inject power (no negative output).
        - Loads of 1 p.u. at bus 1, 2 p.u. at bus 2, and 4 p.u. at bus 3.
        - Transmission lines connecting every pair of buses, with susceptances (B):
            - Line 1–2: B₁₂ = 10
            - Line 1–3: B₁₃ = 20
            - Line 2–3: B₂₃ = 30

    Decision variables:
        - Voltage angles θ₁, θ₂, θ₃ (in radians) at buses 1–3.
        - Generator outputs Pᵍ₁, Pᵍ₂, Pᵍ₃ ≥ 0 (in per-unit).

    Reference angle:
        - To fix the overall angle‐shift ambiguity, we set θ₁ = 0 (“slack” or reference bus).

    Objective:
        - Minimize total generation cost with
            - 𝑐1 = 1
            - 𝑐2 = 10
            - 𝑐3 = 100

    Line‐flow limits
        - Lines 1-2 and 1-3 are thermal‐limited to ±0.5 p.u., line 2-3 is unconstrained.
    
    In words:
    We choose how much each generator should produce (at non-negative cost) and the voltage angles at each bus (with bus 1 set to zero) so that supply exactly meets demand, flows on the critical lines don’t exceed their limits, and the total cost is as small as possible. 
    Use the tools at your disposal to check if your formulation is feasible.
    """
    inputs = {"user_input": problem_string}
    result = execution_agent.invoke(inputs)
    print(result["messages"][-1].content)
    return result


if __name__ == "__main__":
    main()


#         min⁡ 𝑃𝑔  𝑐1*𝑃1 + 𝑐2 * 𝑃2 + 𝑐3 * 𝑃3
