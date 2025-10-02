import os, sys, json
from ast               import literal_eval
from typing            import Annotated, Literal, List, Dict
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.tools    import tool
from langgraph.prebuilt      import ToolNode

from langgraph.prebuilt      import InjectedState, tools_condition
from langgraph.graph         import END, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage

from langchain_openai               import ChatOpenAI
from langchain_community.tools      import DuckDuckGoSearchResults
from langchain_community.tools      import TavilySearchResults
from langchain_core.runnables.graph import MermaidDrawMethod

import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # uses PIL under the hood
import pprint, io


from .base                              import BaseAgent
from ..util.parse import extract_json
from ..util.optimization_schema import ProblemSpec, SolverSpec
from ..util.helperFunctions import extract_tool_calls, run_tool_calls

from ..tools.feasibility_checker   import heuristic_feasibility_check as hfc
from ..tools.feasibility_tools import feasibility_check_auto as fca
# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE  = "\033[94m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"


extractor_prompt = '''You are a Problem Extractor.  
      Goal: Using user’s plain-text description of an optimization problem formulate a rigorous mathematical description of the problem in natural language. Adhere to following instructions.  
      Instructions:  
      1. Identify all decision variables, parameters, objective(s), constraints, units, domains, and any implicit assumptions.  
      2. Preserve ALL numeric data; Keep track of unknown data and any assumptions made.
      3. Do not invent information. If unsure, include a ‘TO_CLARIFY’ note at the end.  
'''

math_formulator_prompt = '''
You are Math Formulator.  
Goal: convert the structured Problem into a Python dictionary described below. Make sure the expressions are Python sympy readable strings. 
class DecisionVariableType(TypedDict):
    name: str                                         # decision variable name
    type: Literal["continuous", "integer", "logical", "infinite-dimensional", "finite-dimensional"] # decision variable type
    domain: str                                       # allowable values of variable 
    description: str                                  # natural language description

class ParameterType(TypedDict):
    name: str                 # parameter name
    value: Optional[Any]      # parameter value; None
    description: str          # natural language description
    is_user_supplied: bool    # 1 if user supplied parameter

class ObjectiveType(TypedDict):
    sense: Literal["minimize", "maximize"]                                          # objective sense
    expression_nl: str                                                              # sympy-representable mathematical expression
    tags: List[Literal["linear", "quadratic", "nonlinear", "convex", "nonconvex"]]  # objective type

class ConstraintType(TypedDict):
    name: str                                                                       # constraint name
    expression_nl: str                                                              # sympy-representable mathematical expression
    tags: List[Literal["linear", "integer", "nonlinear", "equality", "inequality", "infinite-dimensional", "finite-dimensional"]]  # constraint type

class NotesType(TypedDict):
    verifier: str         # problem verification status and explanation
    feasibility: str      # problem feasibility status
    user: str             # notes to user 
    assumptions: str      # assumptions made during formulation

class ProblemSpec(TypedDict):
    title: str                                      # name of the problem
    description_nl: str                             # natural language description 
    decision_variables: List[DecisionVariableType]  # list of all decision variables
    parameters: List[ParameterType]                 # list of all parameters
    objective: ObjectiveType                        # structred objective function details
    constraints: List[ConstraintType]               # structured constraint details
    problem_class: Optional[str]                    # optimization problem class
    latex: Optional[str]                            # latex formulation of the problem
    status: Literal["DRAFT", "VERIFIED", "ERROR"]   # problem status
    notes: NotesType                                # structured notes data

class SolverSpec(TypedDict):
    solver: str                                     # name of the solver, replace with Literal["Gurobi","Ipopt",...] to restrict solvers
    library: str                                    # library or relevant packages for the solver
    algorithm: Optional[str]                        # algorithm used to solve the problem
    license: Optional[str]                          # License status of the solver (open-source, commercial,etc.)
    parameters: Optional[List[dict]]                # other parameters relevant to the problem
    notes: Optional[str]                            # justifying the choice of solver
'''

discretizer_prompt = '''
Remember that only optimization problems with finite dimensional variables can solved on a computer. 
Therefore, given the optimization problem, decide if discretization is needed to optimize. 
If a discretization is needed, reformulate the problem with an appropriate discretization scheme:
0) Ensure all decision variables and constraints of 'infinite-dimensional' type are reduced to 'finite-dimensional' type
1) Make the discretization is numerically stable
2) Accurate
3) Come up with plans to verify convergence and add the plans to notes.user.
'''

feasibility_prompt =  '''
Given the code for an Optimization problem, utilize the tools available to you to test feasibility of
the problem and constraints. Select appropriate tool to perform feasibility tests. 
'''

solver_selector_prompt = """
You are Solver Selector, an expert in optimization algorithms.  
Goal: choose an appropriate solution strategy & software.  
Instructions:  
1. Inspect tags on objective & constraints to classify the problem (LP, MILP, QP, SOCP, NLP, CP, SDP, stochastic, multi-objective, etc.).  
2. Decide convex vs non-convex, deterministic vs stochastic.  
3. Write in the Python Dictionary format below: 
    class SolverSpec(TypedDict):
        solver: str                                     # name of the solver
        library: str                                    # library or relevant packages for the solver
        algorithm: Optional[str]                        # algorithm used to solve the problem
        license: Optional[str]                          # License status of the solver (open-source, commercial,etc.)
        parameters: Optional[List[Dict]]                # other parameters relevant to the problem
        notes: Optional[str]                            # justifying the choice of solver
"""

code_generator_prompt = '''
You are Code Generator, a senior software engineer specialized in optimization libraries.  
Goal: produce runnable code that builds the model, solves it with the recommended solver, and prints the solution clearly. Do not generate anything other than the code.  
Constraints & style guide:  
1. Language: Python ≥3.9 unless another is specified in SolverSpec.  
2. Use a popular modeling interface compatible with the solver (e.g., Pyomo, CVXPY, PuLP, Gurobi API, OR-Tools, JuMP).  
3. Parameter placeholders: values from ProblemSpec.parameters that are null must be read from a YAML/JSON file or user input.  
4. Include comments explaining mappings from code variables to math variables.  
5. Provide minimal CLI wrapper & instructions.  
6. Add unit-test stub that checks feasibility if sample data provided.  
'''

verifier_prompt = '''
You are Verifier, a meticulous QA engineer.  
Goal: statically inspect the formulation & code for logical or syntactic errors. Do NOT execute code.  
Checklist:  
1. Are decision variables in code identical to math notation?  
2. Objective & constraints correctly translated?  
3. Data placeholders clear?  
4. Library imports consistent with recommended_solver?  
5. Any obvious inefficiencies or missing warm-starts? 
6. Check for any numerical instabilities or ill-conditioning.
Actions:  
• If all good → ProblemSpec.status = ‘VERIFIED’; ProblemSpec.notes.verifier = ‘PASS: …’. 
• Else → ProblemSpec.status = ‘ERROR’; ProblemSpec.notes.verifier = detailed list of issues.  
Output updated JSON only.
'''

explainer_prompt  = '''
You are Explainer.  
Goal: craft a concise, user-friendly report.  
Include:  
1. Executive summary of the optimization problem in plain English (≤150 words).  
2. Math formulation (render ProblemSpec.latex).  
2a. Comment on formulation feasibility and expected solution.
3. Explanation of chosen solver & its suitability.  
4. How to run the code, including dependency installation.  
5. Next steps if parameters are still needed.  
Return: Markdown string (no JSON).
'''
class OptimizerState(TypedDict):
    user_input: str
    problem: str
    problem_spec: ProblemSpec
    solver: SolverSpec
    code: str
    problem_diagnostic: List[Dict]
    summary: str

class OptimizationAgent(BaseAgent):
    def __init__(self, llm = "OpenAI/gpt-4o", *args, **kwargs):
        super().__init__(llm, *args, **kwargs)
        self.extractor_prompt         = extractor_prompt
        self.explainer_prompt         = explainer_prompt
        self.verifier_prompt          = verifier_prompt
        self.code_generator_prompt    = code_generator_prompt
        self.solver_selector_prompt   = solver_selector_prompt
        self.math_formulator_prompt   = math_formulator_prompt
        self.discretizer_prompt       = discretizer_prompt
        self.feasibility_prompt       = feasibility_prompt
        self.tools                    = [fca] #[run_cmd, write_code, search_tool, fca]
        self.llm                      = self.llm.bind_tools(self.tools)
        self.tool_maps                = {(getattr(t, "name", None) or getattr(t, "__name__", None)): t for i, t in enumerate(self.tools)} 

        self._initialize_agent()    

    # Define the function that calls the model
    def extractor(self, state:OptimizerState) -> OptimizerState:
        new_state  = state.copy()
        new_state["problem"] = self.llm.invoke([SystemMessage(content=self.extractor_prompt), HumanMessage(content=new_state["user_input"])]).content
        
        new_state["problem_diagnostic"] = []

        print("Extractor:\n")
        pprint.pprint(new_state["problem"])
        return new_state

    def formulator(self, state:OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(ProblemSpec, include_raw=True).invoke([SystemMessage(content=self.math_formulator_prompt), HumanMessage(content=state["problem"])])
        new_state["problem_spec"] = llm_out["parsed"]
        new_state["problem_diagnostic"].extend(extract_tool_calls(llm_out["raw"]))
        
        print("Formulator:\n")
        pprint.pprint(new_state["problem_spec"])
        return new_state

    def discretizer(self, state:OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(ProblemSpec, include_raw=True).invoke([SystemMessage(content=self.discretizer_prompt), HumanMessage(content=state["problem"])])
        new_state["problem_spec"] = llm_out["parsed"]
        new_state["problem_diagnostic"].extend(extract_tool_calls(llm_out["raw"]))
        
        print("Discretizer:\n")
        pprint.pprint(new_state["problem_spec"])
        
        return new_state

    def tester(self, state:OptimizerState) -> OptimizerState:
        new_state = state.copy()
        
        llm_out = self.llm.bind(tool_choice="required").invoke([SystemMessage(content=self.feasibility_prompt), HumanMessage(content=str(state["code"]))])

        tool_log = run_tool_calls(llm_out, self.tool_maps) 
        new_state["problem_diagnostic"].extend(tool_log)

        print("Feasibility Tester:\n")
        for msg in new_state["problem_diagnostic"]:
            msg.pretty_print()
        return new_state

    def selector(self, state:OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(SolverSpec, include_raw=True).invoke([SystemMessage(content=self.solver_selector_prompt), HumanMessage(content=str(state["problem_spec"]))])
        new_state["solver"] = llm_out["parsed"]
    
        print("Selector:\n ")
        pprint.pprint(new_state["solver"])
        return new_state

    def generator(self, state:OptimizerState) -> OptimizerState:
        new_state = state.copy()

        new_state["code"] = self.llm.invoke([SystemMessage(content=self.code_generator_prompt), HumanMessage(content=str(state["problem_spec"]))]).content

        print("Generator:\n")
        pprint.pprint(new_state["code"])
        return new_state
    
    def verifier(self, state:OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(ProblemSpec, include_raw=True).invoke([SystemMessage(content=self.verifier_prompt), HumanMessage(content=str(state["problem_spec"])+state["code"])])
        new_state["problem_spec"] = llm_out["parsed"]
        if hasattr(llm_out,"tool_calls"):
            tool_log = run_tool_calls(llm_out, self.tool_maps) 
            new_state["problem_diagnostic"].extend(tool_log)
        
        print("Verifier:\n ")
        pprint.pprint(new_state["problem_spec"])
        return new_state
    
    def explainer(self, state:OptimizerState) -> OptimizerState:
        new_state = state.copy()

        new_state["summary"] = self.llm.invoke([SystemMessage(content=self.explainer_prompt), HumanMessage(content=state["problem"]+str(state["problem_spec"])), *state["problem_diagnostic"]]).content

        print("Summary:\n")
        pprint.pprint(new_state["summary"])
        return new_state
    
    def _initialize_agent(self):
        self.graph = StateGraph(OptimizerState)

        self.graph.add_node("Problem Extractor",  self.extractor)
        self.graph.add_node("Math Formulator",   self.formulator)
        self.graph.add_node("Solver Selector",     self.selector)
        self.graph.add_node("Code Generator",     self.generator)
        self.graph.add_node("Verifier",            self.verifier)
        self.graph.add_node("Explainer",          self.explainer)
        self.graph.add_node("Feasibility Tester",    self.tester)
        self.graph.add_node("Discretizer",      self.discretizer)

        self.graph.add_edge(START, "Problem Extractor")
        self.graph.add_edge("Problem Extractor",  "Math Formulator")
        # self.graph.add_edge("Math Formulator", "Feasibility Tester")
        self.graph.add_conditional_edges("Math Formulator", should_discretize, 
                                                            {"discretize": "Discretizer", "continue": "Solver Selector"})
        self.graph.add_edge("Discretizer", "Solver Selector")
        self.graph.add_edge("Solver Selector",     "Code Generator")
        self.graph.add_edge("Code Generator",      "Feasibility Tester")
        self.graph.add_edge("Feasibility Tester",  "Verifier")
        self.graph.add_conditional_edges("Verifier",should_continue,
                                                    {"continue":"Explainer", "error":"Problem Extractor"})
        self.graph.add_edge("Explainer",                        END)

        

        self.action = self.graph.compile()
        
        try:
            png_bytes = self.action.get_graph().draw_mermaid_png()
            img = mpimg.imread(io.BytesIO(png_bytes), format='png')  # decode bytes -> array

            plt.imshow(img)
            plt.axis('off')
            plt.show()
        except Exception as e:
            # This requires some extra dependencies and is optional
            print(e)
            pass

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
            cwd=workspace_dir
        )

        stdout, stderr = process.communicate(timeout=60000)
    except KeyboardInterrupt:
        print("Keyboard Interrupt of command: ", query)
        stdout, stderr = "", "KeyboardInterrupt:"

    print("STDOUT: ", stdout)
    print("STDERR: ", stderr)

    return f"STDOUT: {stdout} and STDERR: {stderr}"

@tool
def write_code(code: str, filename: str, state: Annotated[dict, InjectedState]):
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
def should_discretize(state:OptimizerState) -> Literal["Discretize", "continue"]:
    cons = state["problem_spec"]["constraints"]
    decs = state["problem_spec"]["decision_variables"]

    if any("infinite-dimensional" in t["tags"] for t in cons) or any("infinite-dimensional" in t["type"] for t in decs):
        # print(f"Problem has infinite-dimensional constraints/decision variables. Needs to be discretized")
        return "discretize"

    return "continue"


# Define the function that determines whether to continue or not
def should_continue(state: OptimizerState) -> Literal["error", "continue"]:
    spec = state["problem_spec"]
    try:
        status = spec["status"].lower()
    except:
        status = spec["spec"]["status"].lower()
    if "VERIFIED".lower() in status:
        return "continue"
    # Otherwise if there is, we continue
    else:
        return "error"

def main():
    model = ChatOpenAI(
        model       = "gpt-4o",
        max_tokens  = 10000,
        timeout     = None,
        max_retries = 2)
    execution_agent = OptimizationAgent(llm=model)
    # execution_agent = execution_agent.bind_tools(feasibility_checker)
    problem_string = '''
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
    '''
    inputs = {"problem": problem_string}
    result = execution_agent.action.invoke(inputs)
    print(result["messages"][-1].content)
    return result

if __name__ == "__main__":
    main()


#         min⁡ 𝑃𝑔  𝑐1*𝑃1 + 𝑐2 * 𝑃2 + 𝑐3 * 𝑃3