import json
import os
import subprocess
from typing import Any, Mapping, Optional, TypedDict

import tiktoken
from langchain.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.rule import Rule
from rich.table import Table
from rich import box
from rich.console import Group
import difflib

from .base import BaseAgent

working = True
try:
    import atomman as am
    import trafilatura
except Exception:
    working = False


class LammpsState(TypedDict, total=False):
    simulation_task: str
    elements: list[str]
    template: Optional[str]
    chosen_potential: Optional[Any]

    matches: list[Any]
    idx: int
    summaries: list[str]
    full_texts: list[str]
    summaries_combined: str

    input_script: str
    run_returncode: Optional[int]
    run_stdout: str
    run_stderr: str
    run_history: list[dict[str, Any]]  
    
    fix_attempts: int


class LammpsAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        potential_files: Optional[list[str]] = None,
        pair_style: Optional[str] = None,
        pair_coeff: Optional[str] = None,
        max_potentials: int = 5,
        max_fix_attempts: int = 10,
        find_potential_only: bool = False,
        data_file: str = None,
        data_max_lines: int = 50,
        ngpus: int = -1,
        mpi_procs: int = 8,
        workspace: str = "./workspace",
        lammps_cmd: str = "lmp_mpi",
        mpirun_cmd: str = "mpirun",
        tiktoken_model: str = "gpt-5-mini",
        max_tokens: int = 200000,
        **kwargs,
    ):
        if not working:
            raise ImportError(
                "LAMMPS agent requires the atomman and trafilatura dependencies. These can be installed using 'pip install ursa-ai[lammps]' or, if working from a local installation, 'pip install -e .[lammps]' ."
            )
        
        self.user_potential_files = potential_files
        self.user_pair_style = pair_style
        self.user_pair_coeff = pair_coeff
        self.use_user_potential = (potential_files is not None and pair_style is not None and pair_coeff is not None)
        
        self.max_potentials = max_potentials
        self.max_fix_attempts = max_fix_attempts
        self.find_potential_only = find_potential_only
        self.data_file = data_file
        self.data_max_lines = data_max_lines
        self.ngpus = ngpus
        self.mpi_procs = mpi_procs
        self.lammps_cmd = lammps_cmd
        self.mpirun_cmd = mpirun_cmd
        self.tiktoken_model = tiktoken_model
        self.max_tokens = max_tokens

        self.console = Console()
        
        self.pair_styles = [
            "eam",
            "eam/alloy",
            "eam/fs",
            "meam",
            "adp",
            "kim",
            "snap",
            "quip",
            "mlip",
            "pace",
            "nep",
        ]

        self.workspace = workspace
        os.makedirs(self.workspace, exist_ok=True)

        super().__init__(llm, **kwargs)

        self.str_parser = StrOutputParser()

        self.summ_chain = (
            ChatPromptTemplate.from_template(
                "Here is some data about an interatomic potential: {metadata}\n\n"
                "Briefly summarize why it could be useful for this task: {simulation_task}."
            )
            | self.llm
            | self.str_parser
        )

        self.choose_chain = (
            ChatPromptTemplate.from_template(
                "Here are the summaries of a certain number of interatomic potentials: {summaries_combined}\n\n"
                "Pick one potential which would be most useful for this task: {simulation_task}.\n\n"
                "Return your answer **only** as valid JSON, with no extra text or formatting.\n\n"
                "Use this exact schema:\n"
                "{{\n"
                '  "Chosen index": <int>,\n'
                '  "rationale": "<string>",\n'
                '  "Potential name": "<string>"\n'
                "}}\n"
            )
            | self.llm
            | self.str_parser
        )

        
        self.author_chain = (
            ChatPromptTemplate.from_template(
                "Your task is to write a LAMMPS input file for this purpose: {simulation_task}.\n"
                "Note that all potential files are in the './' directory.\n"
                "Here is some information about the pair_style and pair_coeff that might be useful in writing the input file: {pair_info}.\n"
                "If a template for the input file is provided, you should adapt it appropriately to meet the task requirements.\n"
                "Template provided (if any): {template}\n"
                "If a data file is provided, use it in the input script via the 'read_data' command.\n"
                "Name of data file (if any): {data_file}\n"
                "First few lines of data file (if any):\n{data_content}\n"
                "Ensure that all logs are recorded in a './log.lammps' file.\n"
                "To create the log file, use may use the 'log ./log.lammps' command. \n"
                "Return your answer **only** as valid JSON, with no extra text or formatting.\n"
                "IMPORTANT: Properly escape all special characters in the input_script string (use \\n for newlines, \\\\ for backslashes, etc.).\n"
                "Use this exact schema:\n"
                "{{\n"
                '  "input_script": "<string>"\n'
                "}}\n"
            )
            | self.llm
            | self.str_parser
        )

        self.fix_chain = (
            ChatPromptTemplate.from_template(
                "You are part of a larger scientific workflow whose purpose is to accomplish this task: {simulation_task}\n"
                "Multiple attempts at writing and running a LAMMPS input file have been made.\n"
                "Here is the run history across attempts (each includes the input script and its stdout/stderr):{err_message}\n"
                "Use the history to identify what changed between attempts and avoid repeating failed approaches.\n"
                "Your task is to write a new input file that resolves the latest error.\n"
                "Note that all potential files are in the './' directory.\n"
                "Here is some information about the pair_style and pair_coeff that might be useful in writing the input file: {pair_info}.\n"
                "If a template for the input file is provided, you should adapt it appropriately to meet the task requirements.\n"
                "Template provided (if any): {template}\n"
                "If a data file is provided, use it in the input script via the 'read_data' command.\n"
                "Name of data file (if any): {data_file}\n"
                "First few lines of data file (if any):\n{data_content}\n"
                "Ensure that all logs are recorded in a './log.lammps' file.\n"
                "To create the log file, use may use the 'log ./log.lammps' command. \n"
                "Return your answer **only** as valid JSON, with no extra text or formatting.\n"
                "IMPORTANT: Properly escape all special characters in the input_script string (use \\n for newlines, \\\\ for backslashes, etc.).\n"
                "Use this exact schema:\n"
                "{{\n"
                '  "input_script": "<string>"\n'
                "}}\n"
            )
            | self.llm
            | self.str_parser
        )

        self._action = self._build_graph()

    def _section(self, title: str):
        self.console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))
    
    def _panel(self, title: str, body: str, style: str = "cyan"):
        self.console.print(Panel(body, title=f"[bold]{title}[/bold]", border_style=style))
    
    def _code_panel(self, title: str, code: str, language: str = "bash", style: str = "magenta"):
        syn = Syntax(code, language, theme="monokai", line_numbers=True, word_wrap=True)
        self.console.print(Panel(syn, title=f"[bold]{title}[/bold]", border_style=style))

    def _diff_panel(self, old: str, new: str, title: str = "LAMMPS input diff"):
        diff = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile="in.lammps (before)",
                tofile="in.lammps (after)",
                lineterm="",
            )
        )
        if not diff.strip():
            diff = "(no changes)"
        syn = Syntax(diff, "diff", theme="monokai", line_numbers=False, word_wrap=True)
        self.console.print(Panel(syn, title=f"[bold]{title}[/bold]", border_style="cyan"))


    @staticmethod
    def _safe_json_loads(s: str) -> dict[str, Any]:
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("`")
            i = s.find("\n")
            if i != -1:
                s = s[i + 1 :].strip()
        return json.loads(s)

    def _read_and_trim_data_file(self, data_file_path: str) -> str:
        """Read LAMMPS data file and trim to token limit for LLM context."""
        if os.path.exists(data_file_path):
            with open(data_file_path, 'r') as f:
                content = f.read()
            lines = content.splitlines()
            if len(lines) > self.data_max_lines:
                content = "\n".join(lines[:self.data_max_lines])
                print(f"Data file trimmed from {len(lines)} to {self.data_max_lines} lines")
            return content
        else:       
            return (f"Could not read data file.")
        
    def _copy_data_file(self, data_file_path: str) -> str:
        """Copy data file to workspace and return new path."""
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
        filename = os.path.basename(data_file_path)
        dest_path = os.path.join(self.workspace, filename)
        os.system(f"cp {data_file_path} {dest_path}")
        print(f"Data file copied to workspace: {dest_path}")
        return dest_path

    def _copy_user_potential_files(self):
        """Copy user-provided potential files to workspace."""
        print("Copying user-provided potential files to workspace...")
        for pot_file in self.user_potential_files:
            if not os.path.exists(pot_file):
                raise FileNotFoundError(f"Potential file not found: {pot_file}")
            
            filename = os.path.basename(pot_file)
            dest_path = os.path.join(self.workspace, filename)
            
            try:
                os.system(f"cp {pot_file} {dest_path}")
                print(f"Potential files copied to workspace: {dest_path}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")
                raise

    def _create_user_potential_wrapper(self, state: LammpsState) -> LammpsState:
        """Create a wrapper object for user-provided potential to match atomman interface."""
        self._copy_user_potential_files()
        
        # Create a simple object that mimics the atomman potential interface
        class UserPotential:
            def __init__(self, pair_style, pair_coeff):
                self._pair_style = pair_style
                self._pair_coeff = pair_coeff
            
            def pair_info(self):
                return f"pair_style {self._pair_style}\npair_coeff {self._pair_coeff}"
        
        user_potential = UserPotential(self.user_pair_style, self.user_pair_coeff)
        
        return {**state, "chosen_potential": user_potential, "fix_attempts": 0,"run_history": []}
    
    def _fetch_and_trim_text(self, url: str) -> str:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "No metadata available"
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            include_links=False,
            favor_recall=True,
        )
        if not text:
            return "No metadata available"
        text = text.strip()
        try:
            enc = tiktoken.encoding_for_model(self.tiktoken_model)
            toks = enc.encode(text)
            if len(toks) > self.max_tokens:
                toks = toks[: self.max_tokens]
                text = enc.decode(toks)
        except Exception:
            pass
        return text

    def _entry_router(self, state: LammpsState) -> dict:
        # Check if using user-provided potential
        if self.use_user_potential:
            if self.find_potential_only:
                raise Exception(
                    "Cannot set find_potential_only=True when providing your own potential!"
                )
            print("Using user-provided potential files")

        if self.find_potential_only and state.get("chosen_potential"):
            raise Exception(
                "You cannot set find_potential_only=True and also specify your own potential!"
            )
        
        if self.data_file:
            try:
                self._copy_data_file(self.data_file)
            except Exception as e:
                print(f"Warning: Could not process data file: {e}")

        if not state.get("chosen_potential"):
            self.potential_summaries_dir = os.path.join(
                self.workspace, "potential_summaries"
            )
            os.makedirs(self.potential_summaries_dir, exist_ok=True)
        return {}

    def _find_potentials(self, state: LammpsState) -> LammpsState:
        db = am.library.Database(remote=True)
        matches = db.get_lammps_potentials(
            pair_style=self.pair_styles, elements=state["elements"]
        )

        return {
            **state,
            "matches": list(matches),
            "idx": 0,
            "summaries": [],
            "full_texts": [],
            "fix_attempts": 0,
            "run_history": [],
        }

    def _should_summarize(self, state: LammpsState) -> str:
        matches = state.get("matches", [])
        i = state.get("idx", 0)
        if not matches:
            print("No potentials found in NIST for this task. Exiting....")
            return "done_no_matches"
        if i < min(self.max_potentials, len(matches)):
            return "summarize_one"
        return "summarize_done"

    def _summarize_one(self, state: LammpsState) -> LammpsState:
        i = state["idx"]
        self._section(f"Summarizing potential #{i}")
        match = state["matches"][i]
        md = match.metadata()

        if md.get("comments") is None:
            text = "No metadata available"
            summary = "No summary available"
        else:
            lines = md["comments"].split("\n")
            url = lines[1] if len(lines) > 1 else ""
            text = (
                self._fetch_and_trim_text(url)
                if url
                else "No metadata available"
            )
            summary = self.summ_chain.invoke({
                "metadata": text,
                "simulation_task": state["simulation_task"],
            })

        summary_file = os.path.join(
            self.potential_summaries_dir, "potential_" + str(i) + ".txt"
        )
        with open(summary_file, "w") as f:
            f.write(summary)

        return {
            **state,
            "idx": i + 1,
            "summaries": [*state["summaries"], summary],
            "full_texts": [*state["full_texts"], text],
        }

    def _build_summaries(self, state: LammpsState) -> LammpsState:
        parts = []
        for i, s in enumerate(state["summaries"]):
            rec = state["matches"][i]
            parts.append(f"\nSummary of potential #{i}: {rec.id}\n{s}\n")
        return {**state, "summaries_combined": "".join(parts)}

    def _choose(self, state: LammpsState) -> LammpsState:
        self._section("Choosing potential")
        choice = self.choose_chain.invoke({
            "summaries_combined": state["summaries_combined"],
            "simulation_task": state["simulation_task"],
        })
        choice_dict = self._safe_json_loads(choice)
        chosen_index = int(choice_dict["Chosen index"])
        
        chosen_potential = state["matches"][chosen_index]

        self._panel(
            "Chosen Potential",
            f"[bold]Index:[/bold] {chosen_index}\n[bold]ID:[/bold] {chosen_potential.id}\n\n[bold]Rationale:[/bold]\n{choice_dict['rationale']}",
            style="green",
        )
        
        out_file = os.path.join(self.potential_summaries_dir, "Rationale.txt")
        with open(out_file, "w") as f:
            f.write(f"Chosen potential #{chosen_index}")
            f.write("\n")
            f.write("Rationale for choosing this potential:")
            f.write("\n")
            f.write(choice_dict["rationale"])

        return {**state, "chosen_potential": chosen_potential}

    def _route_after_summarization(self, state: LammpsState) -> str:
        if self.find_potential_only:
            return "Exit"
        return "continue_author"

    def _author(self, state: LammpsState) -> LammpsState:
        self._section("First attempt at writing LAMMPS input file")

        if not self.use_user_potential:
            state["chosen_potential"].download_files(self.workspace)
        pair_info = state["chosen_potential"].pair_info()
        
        data_content = ""
        if self.data_file:
            data_content = self._read_and_trim_data_file(self.data_file)

        authored_json = self.author_chain.invoke({
            "simulation_task": state["simulation_task"],
            "pair_info": pair_info,
            "template": state["template"],
            "data_file": self.data_file,
            "data_content": data_content,
        })
        script_dict = self._safe_json_loads(authored_json)
        input_script = script_dict["input_script"]
        with open(os.path.join(self.workspace, "in.lammps"), "w") as f:
            f.write(input_script)

        self._section("Authored LAMMPS input")
        self._code_panel("in.lammps", input_script, language="bash", style="magenta")
                    
        return {**state, "input_script": input_script}

    def _run_lammps(self, state: LammpsState) -> LammpsState:
        self._section("Running LAMMPS")
        
        if self.ngpus >= 0:
            result = subprocess.run(
                [
                    self.mpirun_cmd,
                    "-np",
                    str(self.mpi_procs),
                    self.lammps_cmd,
                    "-in",
                    "in.lammps",
                    "-k",
                    "on",
                    "g",
                    str(self.ngpus),
                    "-sf",
                    "kk",
                    "-pk",
                    "kokkos",
                    "neigh",
                    "half",
                    "newton",
                    "on",
                ],
                cwd=self.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            print(result)
        else:
            result = subprocess.run(
                [
                    self.mpirun_cmd,
                    "-np",
                    str(self.mpi_procs),
                    self.lammps_cmd,
                    "-in",
                    "in.lammps",
                ],
                cwd=self.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

        status_style = "green" if result.returncode == 0 else "red"
        self._panel("Run Result", f"returncode = {result.returncode}", style=status_style)

        if result.returncode != 0:
            err_view = (result.stderr.strip() + "\n" + result.stdout.strip()).strip() or "(no output captured)"
            self._panel("Run error/output", err_view[-6000:], style="red") 
        
        hist = list(state.get("run_history", []))
        hist.append({
            "attempt": state.get("fix_attempts", 0),
            "input_script": state.get("input_script", ""),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        })

        return {
            **state,
            "run_returncode": result.returncode,
            "run_stdout": result.stdout,
            "run_stderr": result.stderr,
            "run_history": hist,
        }


    def _route_run(self, state: LammpsState) -> str:
        rc = state.get("run_returncode", 0)
        attempts = state.get("fix_attempts", 0)
        if rc == 0:
            self._section("LAMMPS run successful! Exiting...")
            return "done_success"
        if attempts < self.max_fix_attempts:
            self._section("LAMMPS run Failed. Attempting to rewrite input file...")
            return "need_fix"
        self._section("LAMMPS run Failed and maximum fix attempts reached. Exiting..")
        return "done_failed"

    def _fix(self, state: LammpsState) -> LammpsState:
        pair_info = state["chosen_potential"].pair_info()

        hist = state.get("run_history", [])
        if not hist:
            hist = [{
                "attempt": state.get("fix_attempts", 0),
                "input_script": state.get("input_script", ""),
                "returncode": state.get("run_returncode"),
                "stdout": state.get("run_stdout", ""),
                "stderr": state.get("run_stderr", ""),
            }]
        
        parts = []
        for h in hist:
            parts.append(
                "=== Attempt {attempt} | returncode={returncode} ===\n"
                "--- input_script ---\n{input_script}\n"
                "--- stdout ---\n{stdout}\n"
                "--- stderr ---\n{stderr}\n"
            .format(**h)
            )
        err_blob = "\n".join(parts)
        
        data_content = ""
        if self.data_file:
            data_content = self._read_and_trim_data_file(self.data_file)

        fixed_json = self.fix_chain.invoke({
            "simulation_task": state["simulation_task"],
            "err_message": err_blob,
            "pair_info": pair_info,
            "template": state["template"],
            "data_file": self.data_file,
            "data_content": data_content,
        })
        script_dict = self._safe_json_loads(fixed_json)
        
        new_input = script_dict["input_script"]
        old_input = state["input_script"]
        self._diff_panel(old_input, new_input)
                
        with open(os.path.join(self.workspace, "in.lammps"), "w") as f:
            f.write(new_input)
            
        return {
            **state,
            "input_script": new_input,
            "fix_attempts": state.get("fix_attempts", 0) + 1,
        }

    def _build_graph(self):
        g = StateGraph(LammpsState)

        self.add_node(g, self._entry_router)
        self.add_node(g, self._find_potentials)
        self.add_node(g, self._summarize_one)
        self.add_node(g, self._build_summaries)
        self.add_node(g, self._choose)
        self.add_node(g, self._create_user_potential_wrapper)
        self.add_node(g, self._author)
        self.add_node(g, self._run_lammps)
        self.add_node(g, self._fix)

        g.set_entry_point("_entry_router")

        g.add_conditional_edges(
            "_entry_router",
            lambda state: "user_potential" if self.use_user_potential else ( 
                "user_choice" if state.get("chosen_potential") else "agent_choice"
            ),
            {   
                "user_potential": "_create_user_potential_wrapper",
                "user_choice": "_author",
                "agent_choice": "_find_potentials",
            },
        )

        g.add_conditional_edges(
            "_find_potentials",
            self._should_summarize,
            {
                "summarize_one": "_summarize_one",
                "summarize_done": "_build_summaries",
                "done_no_matches": END,
            },
        )

        g.add_conditional_edges(
            "_summarize_one",
            self._should_summarize,
            {
                "summarize_one": "_summarize_one",
                "summarize_done": "_build_summaries",
            },
        )

        g.add_edge("_build_summaries", "_choose")

        g.add_conditional_edges(
            "_choose",
            self._route_after_summarization,
            {
                "continue_author": "_author",
                "Exit": END,
            },
        )

        g.add_edge("_create_user_potential_wrapper", "_author")
        g.add_edge("_author", "_run_lammps")

        g.add_conditional_edges(
            "_run_lammps",
            self._route_run,
            {
                "need_fix": "_fix",
                "done_success": END,
                "done_failed": END,
            },
        )
        g.add_edge("_fix", "_run_lammps")
        return g.compile(checkpointer=self.checkpointer)

    def _invoke(
        self,
        inputs: Mapping[str, Any],
        *,
        summarize: bool | None = None,
        recursion_limit: int = 999_999,
        **_,
    ) -> str:
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )

        if "simulation_task" not in inputs or "elements" not in inputs:
            raise KeyError(
                "'simulation_task' and 'elements' are required arguments"
            )

        if "template" not in inputs:
            inputs = {**inputs, "template": "No template provided."}

        return self._action.invoke(inputs, config)
