import json
import os
import subprocess
from typing import Any, Dict, List, Optional, TypedDict

import atomman as am
import tiktoken
import trafilatura
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .base import BaseAgent


class LammpsState(TypedDict, total=False):
    simulation_task: str
    elements: List[str]

    matches: List[Any]
    db_message: str

    idx: int
    summaries: List[str]
    full_texts: List[str]

    summaries_combined: str
    choice_json: str
    chosen_index: int

    input_script: str
    run_returncode: Optional[int]
    run_stdout: str
    run_stderr: str

    fix_attempts: int


class LammpsAgent(BaseAgent):
    def __init__(
        self,
        llm,
        max_potentials: int = 5,
        max_fix_attempts: int = 10,
        mpi_procs: int = 8,
        workspace: str = "./workspace",
        lammps_cmd: str = "lmp_mpi",
        mpirun_cmd: str = "mpirun",
        tiktoken_model: str = "o3",
        max_tokens: int = 200000,
        **kwargs,
    ):
        self.max_potentials = max_potentials
        self.max_fix_attempts = max_fix_attempts
        self.mpi_procs = mpi_procs
        self.lammps_cmd = lammps_cmd
        self.mpirun_cmd = mpirun_cmd
        self.tiktoken_model = tiktoken_model
        self.max_tokens = max_tokens

        self.pair_styles = [
            "eam",
            "eam/alloy",
            "eam/fs",
            "meam",
            "adp",  # classical, HEA-relevant
            "kim",  # OpenKIM models
            "snap",
            "quip",
            "mlip",
            "pace",
            "nep",  # ML/ACE families (if available)
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
                "Here is metadata about the interatomic potential that will be used: {metadata}.\n"
                "Note that all potential files are in the './' directory.\n"
                "Here is some information about the pair_style and pair_coeff that might be useful in writing the input file: {pair_info}.\n"
                "Ensure that all output data is written only to the './log.lammps' file. Do not create any other output file.\n"
                "To create the log, use only the 'log ./log.lammps' command. Do not use any other command like 'echo' or 'screen'.\n"
                "Return your answer **only** as valid JSON, with no extra text or formatting.\n"
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
                "For this purpose, this input file for LAMMPS was written: {input_script}\n"
                "However, when running the simulation, an error was raised.\n"
                "Here is the full stdout message that includes the error message: {err_message}\n"
                "Your task is to write a new input file that resolves the error.\n"
                "Here is metadata about the interatomic potential that will be used: {metadata}.\n"
                "Note that all potential files are in the './' directory.\n"
                "Here is some information about the pair_style and pair_coeff that might be useful in writing the input file: {pair_info}.\n"
                "Ensure that all output data is written only to the './log.lammps' file. Do not create any other output file.\n"
                "To create the log, use only the 'log ./log.lammps' command. Do not use any other command like 'echo' or 'screen'.\n"
                "Return your answer **only** as valid JSON, with no extra text or formatting.\n"
                "Use this exact schema:\n"
                "{{\n"
                '  "input_script": "<string>"\n'
                "}}\n"
            )
            | self.llm
            | self.str_parser
        )

        self.graph = self._build_graph().compile()

    @staticmethod
    def _safe_json_loads(s: str) -> Dict[str, Any]:
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("`")
            i = s.find("\n")
            if i != -1:
                s = s[i + 1 :].strip()
        return json.loads(s)

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

    def _find_potentials(self, state: LammpsState) -> LammpsState:
        db = am.library.Database(remote=True)
        matches = db.get_lammps_potentials(
            pair_style=self.pair_styles, elements=state["elements"]
        )
        msg_lines = []
        if not list(matches):
            msg_lines.append("No potentials found for this task in NIST.")
        else:
            msg_lines.append("Found these potentials in NIST:")
            for rec in matches:
                msg_lines.append(f"{rec.id}  {rec.pair_style}  {rec.symbols}")
        return {
            **state,
            "matches": list(matches),
            "db_message": "\n".join(msg_lines),
            "idx": 0,
            "summaries": [],
            "full_texts": [],
            "fix_attempts": 0,
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
        print(f"Summarizing potential #{i}")
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
        print("Choosing one potential for this task...")
        choice = self.choose_chain.invoke({
            "summaries_combined": state["summaries_combined"],
            "simulation_task": state["simulation_task"],
        })
        choice_dict = self._safe_json_loads(choice)
        chosen_index = int(choice_dict["Chosen index"])
        print(f"Chosen potential #{chosen_index}")
        print("Rationale for choosing this potential:")
        print(choice_dict["rationale"])
        return {**state, "choice_json": choice, "chosen_index": chosen_index}

    def _author(self, state: LammpsState) -> LammpsState:
        print("First attempt at writing LAMMPS input file....")
        match = state["matches"][state["chosen_index"]]
        match.download_files(self.workspace)
        text = state["full_texts"][state["chosen_index"]]
        pair_info = match.pair_info()
        authored_json = self.author_chain.invoke({
            "simulation_task": state["simulation_task"],
            "metadata": text,
            "pair_info": pair_info,
        })
        script_dict = self._safe_json_loads(authored_json)
        input_script = script_dict["input_script"]
        with open(os.path.join(self.workspace, "in.lammps"), "w") as f:
            f.write(input_script)
        return {**state, "input_script": input_script}

    def _run_lammps(self, state: LammpsState) -> LammpsState:
        print("Running LAMMPS....")
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
        return {
            **state,
            "run_returncode": result.returncode,
            "run_stdout": result.stdout,
            "run_stderr": result.stderr,
        }

    def _route_run(self, state: LammpsState) -> str:
        rc = state.get("run_returncode", 0)
        attempts = state.get("fix_attempts", 0)
        if rc == 0:
            print("LAMMPS run successful! Exiting...")
            return "done_success"
        if attempts < self.max_fix_attempts:
            print("LAMMPS run Failed. Attempting to rewrite input file...")
            return "need_fix"
        print("LAMMPS run Failed and maximum fix attempts reached. Exiting...")
        return "done_failed"

    def _fix(self, state: LammpsState) -> LammpsState:
        match = state["matches"][state["chosen_index"]]
        text = state["full_texts"][state["chosen_index"]]
        pair_info = match.pair_info()
        err_blob = state.get("run_stdout")

        fixed_json = self.fix_chain.invoke({
            "simulation_task": state["simulation_task"],
            "input_script": state["input_script"],
            "err_message": err_blob,
            "metadata": text,
            "pair_info": pair_info,
        })
        script_dict = self._safe_json_loads(fixed_json)
        new_input = script_dict["input_script"]
        with open(os.path.join(self.workspace, "in.lammps"), "w") as f:
            f.write(new_input)
        return {
            **state,
            "input_script": new_input,
            "fix_attempts": state.get("fix_attempts", 0) + 1,
        }

    def _build_graph(self):
        g = StateGraph(LammpsState)

        g.add_node("find_potentials", self._find_potentials)
        g.add_node("summarize_one", self._summarize_one)
        g.add_node("build_summaries", self._build_summaries)
        g.add_node("choose", self._choose)
        g.add_node("author", self._author)
        g.add_node("run_lammps", self._run_lammps)
        g.add_node("fix", self._fix)

        g.set_entry_point("find_potentials")

        g.add_conditional_edges(
            "find_potentials",
            self._should_summarize,
            {
                "summarize_one": "summarize_one",
                "summarize_done": "build_summaries",
                "done_no_matches": END,
            },
        )

        g.add_conditional_edges(
            "summarize_one",
            self._should_summarize,
            {
                "summarize_one": "summarize_one",
                "summarize_done": "build_summaries",
            },
        )

        g.add_edge("build_summaries", "choose")
        g.add_edge("choose", "author")
        g.add_edge("author", "run_lammps")

        g.add_conditional_edges(
            "run_lammps",
            self._route_run,
            {
                "need_fix": "fix",
                "done_success": END,
                "done_failed": END,
            },
        )
        g.add_edge("fix", "run_lammps")
        return g

    def run(self, simulation_task, elements):
        return self.graph.invoke(
            {"simulation_task": simulation_task, "elements": elements},
            {"recursion_limit": 999_999},
        )
