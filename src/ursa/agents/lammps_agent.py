import json
import os
import subprocess
from typing import Any, Mapping, Optional, TypedDict

import tiktoken
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

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

    fix_attempts: int


class LammpsAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel = init_chat_model(
            model="openai:gpt-5-mini", max_completion_tokens=200000
        ),
        max_potentials: int = 5,
        max_fix_attempts: int = 10,
        find_potential_only: bool = False,
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
        self.max_potentials = max_potentials
        self.max_fix_attempts = max_fix_attempts
        self.find_potential_only = find_potential_only
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
                "Note that all potential files are in the './' directory.\n"
                "Here is some information about the pair_style and pair_coeff that might be useful in writing the input file: {pair_info}.\n"
                "If a template for the input file is provided, you should adapt it appropriately to meet the task requirements.\n"
                "Template provided (if any): {template}\n"
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

        self._action = self._build_graph()

    @staticmethod
    def _safe_json_loads(s: str) -> dict[str, Any]:
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

    def _entry_router(self, state: LammpsState) -> dict:
        if self.find_potential_only and state.get("chosen_potential"):
            raise Exception(
                "You cannot set find_potential_only=True and also specify your own potential!"
            )

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

        chosen_potential = state["matches"][chosen_index]

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
        print("First attempt at writing LAMMPS input file....")
        state["chosen_potential"].download_files(self.workspace)
        pair_info = state["chosen_potential"].pair_info()
        authored_json = self.author_chain.invoke({
            "simulation_task": state["simulation_task"],
            "pair_info": pair_info,
            "template": state["template"],
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
        pair_info = state["chosen_potential"].pair_info()
        err_blob = state.get("run_stdout")

        fixed_json = self.fix_chain.invoke({
            "simulation_task": state["simulation_task"],
            "input_script": state["input_script"],
            "err_message": err_blob,
            "pair_info": pair_info,
            "template": state["template"],
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

        self.add_node(g, self._entry_router)
        self.add_node(g, self._find_potentials)
        self.add_node(g, self._summarize_one)
        self.add_node(g, self._build_summaries)
        self.add_node(g, self._choose)
        self.add_node(g, self._author)
        self.add_node(g, self._run_lammps)
        self.add_node(g, self._fix)

        g.set_entry_point("_entry_router")

        g.add_conditional_edges(
            "_entry_router",
            lambda state: "user_choice"
            if state.get("chosen_potential")
            else "agent_choice",
            {
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
