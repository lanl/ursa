# LammpsAgent Documentation

`LammpsAgent` is a class that helps set up and run a LAMMPS simulation workflow. At the highest level, it can:

- discover candidate interatomic potentials from the NIST database for a set of elements,
- summarize and choose a potential for the simulation task at hand,
- author a LAMMPS input script using the chosen potential (and an optional template / data file),
- execute LAMMPS via MPI (CPU or Kokkos GPU),
- iteratively “fix” the input script on failures by using run history until success or a max attempt limit.

The agent writes the outputs into a local `workspace` directory and uses rich console panels to display progress, choices, diffs, and errors.

---

## Dependencies

The main dependency is the [LAMMPS](https://www.lammps.org) code that needs to be installed separately. LAMMPS is a classical molecular dynamics code developed by Sandia National Laboratories. Installation instructions can be found [here](https://docs.lammps.org/Install.html). On macOS and linux systems, the simplest way to install LAMMPS is often via [Conda](https://anaconda.org/channels/conda-forge/packages/lammps/overview), in the same conda environment where `ursa` is installed.  

Two additional dependencies, that are not installed along with `ursa`, are `atomman` and `trafilatura`. These can be installed via `pip install atomman` and `pip install trafilatura`.  

---

## Basic Usage

```python
from ursa.agents import LammpsAgent
from langchain_openai import ChatOpenAI

agent = LammpsAgent(llm = ChatOpenAI(model='gpt-5'))

result = agent.invoke({
    "simulation_task": "Carry out a LAMMPS simulation of Cu to determine its equation of state.",
    "elements": ["Cu"],
    "template": "No template provided."  #Template for the input file
})
```

For more advanced usage see examples here: `ursa/examples/two_agent_examples/lammps_execute/`.

---

## High-level flow

The agent compiles a `StateGraph(LammpsState)` with this logic:

### Entry routing
Chooses one of three paths:

1. **User-provided potential**:
   - This path is chosen when the user provides a specific potential file, along with the `pair_style`/`pair_coeff` information required to generate the input script
   - In this case the autonomous potential search/selection by the agent is skipped
   - The provided potential file is copied to `workspace`

2. **User-chosen potential already in state** (`state["chosen_potential"]` exists):
   - This is similar to the above path, but the user selects a potential from the `atomman` database and initializes the state with this entry before invoking the agent
   - This path also skips the potential search/selection and goes straight to authoring a LAMMPS input script for the user-chosen potential

3. **Agent-selected potential**:
   - Agent queries NIST (via atomman) for potentials matching the requested elements
   - Summarizes NIST's data on each potential (up to `max_potentials`) with regards to the applicability of the potential for the given `simulation task`
   - Ultimately picks one potential 

If a `data_file` is provided to the agent, the entry router attempts to copy it into the workspace.

### Potential search & selection (agent-selected path)
- `_find_potentials`: queries `atomman.library.Database(remote=True)` for potentials matching:
  - `elements` from state
  - supported `pair_styles` list (see `self.pair_styles`)
- `_summarize_one`: for each candidate potential:
  - extracts data on potential from  NIST
  - trims extracted text to a token budget using `tiktoken`
  - summarizes usefulness for the requested `simulation_task`
  - writes summary to `workspace/potential_summaries/potential_<i>.txt`
- `_build_summaries`: builds a combined string of summaries for selection
- `_choose`: the agent selects the final potential to be used and the rationale for choosing it
  - writes rationale to `workspace/potential_summaries/Rationale.txt`
  - stores `chosen_potential` in state

If `find_potential_only=True`, the graph exits after choosing the potential (or finding no matches).

### Author input
- Downloads potential files into `workspace` (only if not user-provided)
- Gets `pair_info` via `chosen_potential.pair_info()`
- Optionally includes:
  - `template` from state for the LAMMPS input script
  - `data_file` (usually for the atomic structure that can be included in the input script) 
- The agent authors the input script: `{ "input_script": "<string>" }`
- Writes `workspace/in.lammps`
- Enforces that logs should go to `./log.lammps` 

### Run LAMMPS 

Runs `<mpirun_cmd>` with `-np <mpi_procs>` in `workspace`:

Allowed options for `<mpirun_cmd>` are `mpirun` and `mpiexec` (see also Parameters section below).

For example, LAMMPS run commands executed by the agent look like: 

- **CPU mode** (default, when `ngpus < 0`):
  - `mpirun -np <mpi_procs> <lammps_cmd> -in in.lammps`

- **GPU/Kokkos mode** (when `ngpus >= 0`):
  - `mpirun -np <mpi_procs> <lammps_cmd> -in in.lammps -k on g <ngpus> -sf kk -pk kokkos neigh half newton on`

Note that the running under GPU mode is preliminary.

The agent captures `stdout`, `stderr`, and `returncode`, and appends an entry to `run_history`.

### Fix loop 
If the run fails:
- formats the entire `run_history` (scripts + stdout/stderr) into an error blob
- the agent produces a new `input_script` 
- prints a unified diff between old and new scripts
- overwrites `workspace/in.lammps`
- increments `fix_attempts`
- reruns LAMMPS

Stops when:
- run succeeds (`returncode == 0`), or
- `fix_attempts >= max_fix_attempts`

---

## State model (`LammpsState`)

The graph state is a `TypedDict` containing (key fields):

- **Inputs / problem definition**
  - `simulation_task: str` — natural language description of what to simulate
  - `elements: list[str]` — chemical symbols used to identify candidate potentials
  - `template: Optional[str]` — optional LAMMPS input template to adapt
  - `chosen_potential: Optional[Any]` — selected potential object (user-chosen)

- **Potential selection internals**
  - `matches: list[Any]` — candidate potentials from atomman
  - `idx: int` — index used for summarization loop
  - `summaries: list[str]` — a brief summary of each potential
  - `full_texts: list[str]` — the data/metadata on the potential from NIST (capped at `max_tokens`)
  - `summaries_combined: str` - a single string with the summaries of all the considered potentials

- **Run artifacts**
  - `input_script: str` — current LAMMPS input text written to `in.lammps`
  - `run_returncode: Optional[int]` - generally, `returncode = 0` indicates a successful simulation run
  - `run_stdout: str` - the stdout from the LAMMPS execution
  - `run_stderr: str` - the stderr from the LAMMPS execution 
  - `run_history: list[dict[str, Any]]` — attempt-by-attempt record
  - `fix_attempts: int` - the number of times the agent has attempted to fix the LAMMPS input script

---

## Parameters

Key parameters you can tune:

### Potential selection
- `potential_files`, `pair_style`, `pair_coeff`: if all provided, the agent uses the user's potential files and skips search
- `max_potentials` (default `5`): max number of candidate potentials to summarize before choosing one
- `find_potential_only` (default `False`): exit after selecting a potential (no input LAMMPS input writing/running)

### Fix loop
- `max_fix_attempts` (default `10`): maximum number of input rewrite attempts after failures

### Data file support
- `data_file` (default `None`): path to a LAMMPS data file; the agent copies it to `workspace`
- `data_max_lines` (default `50`): number of lines from data included in the agent's prompt

### Execution
- `workspace` (default `./workspace`): where `in.lammps`, potentials, and summaries are written
- `mpi_procs` (default `8`): number of mpi processes for LAMMPS run
- `ngpus` (default `-1`): set `>= 0` to enable Kokkos GPU flags
- `lammps_cmd` (default `lmp_mpi`): the name of the LAMMPS executable to launch
- `mpirun_cmd` (default `mpirun`): currently available options are `mpirun` and `mpiexec`. Other options such as `srun` will be added soon

### LLM / context trimming
- `tiktoken_model` (default `gpt-5-mini`): tokenizer model name used to trim fetched potential metadata text
- `max_tokens` (default `200000`): token cap for extracted metadata text

---

## Files and directories created

Inside `workspace/`:

- `in.lammps` — generated/updated input script
- `log.lammps` — expected LAMMPS log output (the LLM is instructed to create it)
- `potential_summaries/`
  - `potential_<i>.txt` — per-potential LLM summaries
  - `Rationale.txt` — rationale for the selected potential
- downloaded potential files (from atomman or copied from user paths)
- copied `data_file` (if provided)

