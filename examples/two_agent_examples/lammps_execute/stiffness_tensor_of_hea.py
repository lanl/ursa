from langchain_openai import ChatOpenAI
from rich import get_console

from ursa.agents import LammpsAgent

console = get_console()

model = "gpt-5"

llm = ChatOpenAI(model=model, timeout=None, max_retries=2)

workspace = "./workspace_stiffness_tensor"

wf = LammpsAgent(
    llm=llm,
    max_potentials=5,
    max_fix_attempts=15,
    find_potential_only=False,
    mpi_procs=8,
    workspace=workspace,
    lammps_cmd="lmp_mpi",
    mpirun_cmd="mpirun",
    summarize_results=True,
)

with open("elastic_template.txt", "r") as file:
    template = file.read()

simulation_task = (
    "Carry out a LAMMPS simulation of the high entropy alloy Co-Cr-Fe-Mn-Ni "
    "to determine its stiffness tensor."
)

elements = ["Co", "Cr", "Fe", "Mn", "Ni"]

final_lammps_state = wf.invoke(
    simulation_task=simulation_task, elements=elements, template=template
)

if final_lammps_state.get("run_returncode") == 0:
    console.print(
        "\n[green]LAMMPS Workflow completed successfully.[/green] Exiting....."
    )
