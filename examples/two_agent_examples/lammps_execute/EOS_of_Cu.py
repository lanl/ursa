from langchain_openai import ChatOpenAI
from ursa.agents import LammpsAgent

try:
    import atomman as am
except Exception:
    raise ImportError(
        "This example requires the atomman dependency. "
        "This can be installed using 'pip install ursa-ai[lammps]' or, "
        "if working from a local installation, 'pip install -e .[lammps]' ."
    )

model = "gpt-5"

llm = ChatOpenAI(model=model, timeout=None, max_retries=2)

workspace = "./workspace_eos_cu"

wf = LammpsAgent(
    llm=llm,
    max_potentials=2,
    max_fix_attempts=5,
    find_potential_only=False,
    ngpus=-1,  # if -1  will not use gpus. Lammps executable must be installed with kokkos package for gpu usage
    mpi_procs=8,
    workspace=workspace,
    lammps_cmd="lmp_mpi",
    mpirun_cmd="mpirun",
    summarize_results=True,
)

with open("eos_template.txt", "r") as file:
    template = file.read()

simulation_task = (
    "Carry out a LAMMPS simulation of Cu to determine its equation of state."
)

elements = ["Cu"]

db = am.library.Database(remote=True)
matches = db.get_lammps_potentials(pair_style=["eam"], elements=elements)
chosen_potential = matches[-1]

final_lammps_state = wf.invoke(
    simulation_task=simulation_task,
    elements=elements,
    template=template,
    chosen_potential=chosen_potential,
)

if final_lammps_state.get("run_returncode") == 0:
    print("\nLAMMPS Workflow completed successfully. Exiting.....")
