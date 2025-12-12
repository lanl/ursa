from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ursa.agents import ExecutionAgent, LammpsAgent

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
    lammps_cmd="/usr/projects/artimis/ahabib/envs/p311g-alf-packages/lammps/build-ml4chem/lmp",
    mpirun_cmd="mpirun",
)

with open("elastic_template.txt", "r") as file:
    template = file.read()

simulation_task = "Carry out a LAMMPS simulation of the high entropy alloy Co-Cr-Fe-Mn-Ni to determine its stiffness tensor."

elements = ["Co", "Cr", "Fe", "Mn", "Ni"]

final_lammps_state = wf.invoke(
    simulation_task=simulation_task, elements=elements, template=template
)

if final_lammps_state.get("run_returncode") == 0:
    print("\nNow handing things off to execution agent.....")

    executor = ExecutionAgent(llm=llm)
    exe_plan = f"""
    You are part of a larger scientific workflow whose purpose is to accomplish this task: {simulation_task}
    
    A LAMMPS simulation has been done and the output is located in the file 'log.lammps'.
    
    Summarize the contents of this file in a markdown document. Include a plot, if relevent.
    """

    final_results = executor.invoke({
        "messages": [HumanMessage(content=exe_plan)],
        "workspace": workspace,
    })

    for x in final_results["messages"]:
        print(x.content)
