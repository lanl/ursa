---
title: 'URSA: The Universal Research and Scientific Agent'
tags:
  - Python
  - Agentic AI
authors:
  - name: Michael Grosskopf
    orcid: 0000-0002-7838-3609
    affiliation: 1
    
  - name: Nathan Debardeleben
    orcid: 0000-0002-5593-9205
    affiliation: 1

  - name: Russell Bent
    orcid: 0000-0002-7300-151X
    affiliation: 1

  - name: Rahul Somasundaram
    orcid: 0000-0003-0427-3893
    affiliation: 1

  - name: Isaac Michaud
    orcid: 0000-0003-3349-0467
    affiliation: 1

  - name: Arthur Lui
    orcid: 0000-0002-3652-0599
    affiliation: 1

  - name: Alexius Wadell
    orcid: 0000-0001-7589-1870
    affiliation: 1

  - name: Warren D. Graham
    orcid: 0009-0002-7517-3275
    affiliation: 1

  - name: Golo A Wimmer
    orcid: 0000-0002-7871-1748
    affiliation: 1

  - name: Sachin Shivakumar
    orcid: 0000-0003-1353-2834
    affiliation: 1

  - name: Joan Vendrell Gallart
    orcid: 0000-0002-3645-8692
    affiliation: 1

  - name: Harsha Nagarajan
    orcid: 0000-0003-4550-1100
    affiliation: 1

  - name: Jamal Mohd-Yusof
    orcid: 0000-0002-9844-689X
    affiliation: 1

  - name: Adela Habib 
    orcid: 0000-0001-8351-4610
    affiliation: 1

  - name: Earl Lawrence
    orcid: 0000-0002-6473-1887
    affiliation: 1
    
affiliations:
 - name: Los Alamos National Laboratory, Los Alamos, New Mexico, United States
   index: 1
   
date: 01 June 2026

bibliography: paper.bib

---

# Summary

Large language models (LLMs) [@zhao2026survey] are increasingly being integrated within agentic workflows to automate complex tasks, including software engineering, information retrieval, and scientific research. These developments have created new opportunities for AI systems that assist researchers with planning, computation, analysis, and validation. Here, we present URSA (Universal Research and Scientific Agent), an open-source framework for building modular and extensible AI agents for specifically for scientific workflows. URSA enables researchers to compose domain-specific agents by integrating large language models with the appropriate scientific computational tools and knowledge, supporting both general-purpose scientific reasoning and specialized research applications.

# Statement of need

Scientific research increasingly requires AI systems that can integrate reasoning with external software, simulation codes, computational resources, and specialized literature. While existing agentic frameworks provide general-purpose planning and tool use, researchers often need specialized agents tailored to particular research questions.

URSA addresses this need through a collection of reusable core agents that implement common research capabilities such as hypothesis generation, planning, and execution. Furthermore, building on these core agents, URSA includes several specialized agents that demonstrate how the framework can be extended to research applications by integrating scientific computational tools and domain knowledge. These agents support tasks such as molecular dynamics simulations, large-scale numerical simulations, and optimization. Beyond serving as end-user applications, they also act as reference implementations that illustrate how new scientific agents can be developed by composing URSA's core reusable components. Importantly, URSA further provides multiple ways to deploy and orchestrate these agents. Researchers can interact with the framework through a command-line interface, web dashboard, or Python API, while execution environments such as Agent Teams and Agent Symposia enable agents to collaborate or independently evaluate scientific problems.

# State of the field                                                                                                                  

General-purpose agentic systems such as Claude Code [@claude_code] and Codex [@codex] have demonstrated the effectiveness of LLMs for software engineering tasks, including code generation and debugging. These systems excel at programming assistance but are not designed as extensible platforms for developing scientific agents. In parallel, several systems have also been proposed specifically for scientific research, including Sakana AI's AI Scientist [@lu2024ai], Google's Co-Scientist [@gottweis2026accelerating], SciAgents [@ghafarollahi2025sciagents], Agent Laboratory [@schmidgall2025agent], and OpenAI's Deep Research [@openai_deep_research_2025]. While these systems demonstrate the growing potential of AI-assisted research, many are designed as end-to-end research assistants.

In contrast, URSA is an open-source software framework for constructing scientific agents rather than a single predefined assistant. It provides reusable core agents for common research capabilities, example implementations that demonstrate how these components can be extended to specialized applications, multiple user interfaces for deploying workflows, and execution environments such as Agent Symposia that support collaborative and deliberative multi-agent reasoning. Together, these capabilities enable researchers to build, deploy, and orchestrate customized AI-assisted workflows across scientific disciplines.

# Software design

URSA is built on top of LangGraph [@langgraph] and is agnostic to the underlying LLM. Its software architecture is organized around three main concepts: (i) reusable core and domain-specific agents, (ii) multiple user interfaces that expose a common execution engine, and (iii) execution environments that orchestrate collaboration among multiple agents.

## Agent Framework

The separation of the framework into (i) core and (ii) domain-specific agents enables the same architectural components to be reused across different research problems.   

### Core Agents

URSA's core agents include, but are not limited to, the following:

* Planning Agent: This agent decomposes a user-specified scientific problem into a sequence of executable tasks. Implemented as a LangGraph workflow, it consists of three LLM-driven nodes: a planner node that generates an initial research plan, a reviewer node that evaluates and iteratively refines the plan, and a formalizer node that converts the approved plan into a structured JSON representation. This structured output can then be passed to downstream agents, such as the Execution Agent.

* Execution Agent: This agent carries out research tasks specified either in natural language or in the structured JSON format produced by the Planning Agent. It interacts with tools through LangGraph tool calls and through the Model Context Protocol (MCP), allowing virtually any user-provided executable to be incorporated into agent workflows. The Execution Agent also includes built-in tools for code generation and execution, file reading and writing, and system command execution. To improve safety, proposed system commands are screened by an LLM-driven safety node before execution.

<!--
* ArXiv Agent: This agent supports literature-review by searching, downloading, and analyzing papers from the arXiv repository. Given a user query, it uses the arXiv search API to identify relevant papers and constructs a retrieval-augmented generation (RAG) database for each paper using a user-specified embedding model. LLM-backed nodes then generate summaries of the individual papers. When the underlying LLM supports multimodal input, the agent can summarize both textual content and figures from the papers. A final aggregator node synthesizes the individual summaries into a report tailored to the user's query.
-->

### Domain-Specific Agents

* Simulation Agent: The Simulation Agent supports the use of computationally intensive simulation codes on high-performance computing resources. This agent is constructed by orchestrating multiple instances of the core Execution Agent within a LangGraph workflow. One execution agent is responsible for documentation and knowledge acquisition, while another is responsible for simulation setup, execution, debugging, and analysis. The documentation stage gathers information from user-provided manuals, web resources, scientific literature, and RAG-based knowledge bases to construct a task-specific user guide. This guide is then passed to the simulation stage, which uses it to configure and execute simulation campaigns, analyze outputs, and iteratively resolve execution failures. 

* LAMMPS Agent: The LAMMPS Agent [@somasundaram] is a domain-specific agent for atomistic simulations. The agent is capable of autonomously orchestrating the full lifecycle of a molecular dynamics simulation, including interatomic potential selection, generation of LAMMPS input scripts, simulation execution, iterative error recovery, and post-processing of results. The agent can operate in a highly autonomous mode requiring only a natural-language description of the desired simulation, or in an expert mode where users can provide simulation templates, interatomic potentials, and other domain-specific inputs. A key feature of the agent is its ability to iteratively refine simulation inputs in response to execution failures and to leverage other agents within the URSA ecosystem for tasks such as visualization, literature review, and validation against published results.

* Optimization Agent: The Optimization Agent is a self-contained LangGraph workflow for formulating and solving optimization and inverse-design problems from natural-language input. The agent first extracts the optimization problem, converts it into a structured mathematical representation, and optionally discretizes the problem when infinite-dimensional variables or constraints are detected. It then selects an appropriate solver, generates executable optimization code, runs feasibility checks using a dedicated tool, verifies the resulting formulation, and produces a final explanation of the solution. If verification fails, the workflow loops back to the problem-extraction stage, allowing the agent to iteratively revise the formulation.

## User Interfaces

URSA's framework is exposed through multiple interfaces that share the same underlying engine. 

### Command-Line Interface

The command-line interface (CLI) provides an interactive REPL for executing agents and composing scientific workflows. A typical session is launched using

```bash
ursa --config config.yaml
```

where the YAML configuration specifies details such as the desired LLM and other runtime options. Within the REPL, users can invoke individual agents using commands such as

```text
ursa> plan Write a workflow for computing elastic constants using a molecular dynamics simulation.
ursa> execute Execute the plan.
```

### Web Dashboard

URSA also provides a browser-based dashboard that exposes the same agent capabilities through a graphical interface. After installing the optional dashboard dependencies, the server can be started with

```bash
ursa-dashboard
```

or configured explicitly, for example,

```bash
ursa-dashboard --host 127.0.0.1 --port 8080
```

### Python Interface

For integration into scientific software, URSA can be imported directly as a Python package. Agents may be instantiated, configured, and executed within Python scripts or Jupyter notebooks. A typical workflow consists of creating an agent instance and invoking it with a natural-language task:

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from ursa.agents import ExecutionAgent

llm = init_chat_model(model="openai:gpt-5.4")
agent = ExecutionAgent(llm=llm)

result = agent.invoke({
    "messages": [
        HumanMessage(
            content="Write and run a Python script that prints the first 10 prime numbers."
        )
    ],
    "workspace": "./ursa-script-workspace",
})

print(result["messages"][-1].content)
```

The Python interface is particularly useful for integration with existing simulation codes, analysis pipelines, and high-performance computing workflows. 

## Agent Execution Environments

URSA provides execution environments for composing multiple agents into larger scientific workflows. These environments define how agents exchange information and coordinate their execution while reusing the same underlying agent implementations. 

### Agent Teams

The Agent Teams environment coordinates multiple agents through a hierarchical collaboration model. A designated principal investigator (PI) agent receives the user's request, decomposes it into subtasks, and delegates those tasks to specialized agents through focused prompts. URSA's core Execution Agent serves as the default PI because it supports the delegation tools required to coordinate the team, although users may substitute their own tool-enabled agents. URSA's documentation includes several examples of Agent Teams, including workflows in which a PI delegates literature review tasks to a Chat Agent before synthesizing the results into a final response.

### Agent Symposia

The Agent Symposia environment follows a peer-review collaboration model, rather than a hierarchical one. Each participant agent develops its own solution or recommendation, critiques the responses of all other participants, and iteratively refines its reasoning. The final response is then synthesized from the discussion. Different participants may bring different tools, assumptions, models, and reasoning styles.
This environment is particularly well suited for scientific tasks that benefit from multiple independent perspectives, such as research planning, hypothesis generation, or evaluation of competing approaches.

# Research impact statement

URSA is actively used in a growing number of scientific research applications, averaging ~4000 downloads a month on PyPI. While the framework was initially developed at Los Alamos National Laboratory, it is distributed as open-source software and is intended to support contributions and adoption by the broader scientific community.

As one example, \autoref{fig:helios} shows the use of URSA in the design of inertial confinement fusion (ICF) capsules. In this workflow, URSA's planning and execution agents were used to autonomously explore candidate designs and optimize neutron yield. More details of this application can be found in @grosskopf2025ursa.

![Comparison of URSA to Bayesian optimization (BO) for designing a direct-drive ICF design. The plots show the iterative running maximum neutron yield. Figure taken from @grosskopf2025ursa.\label{fig:helios}](helios.png){width=70%}

As a second example, \autoref{fig:lammps} shows atomistic calculations performed by URSA's LAMMPS Agent for a high-entropy alloy [@george2019high]. Such calculations are commonly employed in computational materials science to predict material properties and guide the design of novel materials. More details of the LAMMPS Agent can be found in [@somasundaram].

![The stiffness tensor, i.e., the elastic constants calculated for the high entropy alloy Co-Cr-Fe-Mn-Ni. The atomistic calculation was performed by the LAMMPS agent.\label{fig:lammps}](lammps.png){width=70%}

These examples illustrate the breadth of scientific domains in which URSA can be applied, ranging from optimization-driven design problems to large-scale simulation and materials modeling workflows.

# AI usage disclosure

This work is on AI-driven agentic workflows, so LLMs are invoked at multiple instances within URSA. For example, LLMs were involved in the generation of \autoref{fig:helios} and \autoref{fig:lammps}. For the writing of this manuscript, LLMs were used only for minor polishing, such as grammar and spell checks. All LLM generated text was critically reviewed by the authors for accuracy.

# Acknowledgements

This work was supported by the Laboratory Directed Research and Development program of Los Alamos National Laboratory under project number 20250638DI. This research used resources provided by the Los Alamos National Laboratory Institutional Computing Program, which is supported by the U.S. Department of Energy National Nuclear Security Administration under Contract No. 89233218CNA000001.

# References
