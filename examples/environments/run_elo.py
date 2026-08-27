from pathlib import Path

from langchain_openai import ChatOpenAI

from ursa.environments import AgentEloEnvironment


TASK_GENERATION_1 = r"""
Develop, implement, and test a neural-network variational calculation
for the ground state of the one-dimensional quantum harmonic oscillator.

Use units with

    hbar = 1
    m = 1
    omega = 1

so that

    H = -1/2 d^2/dx^2 + 1/2 x^2

and the exact ground-state energy is

    E0 = 0.5.

This is an EXECUTION task, not a proposal-writing task.

You must use your available tools to create and run code in your workspace.

Requirements:

1. Construct a PyTorch neural-network representation of psi(x).

2. Evaluate the variational energy

       E = <psi|H|psi> / <psi|psi>.

3. Actually train or optimize the neural-network wavefunction.

4. Report the final energy and its absolute error relative to 0.5.

5. Check normalization numerically.

6. Compare the learned wavefunction with the exact ground-state form

       psi_0(x) proportional to exp(-x^2 / 2).

7. Save reproducible artifacts in your workspace, including:

   - executable Python source code;
   - training or optimization history;
   - final numerical results;
   - a plot comparing the learned wavefunction with the exact Gaussian.

8. Perform at least one numerical sanity or convergence check.

9. Correct any bugs encountered during execution.

Before finishing, verify that your workspace contains actual code and
numerical output from a completed run.

Your final response should clearly report:

- the neural-network architecture;
- the numerical representation of the wavefunction;
- how the variational energy was evaluated;
- the optimization method;
- the final energy;
- the absolute error relative to 0.5;
- the numerical normalization;
- the comparison with the exact Gaussian;
- the files produced;
- numerical weaknesses or unresolved issues;
- the most useful next improvement.
"""


TASK_GENERATION_2 = r"""
Continue the neural-network harmonic-oscillator research from your existing
research state and workspace.

This generation is specifically an IMPROVEMENT AND CHALLENGE round.

Do not simply repeat, rerun, or summarize the previous calculation.

You must:

1. Review the calculation, code, numerical results, and conclusions already
   present in your research state and workspace.

2. Identify at least one substantive scientific or numerical limitation,
   weakness, uncertainty, or untested assumption in that existing work.

3. Choose a concrete modification that addresses that limitation.

4. Modify the existing calculation or create additional code as needed.

5. Actually execute the modified calculation.

6. Compare the new result quantitatively with the previous result.

7. Determine whether the modification:

   - improved the calculation;
   - worsened the calculation;
   - or clarified an important question about its reliability.

Possible improvements include, but are not limited to:

- changing the neural-network ansatz;
- removing or weakening a physics prior;
- changing the optimizer or optimization strategy;
- improving quadrature or discretization;
- performing a stronger convergence study;
- independently validating derivatives or kinetic energy;
- improving boundary or asymptotic behavior;
- testing robustness to initialization or hyperparameters;
- introducing Monte Carlo sampling;
- using an independent numerical reference calculation.

Do not make a change merely for novelty. The modification should address
a real limitation of the existing work.

This is an EXECUTION task. You must actually modify or add code and run
the new calculation. A purely descriptive answer is incomplete.

Save all new code and results in your workspace.

Your final response should clearly report:

- what limitation you identified;
- why that limitation matters;
- what substantive modification you made;
- the previous numerical result;
- the new numerical result;
- the quantitative comparison;
- whether the modification improved, worsened, or clarified the calculation;
- what you learned;
- the most useful next step.
"""


def print_member_workspaces(
    env: AgentEloEnvironment,
) -> None:
    print("\nMember workspaces")
    print("-----------------")

    for name, member in env.members.items():
        print(
            f"{name:40s} -> {member.workspace}"
        )


def print_generation(
    generation: int,
    result: dict,
) -> None:
    print()
    print("=" * 80)
    print(f"GENERATION {generation}")
    print("=" * 80)

    print("\nAGENT OUTPUTS")
    print("-------------")

    for name, output in result["outputs"].items():
        print()
        print(f"===== {name} =====")
        print(output)

    print("\nMATCHES")
    print("-------")

    for match in result["matches"]:
        print(
            f"{match['player_a']} "
            f"vs "
            f"{match['player_b']}"
        )

        print(
            f"winner: {match['winner']}"
        )

        print(
            f"loser: {match['loser']}"
        )

        print(
            f"reason: {match['reasoning']}"
        )

        print()

    print("\nELIMINATED")
    print("----------")

    if result["eliminated"]:
        for name in result["eliminated"]:
            print(name)
    else:
        print("none")

    print("\nREPRODUCING PARENTS")
    print("-------------------")

    if result["reproducing_parents"]:
        for name in result["reproducing_parents"]:
            print(name)
    else:
        print("none")

    print("\nCHILDREN")
    print("--------")

    if result["children"]:
        for name in result["children"]:
            print(name)
    else:
        print("none")

    print("\nSTANDINGS")
    print("---------")

    for row in result["standings"]:
        print(
            f"{row['rank']:2d}. "
            f"{row['name']:40s} "
            f"Elo={row['rating']:8.2f} "
            f"generation={row['generation']} "
            f"parent={row['parent']}"
        )


def main():
    llm = ChatOpenAI(
        model="gpt-5",
        timeout=None,
        max_retries=2,
    )

    workspace = Path(
        "./workspace_harmonic_oscillator_elo"
    )

    env = AgentEloEnvironment.from_yaml(
        "agent_elo.yaml",
        llm=llm,
        workspace=workspace,
        persist_members=True,
    )

    print("INITIAL POPULATION")
    print("------------------")

    for row in env.standings():
        print(
            f"{row['name']:20s} "
            f"Elo={row['rating']:.2f}"
        )

    print_member_workspaces(
        env
    )

    try:
        # ==========================================================
        # GENERATION 1
        #
        # Independent baseline solutions.
        # ==========================================================

        result_1 = env.invoke(
            {
                "task": TASK_GENERATION_1,
            }
        )

        print_generation(
            generation=1,
            result=result_1,
        )

        print_member_workspaces(
            env
        )

        population_after_generation_1 = list(
            env.players
        )

        print(
            "\nPopulation entering Generation 2:"
        )

        for name in population_after_generation_1:
            player = env.players[name]

            print(
                f"  {name:40s} "
                f"generation={player.generation} "
                f"parent={player.parent}"
            )

        # ==========================================================
        # GENERATION 2
        #
        # All survivors and descendants now receive an explicit
        # improvement/challenge task.
        # ==========================================================

        result_2 = env.invoke(
            {
                "task": TASK_GENERATION_2,
            }
        )

        print_generation(
            generation=2,
            result=result_2,
        )

        print_member_workspaces(
            env
        )

        # ==========================================================
        # FINAL SUMMARY
        # ==========================================================

        print()
        print("=" * 80)
        print("FINAL POPULATION")
        print("=" * 80)

        for row in env.standings():
            print(
                f"{row['rank']:2d}. "
                f"{row['name']:40s} "
                f"Elo={row['rating']:8.2f} "
                f"generation={row['generation']} "
                f"parent={row['parent']}"
            )

        print()
        print(
            "Environment workspace:",
            workspace.resolve(),
        )

    finally:
        for member in env.members.values():
            close = getattr(
                member,
                "close",
                None,
            )

            if callable(close):
                close()


if __name__ == "__main__":
    main()