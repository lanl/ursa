from pathlib import Path

from langchain_openai import ChatOpenAI

from ursa.environments import AgentEloEnvironment


TASK = """
Numerically evaluate

    integral_0^1 exp(-x^2) dx

using Python.

This is an execution task: create and run code in your workspace.

Requirements:

- implement a numerical integration method;
- compare the numerical result with the reference value computed from
  math.erf;
- report the absolute error;
- perform at least one convergence or accuracy check;
- save the executable code and useful numerical results in your workspace.

Keep the calculation small and reproducible.

Your final response should summarize the method, numerical result, error,
validation performed, files produced, and any remaining weakness.
"""


def print_generation(result):
    print(f"\nGeneration {result['generation']}")
    print("-" * 50)

    for match in result["matches"]:
        print(
            f"{match['player_a']} vs {match['player_b']}: "
            f"winner={match['winner']}"
        )
        print(f"  {match['reasoning']}")

    if result["eliminated"]:
        print("Eliminated:", ", ".join(result["eliminated"]))

    if result["children"]:
        print("Children:", ", ".join(result["children"]))

    print("Standings:")
    for row in result["standings"]:
        print(
            f"  {row['rank']}. {row['name']} "
            f"Elo={row['rating']:.2f} "
            f"generation={row['generation']} "
            f"parent={row['parent']}"
        )


def main():
    llm = ChatOpenAI(
        model="gpt-5",
        timeout=None,
        max_retries=2,
    )

    env = AgentEloEnvironment.from_yaml(
        "agent_elo.yaml",
        llm=llm,
        workspace=Path("./workspace_agent_elo"),
        persist_members=True,
    )

    try:
        result = env.invoke({"task": TASK})

        for generation in result["generations"]:
            print_generation(generation)

        print("\nFinal standings")
        print("-" * 50)

        for row in result["standings"]:
            print(
                f"{row['rank']}. {row['name']} "
                f"Elo={row['rating']:.2f}"
            )

        print(
            "\nRestart state:",
            result["environment_state"],
        )

    finally:
        for member in env.members.values():
            close = getattr(member, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main()