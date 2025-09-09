ruff := "uvx ruff@0.12.10"

help:
    just -l -u

test:
	uv run examples/single_agent_examples/execution_agent/bayesian_optimization.py

test-vowels:
	uv run examples/single_agent_examples/websearch_agent/ten_vowel_city.py

# Test neutron star example with uv.lock dependencies
neutron *flags:
    uv run examples/single_agent_examples/arxiv_agent/neutron_star_radius.py {{ flags }}

# Test neutron star example with latest dependencies.
neutron-latest:
    just neutron --isolated --resolution=highest

# Test neutron star example with oldest dependencies.
neutron-lowest:
    just neutron --isolated --resolution=lowest-direct

clean-workspaces:
	rm -rf workspace
	rm -rf workspace_*/

precommit:
    uv run pre-commit run --all-files

lint:
    {{ ruff }} check --fix
    {{ ruff }} format

lint-check *flags:
    {{ ruff }} check {{ flags }}

lint-diff:
    just lint-check --diff

lint-stats:
    just lint-check --statistics

lint-watch:
    just lint-check --watch
