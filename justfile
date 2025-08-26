ruff := "uvx ruff@0.12.10"
lint-options := "--extend-fixable=F401 --ignore='D100,W505'"

help:
    just -l -u

test:
	uv run examples/single_agent_examples/execution_agent/bayesian_optimization.py

test-vowels:
	uv run examples/single_agent_examples/websearch_agent/ten_vowel_city.py

# Test neutron star example with latest dependencies
neutron-latest:
    uv run --with=. examples/single_agent_examples/arxiv_agent/neutron_star_radius.py

# Test neutron star example with uv.lock dependencies
neutron:
    uv run examples/single_agent_examples/arxiv_agent/neutron_star_radius.py

clean-workspaces:
	rm -rf workspace
	rm -rf workspace_*/

lint:
    uv run pre-commit run --all-files

lint-diff:
    {{ ruff }} check {{ lint-options }} --diff

lint-stats:
    {{ ruff }} check {{ lint-options }} --statistics

lint-watch:
    {{ ruff }} check {{ lint-options }} --watch
