help:
    just -l -u

test:
	uv run examples/single_agent_examples/execution_agent/bayesian_optimization.py

test-vowels:
	uv run examples/single_agent_examples/research_agent/ten_vowel_city.py

# Test neutron star example with latest dependencies
neutron-latest:
    uv run --with=. examples/single_agent_examples/arxiv_agent/neutron_star_radius.py

# Test neutron star example with uv.lock dependencies
neutron:
    uv run examples/single_agent_examples/arxiv_agent/neutron_star_radius.py

clean-workspaces:
	rm -rf workspace
	rm -rf workspace_*/
