ruff := "uvx ruff@0.12.10"
name := "ursa"
version := `uv run ursa version`
tag := version + "-" + `arch`
sqfs := name + "-" + tag + ".sqfs"

help:
    just -l -u

test:
    uv run pytest -s

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

test-rag-agent:
    uv run pytest -s tests/agents/test_rag_agent

test-bayesopt:
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

clean: clean-workspaces

test-cli:
    uv run ursa run

docker-build:
    docker buildx \
        build \
        --build-arg GIT_TAG=$(uv run ursa version) \
        --progress=plain -t ursa .

docker-shell:
    docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -it ursa bash

# Build wheel and sqfs
wcc: wheel cc

# Build wheel
[private]
wheel:
    [ ! -d "dist" ] || rm -rf dist/*.whl
    uv build .

# Build sqfs, assumes wheel exists
[private]
cc:
    #!/bin/bash
    module load charliecloud
    unset CH_IMAGE_AUTH
    ch-image build -t {{ name }}:{{ tag }} .
    ch-convert {{ name }}:{{ tag }} {{ sqfs }}

shell:
    #!/bin/bash
    module load charliecloud
    unset CH_IMAGE_AUTH
    ch-run -W {{ name }}:{{ tag }} \
            --unset-env='*' \
            --set-env \
            -- bash

pygrep pattern:
    conda run --live-stream -n base watch \
        grep --exclude-dir=__pycache__ --exclude-dir=.venv -r '{{ pattern }}'
