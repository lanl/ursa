# https://docs.astral.sh/uv/guides/integration/docker/
# docker buildx build --progress=plain -t ursa .
FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD . /app

WORKDIR /app

RUN apt update && apt install -y build-essential curl ca-certificates

RUN uv sync --locked

# # debugging
# docker run -e "OPENAI_API_KEY"=$OPENAI_API_KEY -it ursa /bin/bash

# # run included example
# docker run -e "OPENAI_API_KEY"=$OPENAI_API_KEY ursa bash -c "uv run python examples/single_agent_examples/arxiv_agent/neutron_star_radius.py"

# # run script from host system
# cp examples/single_agent_examples/arxiv_agent/neutron_star_radius.py myscript.py
# docker run -e "OPENAI_API_KEY"=$OPENAI_API_KEY --mount type=bind,src=$PWD,dst=/mnt/workspace ursa bash -c "uv run python /mnt/workspace/myscript.py"
