# URSA - The Universal Research and Scientific Agent

<img src="https://github.com/lanl/ursa/raw/main/logos/logo.png" alt="URSA Logo" width="200" height="200">

[![PyPI Version][pypi-version]](https://pypi.org/project/ursa-ai/)
[![PyPI Downloads][monthly-downloads]](https://pypistats.org/packages/ursa-ai)

The flexible agentic workflow for accelerating scientific tasks. 
Composes information flow between agents for planning, code writing and execution, and online research to solve complex problems.

The original arxiv paper is [here](https://arxiv.org/abs/2506.22653).

## Installation
You can install `ursa` via `pip` or `uv`.

**pip**
```bash
pip install ursa-ai
```

**uv**
```bash
uv add ursa-ai
```

## How to use this code
Better documentation will be incoming, but for now there are examples in the examples folder that should give
a decent idea for how to set up some basic problems. They also should give some idea of how to pass results from
one agent to another. I will look to add things with multi-agent graphs, etc. in the future. 

Documentation for each URSA agent:
- [Planning Agent](docs/planning_agent.md)
- [Execution Agent](docs/execution_agent.md)
- [ArXiv Agent](docs/arxiv_agent.md)
- [Web Search Agent](docs/web_search_agent.md)
- [Hypothesizer Agent](docs/hypothesizer_agent.md)

Documentation for combining agents:
- [ArXiv -> Execution for Materials](docs/combining_arxiv_and_execution.md)
- [ArXiv -> Execution for Neutron Star Properties](docs/combining_arxiv_and_execution_neutronStar.md)


## Command line usage

You can install `ursa` as a command line app with `pip install`; or with `uv` via

```bash
uv tool install ursa-ai
```

To use the command line app, run

```
ursa run
```

This will start a REPL in your terminal.

```
  __  ________________ _
 / / / / ___/ ___/ __ `/
/ /_/ / /  (__  ) /_/ /
\__,_/_/  /____/\__,_/

For help, type: ? or help. Exit with Ctrl+d.
ursa>
```

Within the REPL, you can get help by typing `?` or `help`. 

You can chat with an LLM by simply typing into the terminal.

```
ursa> How are you?
Thanks for asking! I’m doing well. How are you today? What can I help you with?
```

You can run various agents by typing the name of the agent. For example,

```
ursa> plan
Enter your prompt for Planning Agent: Write a python script to do linear regression using only numpy.
```

If you run subsequent agents, the last output will be appended to the prompt for the next agent.

So, to run the Planning Agent followed by the Execution Agent:
```
ursa> plan
Enter your prompt for Planning Agent: Write a python script to do linear regression using only numpy.

...

ursa> execute
Enter your prompt for Execution Agent: Execute the plan.
```

You can get a list of available command line options via
```
ursa run --help
```

## MCP serving

You can install `ursa` as a command line app via `pip` or `uv`:

**pip**

```shell
pip install 'ursa-ai[mcp]'
```

**uv**

```shell
uv tool install 'ursa-ai[mcp]'
```

To start hosting URSA as a local MCP server, run

```shell
ursa serve
```

This will start an MCP server on localhost (127.0.0.1) on port 8000.

You can test the server using curl from another terminal:

```shell
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"agent": "execute", "query": "Plot the first 1000 prime numbers with matplotlib"}' \
    http://localhost:8000/run
```

The resulting code is written in the `ursa_mcp` subfolder of the serving
location. The curl query will get the final summary of what the agent carried
out. 

When served locally, URSA can then be set up as an MCP tool that can be couple
to other agentic workflows. The set of agents is the same as the cli (execute,
plan, arxiv, web, recall, chat)

You can get a list of available command line options via
```
ursa serve --help
```

For API documentation, after `ursa serve`, go to your browser at

```
http://localhost:8000/docs
```

## Sandboxing
The Execution Agent is allowed to run system commands and write/run code. Being able to execute arbitrary system commands or write
and execute code has the potential to cause problems like:
- Damage code or data on the computer
- Damage the computer
- Transmit your local data

The Web Search Agent scrapes data from urls, so has the potential to attempt to pull information from questionable sources.

Some suggestions for sandboxing the agent:
- Creating a specific environment such that limits URSA's access to only what you want. Examples:
    - Creating/using a virtual machine that is sandboxed from the rest of your machine
    - Creating a new account on your machine specifically for URSA 
- Creating a network blacklist/whitelist to ensure that network commands and webscraping are contained to safe sources

You have a duty for ensuring that you use URSA responsibly.

## Container image

To enable limited sandboxing insofar as containerization does this, you can run
the following commands:

### Docker

```shell
# Pull the image
docker pull ghcr.io/lanl/ursa

# Run included example
docker run -e "OPENAI_API_KEY"=$OPENAI_API_KEY ursa \
    bash -c "uv run python examples/single_agent_examples/execution_agnet/integer_sum.py"

# Run script from host system
mkdir -p scripts
echo "import ursa; print('Hello from ursa')" > scripts/my_script.py
docker run -e "OPENAI_API_KEY"=$OPENAI_API_KEY \
    --mount type=bind,src=$PWD/scripts,dst=/mnt/workspace \
    ursa \
    bash -c "uv run /mnt/workspace/my_script.py"
```

### Charliecloud

[Charliecloud](https://charliecloud.io/) is a rootless alternative to docker
that is sometimes preferred on HPC. The following commands replicate the
behaviors above for docker.

```shell
# Pull the image
ch-image pull ghcr.io/lanl/ursa

# Convert image to sqfs, for use on another system
ch-convert ursa ursa.sqfs

# Run included example (if wanted, replace ursa with /path/to/ursa.sqfs)
ch-run -W ursa \
    --unset-env="*" \
    --set-env \
    --set-env="OPENAI_API_KEY"=$OPENAI_API_KEY \
    --cd /app \
    -- bash -c \
    "uv run examples/single_agent_examples/execution_agnet/integer_sum.py"

# Run script from host system (if wanted, replace ursa with /path/to/ursa.sqfs)
mkdir -p scripts
echo "import ursa; print('Hello from ursa')" > scripts/my_script.py
ch-run -W ursa \
    --unset-env="*" \
    --set-env \
    --set-env="OPENAI_API_KEY"=$OPENAI_API_KEY \
    --bind ${PWD}/scripts:/mnt/workspace \
    --cd /app \
    -- bash -c \
    "uv run /mnt/workspace/integer_sum.py"
```

## Development Dependencies

* [`uv`](https://docs.astral.sh/uv/)
    * `uv` is an extremely fast python package and project manager, written in Rust.
      Follow installation instructions
      [here](https://docs.astral.sh/uv/getting-started/installation/)

* [`ruff`](https://docs.astral.sh/ruff/)
    * An extremely fast Python linter and code formatter, written in Rust.
    * After installing `uv`, you can install just ruff `uv tool install ruff`

* [`just`](https://github.com/casey/just)
    * A modern way to save and run project-specific commands
    * After installing `uv`, you can install just with `uv tool install rust-just`

## Development Team

URSA has been developed at Los Alamos National Laboratory as part of the ArtIMis project.

<img src="https://github.com/lanl/ursa/raw/main/logos/artimis.png" alt="ArtIMis Logo" width="200" height="200">

### Notice of Copyright Assertion (O4958):
*This program is Open-Source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:*
- *Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.*
- *Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.*
- *Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.*

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

[pypi-version]: https://img.shields.io/pypi/v/ursa-ai?style=flat-square&label=PyPI
[monthly-downloads]: https://img.shields.io/pypi/dm/ursa-ai?style=flat-square&label=Downloads&color=blue
