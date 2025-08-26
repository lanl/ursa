# URSA - The Universal Research and Scientific Agent

<img src="https://github.com/lanl/ursa/raw/main/logos/logo.png" alt="URSA Logo" width="200" height="200">

[![PyPI Version][pypi-version]](https://pypi.org/project/ursa-ai/)
[![PyPI Downloads][monthly-downloads]](https://pypistats.org/packages/ursa-ai)

The flexible agentic workflow for accelerating scientific tasks. 
Composes information flow between agents for planning, code writing and execution, and online research to solve complex problems.

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

# Sandboxing
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
