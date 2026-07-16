# URSA - The Universal Research and Scientific Agent

<img src="https://github.com/lanl/ursa/raw/main/logos/logo.png" alt="URSA Logo" width="200" height="200">

[![PyPI Version][pypi-version]](https://pypi.org/project/ursa-ai/)
[![PyPI Downloads][monthly-downloads]](https://pypistats.org/packages/ursa-ai)

The flexible agentic workflow for accelerating scientific tasks.
Composes information flow between agents for planning, code writing and execution, and online research to solve complex problems.

The original ArXiv paper is [here](https://arxiv.org/abs/2506.22653).

## Documentation

Detailed documenation including:
- Installation
- Getting Started Guides
- Configuration
- ... and more

are located at: [URSA Documentation](https://lanl.github.io/ursa)


## Installation

URSA is published on PyPI as [`ursa-ai`](https://pypi.org/project/ursa-ai/) and supports Python 3.11 or newer.
You can install it with `uv` or `pip`; `uv` is recommended for new projects.

## Documentation and examples

The MkDocs documentation in `docs/` is organized around installation, getting started, configuration, persistence, agents, best practices, and reference material. The `examples/` folder demonstrates practical workflows and ways to pass results from one agent to another.


## Command Line Usage

You can install `ursa` as a command line app with `pip install`; or with [`uv`](https://docs.astral.sh/uv/) via

```bash
uv tool install ursa-ai
```

A reusable YAML configuration file is the preferred way to select endpoints and runtime settings. For example:

```yaml
llm_model:
  model: openai:gpt-5.2
  api_key_env: OPENAI_API_KEY
workspace: .
```

Then start the command line app with:

```
ursa --config config.yaml
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
plan: Write a python script to do linear regression using only numpy.
```

Or by prepending the agent name to the query:

```shell
ursa> plan Write a python script to do linear regression using only numpy.
```

If you run subsequent agents, the last output will be appended to the prompt for the next agent.

So, to run the Planning Agent followed by the Execution Agent:
```
ursa> plan
plan: Write a python script to do linear regression using only numpy.

...

ursa> execute
execute: Execute the plan.
```

You can get a list of available command line options via
```
ursa --help
```

### Web Dashboard

The URSA web interface can be launched with:
```
ursa-dashboard
```

or with 
```
ursa-dashboard --host 127.0.0.1 --port 8080
```

This requires installing with the optional `[dashboard]` dependencies.

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
