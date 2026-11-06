# Dependencies added for startup benchmarking and extra detection

| Dependency and disposition | Version | Purpose | Maintainer | Nationality or jurisdiction | License and copyright | Enabled features | Direct/transitive impact | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASV, added to development group | 0.6.6 | Startup measurements and historical reports | airspeed-velocity community | Not applicable (community project); origin not publicly documented | BSD-3-Clause; asv Developers, Michael Droettboom, Space Telescope Science Institute, Pauli Virtanen | Default installation; no extras | Adds ASV, asv-runner 0.3.2, and Pympler 1.1; other dependencies already locked | [Release](https://github.com/airspeed-velocity/asv/releases/tag/v0.6.6), [registry](https://pypi.org/project/asv/0.6.6/), [license](https://github.com/airspeed-velocity/asv/blob/v0.6.6/LICENSE.rst) |
| packaging, promoted to a direct runtime dependency | 25.0 retained in lockfile | Parse extra requirements, markers, and version constraints | Python Packaging Authority community | Not applicable (community project); origin not publicly documented | Apache-2.0 OR BSD-2-Clause; Donald Stufft and individual contributors | Default installation; no extras | Already installed transitively; adds no resolved packages | [Registry](https://pypi.org/project/packaging/25.0/), [license](https://github.com/pypa/packaging/blob/25.0/LICENSE), [copyright](https://github.com/pypa/packaging/blob/25.0/LICENSE.BSD) |

ASV supports Python 3.9 and newer, covering this project's Python 3.11 minimum.
Version 0.6.6 is the current ASV release checked for this change. The existing
compatible packaging version was retained to avoid unrelated dependency upgrades.
All packages are installed through uv with hashes recorded in `uv.lock`; no
upstream source is vendored, so no copied-source notices or license files are
added. Upstream security pages were consulted, but no full advisory audit was
performed; this is not a claim that the dependency graph has no vulnerabilities.
