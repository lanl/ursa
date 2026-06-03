# Installation
You can install `ursa` via `pip` or [`uv`](https://docs.astral.sh/uv/). Installing `ursa` in a clean
environment with python 3.11-3.12 may be necessary. (Some `ursa` dependencies
currently do not support `python>=3.13`.)

**uv**

```sh
uv init -p 3.12  # or 3.11
uv add ursa-ai
```

**pip**

```sh
pip install ursa-ai
```

**conda with pip install**

```sh
conda create -y -n ursa-env python=3.12  # or 3.11
conda run --live-stream -n ursa-env python -m pip install ursa-ai
```
