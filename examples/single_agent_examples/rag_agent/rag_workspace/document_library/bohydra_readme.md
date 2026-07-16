<table>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;">
      <img src="https://github.com/lanl/bohydra/raw/main/logo/bohydra.png" alt="bohydra logo" width="200" height="200">
    </td>
    <td style="vertical-align: middle;">
      <h3>Many heads, one optimizer: parallel, multi-fidelity Bayesian optimization.</h3>
    </td>
  </tr>
</table>

Multifidelity Bayesian optimization with serial and MPI-enabled (parallel, asynchronous) workflows.


This package provides:
- Gaussian process (GP) emulators for single- and multi-fidelity modeling
- Bayesian optimization (BO) for maximization with Expected Improvement (EI)
- Constrained BO (probabilistic feasibility weighting)
- Multi-fidelity BO (MF-BO) that trades off low/high fidelity evaluations
- Parallel, asynchronous BO using MPI (coordinator/worker pattern)

If you wish to minimize an objective, pass the negated function and maximize instead.

<mark>This package was previously known as **MultifidelityOpt** during development. </mark>

### Contents
- emulators.py: EmuGP (single-fidelity) and EmuMF (multi-fidelity/prior) with stable numerics
- optimizers.py: Opt (BO), OptMF (MF-BO), ConstrainedOpt (constrained BO)
- mpi_optimizers.py: bo_coordinator and bo_worker for MPI-based async BO
- covariances.py: RBF kernel and cross-covariances
- utils.py: numeric helpers and running_max
- examples/: runnable examples for each capability
- templates/: SLURM/MPI helper scripts (optional)

The public API is re-exported from the package root:
- EmuGP, EmuMF, initialize_emulator
- Opt, OptMF, ConstrainedOpt
- running_max
- bo_worker, bo_coordinator (optional, needs mpi4py)

Backwards-compatibility aliases are provided: gp_emu, mf_emu, gp_opt/OptGP, mf_opt.


### Install and setup
Prerequisites: Python 3.9+ recommended.

Standard install from source (with pyproject.toml):
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install .

For development (editable install):
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -e .

Optional extras:
- Include MPI/example dependencies with extras:
    pip install .[mpi]
  or
    pip install -e .[mpi]

Alternative (without install):
- You can also run via PYTHONPATH during development:
    export PYTHONPATH=$PWD:$PYTHONPATH
  Or prepend the repo root on sys.path as shown in the example scripts.

Core dependencies
- numpy
- scipy
- scikit-learn

Additional dependencies for examples and MPI
- pandas (used by mpi_optimizers)
- pyDOE (Latin hypercube initial design in mpi_optimizers)
- mpi4py (optional; required only for MPI examples)

Install the extras you need, e.g.:
    pip install numpy scipy scikit-learn
    pip install pandas pyDOE mpi4py   # if using MPI examples


## Quick start

### Single-fidelity BO (maximize EI)
This example optimizes a 2D test function in the box [−1,1]^2.

    import numpy as np
    import bohydra as bo
    
    # Objective (maximize). For minimization, negate your function.
    def f_hi(x):
        x = np.atleast_2d(x)
        return -x[:, 0] ** 2 + np.exp(-np.sum((x - 0.5) ** 2, axis=1))
    
    rng = np.random.default_rng(0)
    x_lower = np.array([-1.0, -1.0])
    x_upper = np.array([ 1.0,  1.0])
    
    # Initial data
    X0 = rng.uniform(x_lower, x_upper, size=(10, 2))
    y0 = f_hi(X0)
    
    # Build optimizer (GP emulator)
    opt = bo.Opt(
        func=f_hi,
        data_dict={"x": X0, "y": y0, "nugget": 1e-4},
        emulator_type="GP",
        x_lower=x_lower, x_upper=x_upper,
        random_state=0,
    )
    
    # Iterate BO
    for _ in range(10):
        opt.run_opt(iterations=1)
    
    # Best observed
    best = np.argmax(opt.emulator.y)
    print("x_best:", opt.emulator.x[best], "y_best:", opt.emulator.y[best])

### Multi-fidelity modeling as a prior (single-fidelity BO with MF prior)
You can train a low-fidelity GP and use it as a prior for the high-fidelity emulator.

    # Low-fidelity function
    def f_lo(x):
        x = np.atleast_2d(x)
        return -x[:, 0] ** 2
    
    x_low  = rng.uniform(x_lower, x_upper, size=(50, 2))
    y_low  = f_lo(x_low)
    x_high = rng.uniform(x_lower, x_upper, size=(10, 2))
    y_high = f_hi(x_high)
    
    low_emu = bo.initialize_emulator("GP", {"x": x_low, "y": y_low, "nugget": 1e-4})
    opt = bo.Opt(
        func=f_hi,
        data_dict={"x": x_high, "y": y_high, "prior_emu": low_emu, "nugget": 1e-4},
        emulator_type="MFGP",
        x_lower=x_lower, x_upper=x_upper,
    )
    for _ in range(10):
        opt.run_opt(iterations=1)

### True multi-fidelity BO (OptMF)
OptMF recommends which fidelity (low vs high) to evaluate next using IECI-style logic and a cost ratio.

    mf_opt = bo.OptMF(
        func_low=f_lo,
        func_high=f_hi,
        data_dict={"x": x_high, "y": y_high, "x_low": x_low, "y_low": y_low, "nugget": 1e-4},
        emulator_type="MFGP",
        x_lower=x_lower, x_upper=x_upper,
        random_state=0,
    )
    
    x_ref = rng.uniform(x_lower, x_upper, size=(200, 2))  # reference set for IECI
    for _ in range(10):
        mf_opt.run_opt(x_reference=x_ref, iterations=1, cost_ratio=0.5)
        print("last fidelity:", mf_opt.evaluated_fidelities[-1])

### Constrained BO
Pass one or more constraint datasets with threshold definitions. Feasibility is incorporated via a log-probability weight in the acquisition.

    def g_constraint(x):
        x = np.atleast_2d(x)
        return x[:, 0]  # desire x0 <= 0
    
    const_opt = bo.ConstrainedOpt(
        func=f_hi,
        data_dict={"x": X0, "y": y0, "nugget": 1e-4},
        constraint_dicts=[
            {"x": X0, "y": g_constraint(X0), "value": 0.0, "sign": "lessThan", "nugget": 1e-8}
        ],
        emulator_type="GP",
        x_lower=x_lower, x_upper=x_upper,
        constraint_weight=1.0,
    )
    for _ in range(10):
        const_opt.run_opt(iterations=1)

### Parallel, asynchronous BO with MPI
Use the provided coordinator/worker functions.

- The coordinator owns the BO loop and writes progress to CSV:
  - finished_cases.csv: completed evaluations
  - running_cases.csv: currently running candidates (with predicted stats)
- Workers call your objective and return results.

Run from the repository root (requires mpi4py, pandas, pyDOE):

    mpiexec -n 4 python examples/mpi_opt_quadratic.py
    mpiexec -n 4 python examples/mpi_opt_rosenbrock.py
    mpiexec -n 4 python examples/mpi_opt_6humpcamel.py

Coordinator signature:

    from bohydra import bo_coordinator
    bo_coordinator(
        n_total=100,      # total number of runs (including initial design)
        n_init=10,        # initial LHS samples
        n_params=2,       # dimensionality of x
        bounds=[[-5,5], [-5,5]],  # per-dimension [low, high]
        param_names=None, # optional list of column names
    )

Worker signature:

    from bohydra import bo_worker
    
    def get_target(x, job_id=None):
        # x is a 1D numpy array; return a scalar float
        return -np.sum(x**2)
    
    bo_worker(get_target)

Notes:
- The MPI coordinator uses Latin hypercube sampling from pyDOE for the initial design.
- The internal surrogate uses Opt + EmuGP and supports imputation of pending/missing targets via add_impute_data.
- Always launch with at least 2 ranks: 1 coordinator (rank 0) + >=1 worker.


## API overview

### Emulators
- EmuGP(x, y, nugget=1e-6)
  - fit(n_optimization_restarts=11, random_state=None)
  - predict(x_new, return_full_cov=False) -> [mean, sd] or [mean, cov]
  - add_impute_data(x, y): supply imputed points to condition predictions while jobs run
- EmuMF(x, y, prior_emu, nugget=1e-6)
  - Behaves like EmuGP but with a low-fidelity prior emulator; predictions combine prior and residual GP
- initialize_emulator(emulator_type, data_dict)
  - "GP": expects {"x", "y", "nugget"}
  - "MFGP": expects either {"x", "y", "prior_emu", "nugget"} or full low/high data {"x_low","y_low","x","y","nugget"}
  - "MFGPOld": legacy 2-fidelity emulator (not used by Opt/OptMF)

All x are shape (n, d), y are shape (n,), and predictions broadcast over multiple rows.

### Optimizers
- Opt(func, data_dict, x_lower=None, x_upper=None, emulator_type="GP", nugget=1e-4, random_state=None)
  - find_candidate(explore_discount=1.0) -> x_next
  - run_opt(iterations=1)
  - ei(x), logei(x), ieci(x, explore_discount, x_reference)
  - run_opt_ieci(x_reference, iterations=1, subsample_ref=1.0, explore_discount=1.0)
- OptMF(func_low, func_high, data_dict, emulator_type="MFGP", x_lower=None, x_upper=None, nugget=1e-4, random_state=None)
  - run_opt(x_reference, iterations=1, cost_ratio=1.0, subsample_ref=1.0, explore_discount=1.0)
  - Attributes: evaluated_fidelities (list of "low"/"high" for each step)
- ConstrainedOpt(func, data_dict, constraint_dicts, ..., constraint_weight=1.0)
  - constraint_dicts: list of dicts for each constraint emulator with keys:
    - "x", "y": training data for constraint surrogate
    - "value": threshold
    - "sign": "lessThan" or "greaterThan" (feasible when y <= value or y >= value)
    - optional: "nugget"
  - find_candidate(...) and run_opt(...) behave like Opt but include a log-probability of feasibility with weight constraint_weight

### MPI helpers
- bo_coordinator(n_total, n_init, n_params, bounds, param_names=None)
- bo_worker(query_function)


Running the examples
From the repository root:

    python examples/basic_example.py
    python examples/multifidelity_opt_basic.py
    python examples/constrained_opt_basic.py

MPI examples (requires mpi4py, pandas, pyDOE):

    mpiexec -n 4 python examples/mpi_opt_quadratic.py
    mpiexec -n 4 python examples/mpi_opt_rosenbrock.py
    mpiexec -n 4 python examples/mpi_opt_6humpcamel.py

The examples write results to stdout (and for MPI, CSV files as noted).


Tips and troubleshooting
- Maximization convention: all optimizers maximize. To minimize, negate your objective and any constraint functions accordingly.
- Bounds: Opt/OptMF will fall back to the min/max of current data if x_lower/x_upper are not provided, but explicit bounds are recommended.
- Numerical stability: GPs add a nugget (default ~1e-4) and use Cholesky factorizations with jitter fallback. Emulators standardize y and scale x to [0,1].
- Kernel: RBF (squared exponential) with per-dimension positive lengthscales; a light Gaussian prior regularizes log-lengthscales during fit.
- Missing dependencies: if mpi4py is not present, importing bo_worker/bo_coordinator will raise an ImportError with guidance. Install mpi4py and run under mpiexec/mpirun.
- pyDOE is required for LHS in the MPI coordinator: pip install pyDOE
- If you installed via pip using pyproject.toml, you can import bohydra directly. For development without installation, use PYTHONPATH as shown above.


## License for Reference O4998
This program is Open-Source under the BSD-3 License.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
(End of Notice)

