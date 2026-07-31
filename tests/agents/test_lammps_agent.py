import subprocess

import pytest

from ursa.agents.lammps_agent import LammpsAgent


class EventRecorder:
    def __init__(self):
        self.events = []

    def emit(self, message, *, stage, **payload):
        self.events.append({"message": message, "stage": stage, **payload})


@pytest.mark.parametrize(
    ("ngpus", "execution_mode", "expected_gpu_args"),
    [
        (-1, "cpu", False),
        (2, "gpu", True),
    ],
)
def test_run_lammps_emits_terminal_event_for_cpu_and_gpu(
    monkeypatch,
    tmp_path,
    ngpus,
    execution_mode,
    expected_gpu_args,
):
    recorder = EventRecorder()
    agent = object.__new__(LammpsAgent)
    agent.ngpus = ngpus
    agent.mpi_procs = 4
    agent.mpirun_cmd = "mpirun"
    agent.lammps_cmd = "lmp_mpi"
    agent.workspace = tmp_path
    agent.events = lambda config=None: recorder

    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="LAMMPS stdout",
            stderr="LAMMPS error",
        )

    monkeypatch.setattr(
        "ursa.agents.lammps_agent.subprocess.run",
        fake_run,
    )

    result = agent._run_lammps({
        "input_script": "run 100",
        "fix_attempts": 1,
        "run_history": [],
    })

    assert recorder.events[0] == {
        "message": "Running LAMMPS",
        "stage": "run",
        "phase": "start",
        "attempt": 1,
        "execution_mode": execution_mode,
    }
    assert recorder.events[1] == {
        "message": "LAMMPS run failed",
        "stage": "run",
        "phase": "error",
        "attempt": 1,
        "returncode": 1,
        "stdout_chars": len("LAMMPS stdout"),
        "stderr_chars": len("LAMMPS error"),
        "error_output": "LAMMPS error\nLAMMPS stdout",
    }
    assert ("-k" in captured["args"]) is expected_gpu_args
    assert captured["kwargs"]["cwd"] == tmp_path
    assert result["run_returncode"] == 1
    assert result["run_history"][-1]["attempt"] == 1
