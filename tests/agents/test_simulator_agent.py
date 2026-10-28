def test_simulator_agent_import():
    """The experimental simulator must import with supported dependencies."""
    from ursa.agents.base import BaseAgent
    from ursa.experimental.agents.simulator_agent import SimulatorAgent

    assert issubclass(SimulatorAgent, BaseAgent)
