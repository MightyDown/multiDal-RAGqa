import pytest

from src.multidal.agents.base import BaseAgent


class TestBaseAgent:
    def test_construction(self):
        ba = BaseAgent(name="TestAgent", instructions="Be helpful.")
        assert ba.name == "TestAgent"

    def test_agent_property(self):
        ba = BaseAgent(name="T", instructions="I")
        from agents import Agent
        assert isinstance(ba.agent, Agent)

    def test_name_passed_to_sdk_agent(self):
        ba = BaseAgent(name="MyAgent", instructions="Do stuff.")
        assert ba.agent.name == "MyAgent"

    def test_instructions_passed_to_sdk_agent(self):
        ba = BaseAgent(name="A", instructions="Always greet the user.")
        assert ba.agent.instructions == "Always greet the user."
