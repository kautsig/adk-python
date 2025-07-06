# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents.base_agent import BaseAgent
from google.adk.cli.utils.in_memory_agent_loader import InMemoryAgentLoader
import pytest


class TestInMemoryAgentLoader:
  """Unit tests for InMemoryAgentLoader focusing on interface behavior."""

  def test_agent_caching_returns_same_instance(self):
    """Test that loading the same agent twice returns the same instance."""
    agent_name = "cached_agent"
    agent = BaseAgent(name=agent_name)

    loader = InMemoryAgentLoader({agent_name: agent})
    loaded_agent = loader.load_agent(agent_name)
    same_agent = loader.load_agent(agent_name)

    assert loaded_agent is same_agent

  def test_agent_not_found_error(self):
    """Test that appropriate error is raised when agent is not found."""
    loader = InMemoryAgentLoader({})

    with pytest.raises(KeyError) as exc_info:
      loader.load_agent("nonexistent_agent")

    assert "nonexistent_agent" in str(exc_info.value)

  def test_list_agents_in_order(self):
    """Test that list_agents returns all agents in the loader."""
    some_agent_name = "agent_1"
    some_agent = BaseAgent(name=some_agent_name)
    other_agent_name = "agent_2"
    other_agent = BaseAgent(name=other_agent_name)
    loader = InMemoryAgentLoader({
        some_agent_name: some_agent,
        other_agent_name: other_agent,
    })

    all_agents = loader.list_agents()

    assert all_agents == [some_agent_name, other_agent_name]
