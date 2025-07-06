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

from __future__ import annotations

import logging
from typing import Mapping

from typing_extensions import override

from ...agents.base_agent import BaseAgent
from .base_agent_loader import BaseAgentLoader

logger = logging.getLogger("google_adk." + __name__)


class InMemoryAgentLoader(BaseAgentLoader):
  """Simple agent loader that loads agent from in-memory dict."""

  def __init__(self, agents: Mapping[str, BaseAgent]):
    self._agent_cache: dict[str, BaseAgent] = {}
    for app_name, agent in agents.items():
      self._agent_cache[app_name] = agent
    logger.debug("Added %d agents into memory.", len(self._agent_cache))

  @override
  def load_agent(self, agent_name: str) -> BaseAgent:
    """Load an agent from in-memory dict."""
    return self._agent_cache[agent_name]

  @override
  def list_agents(self) -> list[str]:
    """List all agents available in the in-memory dict (sorted alphabetically)."""
    return list(self._agent_cache.keys())
