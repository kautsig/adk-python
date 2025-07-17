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

"""Unit tests for canonical_xxx fields in LlmAgent."""

from typing import Any
from typing import cast
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.registry import LLMRegistry
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from pydantic import BaseModel
import pytest


async def _create_readonly_context(
    agent: LlmAgent, state: Optional[dict[str, Any]] = None
) -> ReadonlyContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user', state=state
  )
  invocation_context = InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  return ReadonlyContext(invocation_context)


def test_canonical_model_empty():
  agent = LlmAgent(name='test_agent')

  with pytest.raises(ValueError):
    _ = agent.canonical_model


def test_canonical_model_str():
  agent = LlmAgent(name='test_agent', model='gemini-pro')

  assert agent.canonical_model.model == 'gemini-pro'


def test_canonical_model_llm():
  llm = LLMRegistry.new_llm('gemini-pro')
  agent = LlmAgent(name='test_agent', model=llm)

  assert agent.canonical_model == llm


def test_canonical_model_inherit():
  sub_agent = LlmAgent(name='sub_agent')
  parent_agent = LlmAgent(
      name='parent_agent', model='gemini-pro', sub_agents=[sub_agent]
  )

  assert sub_agent.canonical_model == parent_agent.canonical_model


async def test_canonical_instruction_str():
  agent = LlmAgent(name='test_agent', instruction='instruction')
  ctx = await _create_readonly_context(agent)

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_instruction(ctx)
  )
  assert canonical_instruction == 'instruction'
  assert not bypass_state_injection


async def test_canonical_instruction():
  def _instruction_provider(ctx: ReadonlyContext) -> str:
    return f'instruction: {ctx.state["state_var"]}'

  agent = LlmAgent(name='test_agent', instruction=_instruction_provider)
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_instruction(ctx)
  )
  assert canonical_instruction == 'instruction: state_value'
  assert bypass_state_injection


async def test_async_canonical_instruction():
  async def _instruction_provider(ctx: ReadonlyContext) -> str:
    return f'instruction: {ctx.state["state_var"]}'

  agent = LlmAgent(name='test_agent', instruction=_instruction_provider)
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_instruction(ctx)
  )
  assert canonical_instruction == 'instruction: state_value'
  assert bypass_state_injection


async def test_canonical_global_instruction_str():
  agent = LlmAgent(name='test_agent', global_instruction='global instruction')
  ctx = await _create_readonly_context(agent)

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_global_instruction(ctx)
  )
  assert canonical_instruction == 'global instruction'
  assert not bypass_state_injection


async def test_canonical_global_instruction():
  def _global_instruction_provider(ctx: ReadonlyContext) -> str:
    return f'global instruction: {ctx.state["state_var"]}'

  agent = LlmAgent(
      name='test_agent', global_instruction=_global_instruction_provider
  )
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )

  canonical_global_instruction, bypass_state_injection = (
      await agent.canonical_global_instruction(ctx)
  )
  assert canonical_global_instruction == 'global instruction: state_value'
  assert bypass_state_injection


async def test_async_canonical_global_instruction():
  async def _global_instruction_provider(ctx: ReadonlyContext) -> str:
    return f'global instruction: {ctx.state["state_var"]}'

  agent = LlmAgent(
      name='test_agent', global_instruction=_global_instruction_provider
  )
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )
  canonical_global_instruction, bypass_state_injection = (
      await agent.canonical_global_instruction(ctx)
  )
  assert canonical_global_instruction == 'global instruction: state_value'
  assert bypass_state_injection


def test_output_schema_will_disable_transfer(caplog: pytest.LogCaptureFixture):
  with caplog.at_level('WARNING'):

    class Schema(BaseModel):
      pass

    agent = LlmAgent(
        name='test_agent',
        output_schema=Schema,
    )

    # Transfer is automatically disabled
    assert agent.disallow_transfer_to_parent
    assert agent.disallow_transfer_to_peers
    assert (
        'output_schema cannot co-exist with agent transfer configurations.'
        in caplog.text
    )


def test_output_schema_with_sub_agents_will_throw():
  class Schema(BaseModel):
    pass

  sub_agent = LlmAgent(
      name='sub_agent',
  )

  with pytest.raises(ValueError):
    _ = LlmAgent(
        name='test_agent',
        output_schema=Schema,
        sub_agents=[sub_agent],
    )


def test_output_schema_with_tools_will_throw():
  class Schema(BaseModel):
    pass

  def _a_tool():
    pass

  with pytest.raises(ValueError):
    _ = LlmAgent(
        name='test_agent',
        output_schema=Schema,
        tools=[_a_tool],
    )


def test_before_model_callback():
  def _before_model_callback(
      callback_context: CallbackContext,
      llm_request: LlmRequest,
  ) -> None:
    return None

  agent = LlmAgent(
      name='test_agent', before_model_callback=_before_model_callback
  )

  # TODO: add more logic assertions later.
  assert agent.before_model_callback is not None


def test_validate_generate_content_config_thinking_config_throw():
  with pytest.raises(ValueError):
    _ = LlmAgent(
        name='test_agent',
        generate_content_config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig()
        ),
    )


def test_validate_generate_content_config_tools_throw():
  with pytest.raises(ValueError):
    _ = LlmAgent(
        name='test_agent',
        generate_content_config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[])]
        ),
    )


def test_validate_generate_content_config_system_instruction_throw():
  with pytest.raises(ValueError):
    _ = LlmAgent(
        name='test_agent',
        generate_content_config=types.GenerateContentConfig(
            system_instruction='system instruction'
        ),
    )


def test_validate_generate_content_config_response_schema_throw():
  class Schema(BaseModel):
    pass

  with pytest.raises(ValueError):
    _ = LlmAgent(
        name='test_agent',
        generate_content_config=types.GenerateContentConfig(
            response_schema=Schema
        ),
    )


def test_allow_transfer_by_default():
  sub_agent = LlmAgent(name='sub_agent')
  agent = LlmAgent(name='test_agent', sub_agents=[sub_agent])

  assert not agent.disallow_transfer_to_parent
  assert not agent.disallow_transfer_to_peers


@pytest.mark.asyncio
async def test_canonical_tools_without_toolset():
  """Test canonical_tools returns only tools when with_toolset=False (default)."""

  def _test_function():
    pass

  class _TestToolset(BaseToolset):

    async def get_tools(self, readonly_context=None):
      return [FunctionTool(func=_test_function)]

    async def close(self):
      pass

  toolset = _TestToolset()
  agent = LlmAgent(name='test_agent', tools=[toolset])
  ctx = await _create_readonly_context(agent)

  tools = await agent.canonical_tools(ctx)

  # Should only return tools, not the toolset itself
  assert len(tools) == 1
  assert all(isinstance(tool, BaseTool) for tool in tools)
  assert isinstance(tools[0], FunctionTool)


@pytest.mark.asyncio
async def test_canonical_tools_with_toolset():
  """Test canonical_tools returns tools and toolsets when with_toolset=True."""

  def _test_function():
    pass

  class _TestToolset(BaseToolset):

    async def get_tools(self, readonly_context=None):
      return [FunctionTool(func=_test_function)]

    async def close(self):
      pass

  toolset = _TestToolset()
  agent = LlmAgent(name='test_agent', tools=[toolset])
  ctx = await _create_readonly_context(agent)

  tools = await agent.canonical_tools(ctx, with_toolset=True)

  # Should return both the toolset and its tools
  assert len(tools) == 2
  assert any(isinstance(item, BaseToolset) for item in tools)
  assert any(isinstance(item, BaseTool) for item in tools)


@pytest.mark.asyncio
async def test_canonical_tools_mixed_tools_and_toolsets():
  """Test canonical_tools with mixed tools, functions, and toolsets."""

  def _test_function():
    pass

  class _TestTool(BaseTool):

    def __init__(self):
      super().__init__(name='test_tool', description='Test tool')

    async def call(self, **kwargs):
      return 'test'

  class _TestToolset(BaseToolset):

    async def get_tools(self, readonly_context=None):
      return [_TestTool()]

    async def close(self):
      pass

  direct_tool = _TestTool()
  toolset = _TestToolset()

  agent = LlmAgent(
      name='test_agent', tools=[direct_tool, _test_function, toolset]
  )
  ctx = await _create_readonly_context(agent)

  # Test without toolset
  tools_only = await agent.canonical_tools(ctx, with_toolset=False)
  assert len(tools_only) == 3  # direct_tool + function_tool + toolset's tool
  assert all(isinstance(tool, BaseTool) for tool in tools_only)

  # Test with toolset
  tools_with_toolset = await agent.canonical_tools(ctx, with_toolset=True)
  assert (
      len(tools_with_toolset) == 4
  )  # direct_tool + function_tool + toolset + toolset's tool
  assert (
      sum(1 for item in tools_with_toolset if isinstance(item, BaseToolset))
      == 1
  )
  assert (
      sum(1 for item in tools_with_toolset if isinstance(item, BaseTool)) == 3
  )


@pytest.mark.asyncio
async def test_convert_tool_union_to_tools():
  """Test the _convert_tool_union_to_tools function with different tool types."""
  from google.adk.agents.llm_agent import _convert_tool_union_to_tools

  def _test_function():
    pass

  class _TestTool(BaseTool):

    def __init__(self):
      super().__init__(name='test_tool', description='Test tool')

    async def call(self, **kwargs):
      return 'test'

  class _TestToolset(BaseToolset):

    async def get_tools(self, readonly_context=None):
      return [_TestTool()]

    async def close(self):
      pass

  agent = LlmAgent(name='test_agent')
  ctx = await _create_readonly_context(agent)

  # Test with BaseTool
  tool = _TestTool()
  result = await _convert_tool_union_to_tools(tool, ctx, with_toolset=False)
  assert len(result) == 1
  assert isinstance(result[0], BaseTool)

  # Test with function
  result = await _convert_tool_union_to_tools(
      _test_function, ctx, with_toolset=False
  )
  assert len(result) == 1
  assert isinstance(result[0], FunctionTool)

  # Test with toolset without including toolset
  toolset = _TestToolset()
  result = await _convert_tool_union_to_tools(toolset, ctx, with_toolset=False)
  assert len(result) == 1
  assert isinstance(result[0], BaseTool)

  # Test with toolset including toolset
  result = await _convert_tool_union_to_tools(toolset, ctx, with_toolset=True)
  assert len(result) == 2
  assert any(isinstance(item, BaseToolset) for item in result)
  assert any(isinstance(item, BaseTool) for item in result)
