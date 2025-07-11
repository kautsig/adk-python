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

import pytest
from unittest.mock import MagicMock

# Try to import the specific classes we need
try:
    from google.adk.tools.langchain_tool import LangchainTool
    from google.adk.tools.tool_context import ToolContext
    from langchain.tools import tool
    from langchain_core.tools import BaseTool
    imports_successful = True
except ImportError as e:
    imports_successful = False
    import_error = e


pytestmark = pytest.mark.skipif(
    not imports_successful, 
    reason=f"Required imports failed: {import_error if not imports_successful else 'None'}"
)


async def raw_async_function(input: str) -> str:
  """A raw async function."""
  return f"Processed: {input}"


def raw_sync_function(input: str) -> str:
  """A raw sync function."""
  return f"Processed: {input}"


if imports_successful:
  @tool
  def proper_langchain_tool(input: str) -> str:
    """A properly decorated langchain tool."""
    return f"Processed: {input}"


  @tool
  async def proper_async_langchain_tool(input: str) -> str:
    """A properly decorated async langchain tool."""
    return f"Processed: {input}"


  class CustomLangchainTool(BaseTool):
    """Custom langchain tool implementation."""
    
    name = "custom_tool"
    description = "A custom tool"
    
    def _run(self, input: str) -> str:
      return f"Custom processed: {input}"


@pytest.mark.asyncio
async def test_raw_async_function_works():
  """Test that passing a raw async function to LangchainTool works correctly."""
  if not imports_successful:
    pytest.skip("Required imports not available")
  
  langchain_tool = LangchainTool(tool=raw_async_function)
  result = await langchain_tool.run_async(args={"input": "test"}, tool_context=MagicMock())
  assert result == "Processed: test"


@pytest.mark.asyncio
async def test_raw_sync_function_works():
  """Test that passing a raw sync function to LangchainTool works correctly."""
  if not imports_successful:
    pytest.skip("Required imports not available")
  
  langchain_tool = LangchainTool(tool=raw_sync_function)
  result = await langchain_tool.run_async(args={"input": "test"}, tool_context=MagicMock())
  assert result == "Processed: test"


@pytest.mark.asyncio
async def test_proper_langchain_tool_works():
  """Test that properly decorated langchain tools work correctly."""
  if not imports_successful:
    pytest.skip("Required imports not available")
  
  langchain_tool = LangchainTool(tool=proper_langchain_tool)
  result = await langchain_tool.run_async(args={"input": "test"}, tool_context=MagicMock())
  assert result == "Processed: test"


@pytest.mark.asyncio
async def test_proper_async_langchain_tool_works():
  """Test that properly decorated async langchain tools work correctly."""
  if not imports_successful:
    pytest.skip("Required imports not available")
  
  langchain_tool = LangchainTool(tool=proper_async_langchain_tool)
  result = await langchain_tool.run_async(args={"input": "test"}, tool_context=MagicMock())
  assert result == "Processed: test"


@pytest.mark.asyncio
async def test_custom_langchain_tool_works():
  """Test that custom langchain tools work correctly."""
  if not imports_successful:
    pytest.skip("Required imports not available")
  
  custom_tool = CustomLangchainTool()
  langchain_tool = LangchainTool(tool=custom_tool)
  result = await langchain_tool.run_async(args={"input": "test"}, tool_context=MagicMock())
  assert result == "Custom processed: test"


def test_raw_function_validation():
  """Test that raw functions are properly validated during initialization."""
  if not imports_successful:
    pytest.skip("Required imports not available")
  
  # Raw functions should now work fine
  langchain_tool = LangchainTool(tool=raw_async_function)
  assert langchain_tool.name == "raw_async_function"
  assert "raw async function" in langchain_tool.description
  
  langchain_tool = LangchainTool(tool=raw_sync_function)
  assert langchain_tool.name == "raw_sync_function"
  assert "raw sync function" in langchain_tool.description


def test_proper_tool_initialization():
  """Test that proper tools are correctly initialized."""
  if not imports_successful:
    pytest.skip("Required imports not available")
  
  # These should work fine
  langchain_tool = LangchainTool(tool=proper_langchain_tool)
  assert langchain_tool.name == "proper_langchain_tool"
  assert "properly decorated langchain tool" in langchain_tool.description
  
  custom_tool = CustomLangchainTool()
  langchain_tool = LangchainTool(tool=custom_tool)
  assert langchain_tool.name == "custom_tool"
  assert langchain_tool.description == "A custom tool"


def test_invalid_tool_rejection():
  """Test that invalid tools are properly rejected."""
  if not imports_successful:
    pytest.skip("Required imports not available")
  
  # Non-callable objects should be rejected
  with pytest.raises(ValueError, match="Tool must be a Langchain tool"):
    LangchainTool(tool="not_a_function")
  
  with pytest.raises(ValueError, match="Tool must be a Langchain tool"):
    LangchainTool(tool=123)


@pytest.mark.asyncio
async def test_user_bug_scenario():
  """Test the exact scenario from the user's bug report."""
  if not imports_successful:
    pytest.skip("Required imports not available")
  
  # This reproduces the user's exact code
  async def my_custom_tool(input: str) -> str:
    """A custom tool that processes the input string."""
    return 'TOOL CALL SUCCESFULL'

  langchain_custom_tool = my_custom_tool
  adk_custom_tool = LangchainTool(tool=langchain_custom_tool)
  
  # This should now work without throwing TypeError
  result = await adk_custom_tool.run_async(
      args={"input": "testing123"}, 
      tool_context=MagicMock()
  )
  assert result == 'TOOL CALL SUCCESFULL' 