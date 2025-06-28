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

import pathlib
from unittest import mock

from google.adk.tools.bigquery import chat_tool
import pytest
import yaml


@pytest.mark.parametrize(
    "case_file_path",
    [
        pytest.param("test_data/chat_penguins_highest_mass.yaml"),
    ],
)
@mock.patch("google.adk.tools.bigquery.chat_tool.requests.Session.post")
def test_chat_pipeline_from_file(mock_post, case_file_path):
  """Runs a full integration test for the chat pipeline using data from a specific file."""
  # 1. Construct the full, absolute path to the data file
  full_path = pathlib.Path(__file__).parent / case_file_path

  # 2. Load the test case data from the specified YAML file
  with open(full_path, "r", encoding="utf-8") as f:
    case_data = yaml.safe_load(f)

  # 3. Prepare the mock stream and expected output from the loaded data
  mock_stream_str = case_data["mock_api_stream"]
  fake_stream_lines = [
      line.encode("utf-8") for line in mock_stream_str.splitlines()
  ]
  expected_final_string = case_data["expected_output"].strip()

  # 4. Configure the mock for requests.post
  mock_response = mock.Mock()
  mock_response.iter_lines.return_value = fake_stream_lines
  mock_post.return_value.__enter__.return_value = mock_response

  # 5. Call the function under test
  result = chat_tool._get_stream(  # pylint: disable=protected-access
      url="fake_url",
      chat_payload={},
      headers={},
      max_query_result_rows=50,
  )

  # 6. Assert that the final string matches the expected output from the file
  assert result.strip() == expected_final_string


@mock.patch("google.adk.tools.bigquery.chat_tool._get_stream")
def test_chat_success(mock_get_stream):
  """Tests the success path of chat using decorators."""
  # 1. Configure the behavior of the mocked functions
  mock_get_stream.return_value = "Final formatted string from stream"

  # 2. Create mock inputs for the function call
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
  mock_config = mock.Mock()
  mock_config.max_query_result_rows = 100

  # 3. Call the function under test
  result = chat_tool.chat(
      project_id="test-project",
      user_query_with_context="test query",
      table_references=[],
      credentials=mock_creds,
      config=mock_config,
  )

  # 4. Assert the results are as expected
  assert result["status"] == "SUCCESS"
  assert result["response"] == "Final formatted string from stream"
  mock_get_stream.assert_called_once()


@mock.patch("google.adk.tools.bigquery.chat_tool._get_stream")
def test_chat_handles_exception(mock_get_stream):
  """Tests the exception path of chat using decorators."""
  # 1. Configure one of the mocks to raise an error
  mock_get_stream.side_effect = Exception("API call failed!")

  # 2. Create mock inputs
  mock_creds = mock.Mock()
  mock_creds.token = "fake-token"
  mock_config = mock.Mock()

  # 3. Call the function
  result = chat_tool.chat(
      project_id="test-project",
      user_query_with_context="test query",
      table_references=[],
      credentials=mock_creds,
      config=mock_config,
  )

  # 4. Assert that the error was caught and formatted correctly
  assert result["status"] == "ERROR"
  assert "API call failed!" in result["error_details"]
  mock_get_stream.assert_called_once()


@mock.patch("google.adk.tools.bigquery.chat_tool._append_message")
@mock.patch("google.adk.tools.bigquery.chat_tool._handle_error")
@mock.patch("google.adk.tools.bigquery.chat_tool._handle_data_response")
@mock.patch("google.adk.tools.bigquery.chat_tool._handle_schema_response")
@mock.patch("google.adk.tools.bigquery.chat_tool._handle_text_response")
@mock.patch("google.adk.tools.bigquery.chat_tool.requests.Session.post")
def test_get_stream_routes_to_correct_handlers(
    mock_post,
    mock_handle_text,
    mock_handle_schema,
    mock_handle_data,
    mock_handle_error,
    mock_append,
):
  """Tests that _get_stream correctly parses the stream and calls the

  appropriate handler for each message type.
  """
  # 1. Define mock return values for each handler to track their calls.
  mock_handle_text.return_value = "TEXT_PROCESSED"
  mock_handle_schema.return_value = "SCHEMA_PROCESSED"
  mock_handle_data.return_value = "DATA_PROCESSED"
  mock_handle_error.return_value = "ERROR_PROCESSED"

  # 2. Define the fake streaming data with schema and correct formatting.
  fake_stream_lines = [
      b"[{",
      b'  "systemMessage": { "text": { "parts": ["Hello"] } }',
      b"}",
      b",",
      b"{",
      b'  "systemMessage": { "schema": { "query": { "question": "..." } } }',
      b"}",
      b",",
      b"{",
      b'  "systemMessage": { "chart": { "vegaConfig": {} } }',
      b"}",
      b",",
      b"{",
      b'  "systemMessage": { "data": { "generatedSql": "SELECT 1" } }',
      b"}",
      b",",
      b"{",
      b'  "error": { "code": 404, "message": "Not Found" }',
      b"}",
      b"}]",
  ]

  # 3. Configure the mock for requests.post
  mock_response = mock.Mock()
  mock_response.iter_lines.return_value = fake_stream_lines
  mock_post.return_value.__enter__.return_value = mock_response

  # 4. Call the function under test
  chat_tool._get_stream(  # pylint: disable=protected-access
      url="fake_url", chat_payload={}, headers={}, max_query_result_rows=100
  )

  # 5. Assert that the correct handlers were called with the correct data
  mock_handle_text.assert_called_once_with({"parts": ["Hello"]})
  mock_handle_schema.assert_called_once_with({"query": {"question": "..."}})
  mock_handle_data.assert_called_once_with({"generatedSql": "SELECT 1"}, 100)
  mock_handle_error.assert_called_once_with(
      {"code": 404, "message": "Not Found"}
  )

  # 6. Assert that _append_message was called with the output of the handlers
  #    This verifies the routing logic of the main `if/elif` block.
  calls = mock_append.call_args_list
  assert len(calls) == 4  # text, schema, data, and error handlers were called

  # Check that the return values from the mocked handlers were passed on
  appended_messages = [call.args[1] for call in calls]
  assert "TEXT_PROCESSED" in appended_messages
  assert "SCHEMA_PROCESSED" in appended_messages
  assert "DATA_PROCESSED" in appended_messages
  assert "ERROR_PROCESSED" in appended_messages


@pytest.mark.parametrize(
    "initial_messages, new_message, expected_list",
    [
        pytest.param(
            ["## Thinking", "## Schema Resolved"],
            "## SQL Generated",
            ["## Thinking", "## Schema Resolved", "## SQL Generated"],
            id="append_when_last_message_is_not_data",
        ),
        pytest.param(
            ["## Thinking", "## Data Retrieved\n|...table...|"],
            "## Chart\n...",
            ["## Thinking", "## Chart\n..."],
            id="replace_when_last_message_is_data",
        ),
        pytest.param(
            [],
            "## First Message",
            ["## First Message"],
            id="append_to_an_empty_list",
        ),
        pytest.param(
            ["## Data Retrieved\n|...|"],
            "",
            ["## Data Retrieved\n|...|"],
            id="should_not_append_an_empty_new_message",
        ),
    ],
)
def test_append_message(initial_messages, new_message, expected_list):
  """Tests the logic of replacing the last message if it's a data table."""
  messages_copy = initial_messages.copy()
  chat_tool._append_message(messages_copy, new_message)  # pylint: disable=protected-access
  assert messages_copy == expected_list


@pytest.mark.parametrize(
    "response_dict, expected_output",
    [
        pytest.param(
            {"parts": ["The answer", " is 42."]},
            "Answer: The answer is 42.",
            id="multiple_parts",
        ),
        pytest.param({"parts": ["Hello"]}, "Answer: Hello", id="single_part"),
        pytest.param({}, "Answer: ", id="empty_response"),
    ],
)
def test_handle_text_response(response_dict, expected_output):
  """Tests the text response handler."""
  result = chat_tool._handle_text_response(response_dict)  # pylint: disable=protected-access
  assert result == expected_output


@pytest.mark.parametrize(
    "response_dict, expected_contains",
    [
        pytest.param(
            {"query": {"question": "What is the schema?"}},
            "What is the schema?",
            id="schema_query_path",
        ),
        pytest.param(
            {
                "result": {
                    "datasources": [{
                        "bigqueryTableReference": {
                            "projectId": "p",
                            "datasetId": "d",
                            "tableId": "t",
                        },
                        "schema": {
                            "fields": [{"name": "col1", "type": "STRING"}]
                        },
                    }]
                }
            },
            "## Schema Resolved",
            id="schema_result_path",
        ),
        pytest.param(
            {},
            "",
            id="empty_response_returns_empty_string",
        ),
    ],
)
def test_handle_schema_response(response_dict, expected_contains):
  """Tests different paths of the schema response handler."""
  result = chat_tool._handle_schema_response(response_dict)  # pylint: disable=protected-access
  assert expected_contains in result


@pytest.mark.parametrize(
    "response_dict, expected_contains",
    [
        pytest.param(
            {"generatedSql": "SELECT * FROM my_table;"},
            "## SQL Generated\n```sql\nSELECT * FROM my_table;\n```",
            id="format_generated_sql",
        ),
        pytest.param(
            {
                "result": {
                    "schema": {"fields": [{"name": "id"}, {"name": "name"}]},
                    "data": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
                }
            },
            "| id | name |\n| --- | --- |\n| 1 | A |\n| 2 | B |",
            id="format_data_result_table",
        ),
        pytest.param(
            {
                "result": {
                    "schema": {"fields": [{"name": "id"}]},
                    "data": [{"id": i} for i in range(105)],  # 105 rows
                }
            },
            "... *and 5 more rows*.",
            id="check_data_truncation_message",
        ),
        pytest.param(
            {"invalid_key": "some_value"},
            "",
            id="unhandled_response_returns_empty_string",
        ),
    ],
)
def test_handle_data_response(response_dict, expected_contains):
  """Tests different paths of the data response handler, including truncation."""
  result = chat_tool._handle_data_response(response_dict, 100)  # pylint: disable=protected-access
  assert expected_contains in result


@pytest.mark.parametrize(
    "response_dict, expected_output",
    [
        pytest.param(
            {"code": 404, "message": "Not Found"},
            "## Error\n**Code:** 404\n**Message:** Not Found",
            id="full_error_message",
        ),
        pytest.param(
            {"code": 500},
            "## Error\n**Code:** 500\n**Message:** No message provided.",
            id="error_with_missing_message",
        ),
    ],
)
def test_handle_error(response_dict, expected_output):
  """Tests the error response handler."""
  result = chat_tool._handle_error(response_dict)  # pylint: disable=protected-access
  assert result == expected_output
