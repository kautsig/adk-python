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

import json
from typing import Any
from typing import Dict
from typing import List

from google.auth.credentials import Credentials
from google.cloud import bigquery
import requests

from . import client
from .config import BigQueryToolConfig


def chat(
    project_id: str,
    user_query_with_context: str,
    table_references: List[Dict[str, str]],
    credentials: Credentials,
    config: BigQueryToolConfig,
) -> Dict[str, str]:
  """Answers questions about structured data in BigQuery tables using natural language.

  This function acts as a client for a "chat-with-your-data" service. It takes a
  user's question (which can include conversational history for context) and
  references to specific BigQuery tables, and sends them to a stateless
  conversational API.

  The API uses a GenAI agent to understand the question, generate and execute
  SQL queries and Python code, and formulate an answer. This function returns a
  detailed, sequential log of this entire process, which includes any generated
  SQL or Python code, the data retrieved, and the final text answer.

  Use this tool to perform data analysis, get insights, or answer complex
  questions about the contents of specific BigQuery tables.

  Args:
      project_id (str): The project that the chat is performed in.
      user_query_with_context (str): The user's question, potentially including
        conversation history and system instructions for context.
      table_references (List[Dict[str, str]]): A list of dictionaries, each
        specifying a BigQuery table to be used as context for the question.
      credentials (Credentials): The credentials to use for the request.
      config (BigQueryToolConfig): The configuration for the tool.

  Returns:
      A dictionary with two keys:
      - 'status': A string indicating the final status (e.g., "SUCCESS").
      - 'response': A list of dictionaries, where each dictionary
        represents a timestamped system message from the API's execution
        process.

  Example:
      A query joining multiple tables, showing the full return structure.
      >>> chat(
      ...     project_id="some-project-id",
      ...     user_query_with_context="Which customer from New York spent the
      most last month? "
      ...                           "Context: The 'customers' table joins with
      the 'orders' table "
      ...                           "on the 'customer_id' column.",
      ...     table_references=[
      ...         {
      ...             "projectId": "my-gcp-project",
      ...             "datasetId": "sales_data",
      ...             "tableId": "customers"
      ...         },
      ...         {
      ...             "projectId": "my-gcp-project",
      ...             "datasetId": "sales_data",
      ...             "tableId": "orders"
      ...         }
      ...     ]
      ... )
      {
        "status": "SUCCESS",
        "response": (
            "## SQL Generated\\n"
            "```sql\\n"
            "SELECT t1.customer_name, SUM(t2.order_total) AS total_spent "
            "FROM `my-gcp-project.sales_data.customers` AS t1 JOIN "
            "`my-gcp-project.sales_data.orders` AS t2 ON t1.customer_id = "
            "t2.customer_id WHERE t1.state = 'NY' AND t2.order_date >= ... "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 1;\\n"
            "```\\n\\n"
            "Answer: The customer who spent the most from New York last "
            "month was Jane Doe."
        )
      }
  """
  try:
    location = "global"
    if not credentials.token:
      error_message = (
          "Error: The provided credentials object does not have a valid access"
          " token.\n\nThis is often because the credentials need to be"
          " refreshed or require specific API scopes. Please ensure the"
          " credentials are prepared correctly before calling this"
          " function.\n\nThere may be other underlying causes as well."
      )
      return {
          "status": "ERROR",
          "error_details": "Chat requires a valid access token.",
      }
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }
    chat_url = f"https://geminidataanalytics.googleapis.com/v1alpha/projects/{project_id}/locations/{location}:chat"

    chat_payload = {
        "project": f"projects/{project_id}",
        "messages": [{"userMessage": {"text": user_query_with_context}}],
        "inlineContext": {
            "datasourceReferences": {
                "bq": {"tableReferences": table_references}
            },
            "options": {"chart": {"image": {"noImage": {}}}},
        },
    }

    resp = _get_stream(
        chat_url, chat_payload, headers, config.max_query_result_rows
    )
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }
  return {"status": "SUCCESS", "response": resp}


def _get_stream(
    url: str,
    chat_payload: Dict[str, Any],
    headers: Dict[str, str],
    max_query_result_rows: int,
) -> str:
  """Sends a JSON request to a streaming API and returns the response as a string."""
  s = requests.Session()

  accumulator = ""
  messages = []

  with s.post(url, json=chat_payload, headers=headers, stream=True) as resp:
    for line in resp.iter_lines():
      if not line:
        continue

      decoded_line = str(line, encoding="utf-8")

      if decoded_line == "[{":
        accumulator = "{"
      elif decoded_line == "}]":
        accumulator += "}"
      elif decoded_line == ",":
        continue
      else:
        accumulator += decoded_line

      if not _is_json(accumulator):
        continue

      data_json = json.loads(accumulator)
      if "systemMessage" not in data_json:
        if "error" in data_json:
          _append_message(messages, _handle_error(data_json["error"]))
        continue

      system_message = data_json["systemMessage"]
      if "text" in system_message:
        _append_message(messages, _handle_text_response(system_message["text"]))
      elif "schema" in system_message:
        _append_message(
            messages,
            _handle_schema_response(system_message["schema"]),
        )
      elif "data" in system_message:
        _append_message(
            messages,
            _handle_data_response(
                system_message["data"], max_query_result_rows
            ),
        )
      accumulator = ""
  return "\n\n".join(messages)


def _is_json(str):
  try:
    json_object = json.loads(str)
  except ValueError as e:
    return False
  return True


def _get_property(data, field_name, default=""):
  """Safely gets a property from a dictionary."""
  return data[field_name] if field_name in data else default


def _format_section_title(text: str) -> str:
  """Formats text as a Markdown H2 title."""
  return f"## {text}"


def _format_bq_table_ref(table_ref: Dict[str, str]) -> str:
  """Formats a BigQuery table reference dictionary into a string."""
  return f"{table_ref['projectId']}.{table_ref['datasetId']}.{table_ref['tableId']}"


def _format_schema_as_markdown(data: Dict[str, Any]) -> str:
  """Converts a schema dictionary to a Markdown table string without using pandas."""
  fields = data.get("fields", [])
  if not fields:
    return "No schema fields found."

  # Define the table headers
  headers = ["Column", "Type", "Description", "Mode"]

  # Create the header and separator lines for the Markdown table
  header_line = f"| {' | '.join(headers)} |"
  separator_line = f"| {' | '.join(['---'] * len(headers))} |"

  # Create a list to hold each data row string
  data_lines = []
  for field in fields:
    # Extract each property in the correct order for a row
    row_values = [
        _get_property(field, "name"),
        _get_property(field, "type"),
        _get_property(field, "description", "-"),
        _get_property(field, "mode"),
    ]
    # Format the row by joining the values with pipes
    data_lines.append(f"| {' | '.join(map(str, row_values))} |")

  # Combine the header, separator, and data lines into the final table string
  return "\n".join([header_line, separator_line] + data_lines)


def _format_datasource_as_markdown(datasource: Dict[str, Any]) -> str:
  """Formats a full datasource object into a string with its name and schema."""
  source_name = _format_bq_table_ref(datasource["bigqueryTableReference"])

  schema_markdown = _format_schema_as_markdown(datasource["schema"])
  return f"**Source:** `{source_name}`\n{schema_markdown}"


def _handle_text_response(resp: Dict[str, Any]) -> str:
  """Joins and returns text parts from a response."""
  parts = resp.get("parts", [])
  return "Answer: " + "".join(parts)


def _handle_schema_response(resp: Dict[str, Any]) -> str:
  """Formats a schema response into a complete string."""
  if "query" in resp:
    return resp["query"].get("question", "")
  elif "result" in resp:
    title = _format_section_title("Schema Resolved")
    datasources = resp["result"].get("datasources", [])
    # Format each datasource and join them with newlines
    formatted_sources = "\n\n".join(
        [_format_datasource_as_markdown(ds) for ds in datasources]
    )
    return f"{title}\nData sources:\n{formatted_sources}"
  return ""


def _handle_data_response(
    resp: Dict[str, Any], max_query_result_rows: int
) -> str:
  """Formats a data response (query, SQL, or result) into a string."""
  if "query" in resp:
    query = resp["query"]
    title = _format_section_title("Retrieval Query")
    return (
        f"{title}\n"
        f"**Query Name:** {query.get('name', 'N/A')}\n"
        f"**Question:** {query.get('question', 'N/A')}"
    )
  elif "generatedSql" in resp:
    title = _format_section_title("SQL Generated")
    sql_code = resp["generatedSql"]
    # Format SQL in a Markdown code block
    return f"{title}\n```sql\n{sql_code}\n```"
  elif "result" in resp:
    title = _format_section_title("Data Retrieved")
    fields = [
        _get_property(field, "name")
        for field in resp["result"]["schema"]["fields"]
    ]
    data_rows = resp["result"]["data"]
    total_rows = len(data_rows)
    header_line = f"| {' | '.join(fields)} |"
    separator_line = f"| {' | '.join(['---'] * len(fields))} |"

    table_lines = [header_line, separator_line]

    for row_dict in data_rows[:max_query_result_rows]:
      row_values = [str(row_dict.get(field, "")) for field in fields]
      table_lines.append(f"| {' | '.join(row_values)} |")

    table_markdown = "\n".join(table_lines)

    if total_rows > max_query_result_rows:
      table_markdown += (
          f"\n\n... *and {total_rows - max_query_result_rows} more rows*."
      )

    return f"{title}\n{table_markdown}"
  return ""


def _handle_error(resp: Dict[str, str]) -> str:
  """Formats an error response into a string."""
  title = _format_section_title("Error")
  code = resp.get("code", "N/A")
  message = resp.get("message", "No message provided.")
  return f"{title}\n**Code:** {code}\n**Message:** {message}"


def _append_message(messages: List[str], new_message: str):
  if new_message:
    if messages and messages[-1].startswith("## Data Retrieved"):
      messages.pop()
    messages.append(new_message)
