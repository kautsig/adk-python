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

import asyncio
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator


class LiveRequest(BaseModel):
  """Request send to live agents."""

  model_config = ConfigDict(ser_json_bytes='base64', val_json_bytes='base64')
  """The pydantic model config."""

  content: Optional[types.Content] = None
  """If set, send the content to the model in turn-by-turn mode."""
  blob: Optional[types.Blob] = None
  """If set, send the blob to the model in realtime mode."""
  activity_start: bool = False
  """If set, signal the start of user activity to the model."""
  activity_end: bool = False
  """If set, signal the end of user activity to the model."""
  close: bool = False
  """If set, close the queue. queue.shutdown() is only supported in Python 3.13+."""

  @field_validator('activity_start', 'activity_end', 'close')
  @classmethod
  def validate_single_signal(cls, v, info):
    """Validates that only one signal type is set at a time."""
    if v and info.data:
      # Count how many boolean flags are True
      signal_count = sum([
          info.data.get('activity_start', False),
          info.data.get('activity_end', False), 
          info.data.get('close', False),
          v  # Current field being validated
      ])
      
      # Also check if content or blob is set along with activity signals
      has_content = bool(info.data.get('content')) or bool(info.data.get('blob'))
      is_activity_signal = info.field_name in ['activity_start', 'activity_end']
      
      if is_activity_signal and has_content:
        raise ValueError(
            'Activity signals (activity_start, activity_end) cannot be '
            'combined with content or blob in the same request.'
        )
        
      if signal_count > 1:
        raise ValueError(
            'Only one signal type (activity_start, activity_end, close) '
            'can be set per request.'
        )
    
    return v


class LiveRequestQueue:
  """Queue used to send LiveRequest in a live(bidirectional streaming) way."""

  def __init__(self):
    # Ensure there's an event loop available in this thread
    try:
      asyncio.get_running_loop()
    except RuntimeError:
      # No running loop, create one
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)

    # Now create the queue (it will use the event loop we just ensured exists)
    self._queue = asyncio.Queue()

  def close(self):
    self._queue.put_nowait(LiveRequest(close=True))

  def send_content(self, content: types.Content):
    self._queue.put_nowait(LiveRequest(content=content))

  def send_realtime(self, blob: types.Blob):
    self._queue.put_nowait(LiveRequest(blob=blob))

  def send_activity_start(self):
    """Sends an activity start signal to mark the beginning of user input."""
    self._queue.put_nowait(LiveRequest(activity_start=True))

  def send_activity_end(self):
    """Sends an activity end signal to mark the end of user input."""
    self._queue.put_nowait(LiveRequest(activity_end=True))

  def send(self, req: LiveRequest):
    self._queue.put_nowait(req)

  async def get(self) -> LiveRequest:
    return await self._queue.get()
