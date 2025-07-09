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

from dataclasses import dataclass
from typing import Optional

from .audio_cache_manager import AudioCacheConfig


@dataclass
class LiveFlowConfig:
  """Configuration for live streaming flow behavior."""
  
  # Timing configuration
  request_queue_timeout: float = 0.25
  """Timeout for waiting on live request queue in seconds."""
  
  transfer_agent_delay: float = 1.0
  """Delay before transferring to another agent in seconds."""
  
  task_completion_delay: float = 1.0
  """Delay after task completion signal in seconds."""
  
  # Audio configuration
  audio_cache_config: Optional[AudioCacheConfig] = None
  """Configuration for audio caching behavior."""
  
  # Statistics configuration
  enable_cache_statistics: bool = False
  """Whether to log cache statistics during flow execution."""
  
  def __post_init__(self):
    """Initialize default audio cache config if not provided."""
    if self.audio_cache_config is None:
      self.audio_cache_config = AudioCacheConfig()


class ControlEventConfig:
  """Configuration for handling different control events."""
  
  def __init__(
      self,
      flush_on_interrupted: tuple[bool, bool] = (False, True),
      flush_on_turn_complete: tuple[bool, bool] = (True, True),
      flush_on_generation_complete: tuple[bool, bool] = (False, True)
  ):
    """Initialize control event configuration.
    
    Args:
      flush_on_interrupted: (flush_user_audio, flush_model_audio) when interrupted.
      flush_on_turn_complete: (flush_user_audio, flush_model_audio) when turn complete.
      flush_on_generation_complete: (flush_user_audio, flush_model_audio) when generation complete.
    """
    self.flush_on_interrupted = flush_on_interrupted
    self.flush_on_turn_complete = flush_on_turn_complete
    self.flush_on_generation_complete = flush_on_generation_complete
  
  def get_flush_settings(self, event_type: str) -> tuple[bool, bool]:
    """Get flush settings for a specific event type.
    
    Args:
      event_type: The type of control event ('interrupted', 'turn_complete', 'generation_complete').
      
    Returns:
      Tuple of (flush_user_audio, flush_model_audio).
    """
    config_map = {
        'interrupted': self.flush_on_interrupted,
        'turn_complete': self.flush_on_turn_complete,
        'generation_complete': self.flush_on_generation_complete
    }
    return config_map.get(event_type, (False, False)) 