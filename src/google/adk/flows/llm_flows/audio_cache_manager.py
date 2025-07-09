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
import time
from typing import TYPE_CHECKING

from google.genai import types

from ...agents.invocation_context import AudioCacheEntry
from ...events.event import Event

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext

logger = logging.getLogger('google_adk.' + __name__)


class AudioCacheManager:
  """Manages audio caching and flushing for live streaming flows."""

  def __init__(self, config: AudioCacheConfig | None = None):
    """Initialize the audio cache manager.
    
    Args:
      config: Configuration for audio caching behavior.
    """
    self.config = config or AudioCacheConfig()

  def cache_input_audio(
      self, 
      invocation_context: InvocationContext, 
      audio_blob: types.Blob
  ) -> None:
    """Cache incoming user audio data.
    
    Args:
      invocation_context: The current invocation context.
      audio_blob: The audio data to cache.
    """
    if not invocation_context.input_audio_cache:
      invocation_context.input_audio_cache = []
    
    audio_entry = AudioCacheEntry(
        role='user',
        data=audio_blob,
        timestamp=time.time()
    )
    invocation_context.input_audio_cache.append(audio_entry)
    
    logger.debug(
        'Cached input audio chunk: %d bytes, cache size: %d',
        len(audio_blob.data),
        len(invocation_context.input_audio_cache)
    )

  def cache_output_audio(
      self, 
      invocation_context: InvocationContext, 
      audio_blob: types.Blob
  ) -> None:
    """Cache outgoing model audio data.
    
    Args:
      invocation_context: The current invocation context.
      audio_blob: The audio data to cache.
    """
    if not invocation_context.output_audio_cache:
      invocation_context.output_audio_cache = []
    
    audio_entry = AudioCacheEntry(
        role='model',
        data=audio_blob,
        timestamp=time.time()
    )
    invocation_context.output_audio_cache.append(audio_entry)
    
    logger.debug(
        'Cached output audio chunk: %d bytes, cache size: %d',
        len(audio_blob.data),
        len(invocation_context.output_audio_cache)
    )

  async def flush_caches(
      self, 
      invocation_context: InvocationContext,
      flush_user_audio: bool = True,
      flush_model_audio: bool = True
  ) -> None:
    """Flush audio caches to session and artifact services.
    
    Args:
      invocation_context: The invocation context containing audio caches.
      flush_user_audio: Whether to flush the input (user) audio cache.
      flush_model_audio: Whether to flush the output (model) audio cache.
    """
    if flush_user_audio and invocation_context.input_audio_cache:
      success = await self._flush_cache_to_services(
          invocation_context, 
          invocation_context.input_audio_cache, 
          'input_audio'
      )
      if success:
        invocation_context.input_audio_cache = []
        logger.debug('Flushed input audio cache')
    
    if flush_model_audio and invocation_context.output_audio_cache:
      success = await self._flush_cache_to_services(
          invocation_context,
          invocation_context.output_audio_cache,
          'output_audio'
      )
      if success:
        invocation_context.output_audio_cache = []
        logger.debug('Flushed output audio cache')

  async def _flush_cache_to_services(
      self, 
      invocation_context: InvocationContext,
      audio_cache: list[AudioCacheEntry],
      cache_type: str
  ) -> bool:
    """Flush a specific audio cache to session and artifact services.
    
    Args:
      invocation_context: The invocation context.
      audio_cache: The audio cache to flush.
      cache_type: Type identifier for the cache ('input_audio' or 'output_audio').
      
    Returns:
      True if the cache was successfully flushed, False otherwise.
    """
    if not invocation_context.artifact_service or not audio_cache:
      logger.debug('Skipping cache flush: no artifact service or empty cache')
      return False
    
    try:
      # Combine audio chunks into a single file
      combined_audio_data = b''
      mime_type = audio_cache[0].data.mime_type if audio_cache else 'audio/pcm'
      
      for entry in audio_cache:
        combined_audio_data += entry.data.data
      
      # Generate filename with timestamp from first audio chunk (when recording started)
      timestamp = int(audio_cache[0].timestamp * 1000)  # milliseconds
      filename = f"{cache_type}_{timestamp}.{mime_type.split('/')[-1]}"
      
      # Save to artifact service
      combined_audio_part = types.Part(
          inline_data=types.Blob(
              data=combined_audio_data,
              mime_type=mime_type
          )
      )
      
      revision_id = await invocation_context.artifact_service.save_artifact(
          app_name=invocation_context.app_name,
          user_id=invocation_context.user_id,
          session_id=invocation_context.session.id,
          filename=filename,
          artifact=combined_audio_part
      )
      
      # Create artifact reference for session service
      artifact_ref = f"artifact://{invocation_context.app_name}/{invocation_context.user_id}/{invocation_context.session.id}/{filename}#{revision_id}"
      
      # Create event with file data reference to add to session
      audio_event = Event(
          id=Event.new_id(),
          invocation_id=invocation_context.invocation_id,
          author=audio_cache[0].role,
          content=types.Content(
              role=audio_cache[0].role,
              parts=[
                  types.Part(
                      file_data=types.FileData(
                          file_uri=artifact_ref,
                          mime_type=mime_type
                      )
                  )
              ]
          ),
          timestamp=audio_cache[0].timestamp
      )
      
      # Add to session
      await invocation_context.session_service.append_event(
          invocation_context.session, 
          audio_event
      )
      
      logger.info(
          'Successfully flushed %s cache: %d chunks, %d bytes, saved as %s',
          cache_type,
          len(audio_cache),
          len(combined_audio_data),
          filename
      )
      return True
      
    except Exception as e:
      logger.error('Failed to flush %s cache: %s', cache_type, e)
      return False

  def get_cache_stats(self, invocation_context: InvocationContext) -> dict[str, int]:
    """Get statistics about current cache state.
    
    Args:
      invocation_context: The invocation context.
      
    Returns:
      Dictionary containing cache statistics.
    """
    input_count = len(invocation_context.input_audio_cache or [])
    output_count = len(invocation_context.output_audio_cache or [])
    
    input_bytes = sum(
        len(entry.data.data) 
        for entry in (invocation_context.input_audio_cache or [])
    )
    output_bytes = sum(
        len(entry.data.data) 
        for entry in (invocation_context.output_audio_cache or [])
    )
    
    return {
        'input_chunks': input_count,
        'output_chunks': output_count,
        'input_bytes': input_bytes,
        'output_bytes': output_bytes,
        'total_chunks': input_count + output_count,
        'total_bytes': input_bytes + output_bytes
    }


class AudioCacheConfig:
  """Configuration for audio caching behavior."""
  
  def __init__(
      self,
      max_cache_size_bytes: int = 10 * 1024 * 1024,  # 10MB
      max_cache_duration_seconds: float = 300.0,  # 5 minutes
      auto_flush_threshold: int = 100  # Number of chunks
  ):
    """Initialize audio cache configuration.
    
    Args:
      max_cache_size_bytes: Maximum cache size in bytes before auto-flush.
      max_cache_duration_seconds: Maximum duration to keep data in cache.
      auto_flush_threshold: Number of chunks that triggers auto-flush.
    """
    self.max_cache_size_bytes = max_cache_size_bytes
    self.max_cache_duration_seconds = max_cache_duration_seconds
    self.auto_flush_threshold = auto_flush_threshold 