"""Filters interruptions during greeting playback."""

from loguru import logger
from pipecat.frames.frames import (
    Frame, BotStoppedSpeakingFrame, StartInterruptionFrame,
    InterruptionFrame, UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class GreetingInterruptionFilter(FrameProcessor):
    """Blocks interruption frames while the greeting is being played."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._greeting_in_progress = False

    def start_greeting(self):
        self._greeting_in_progress = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStoppedSpeakingFrame) and self._greeting_in_progress:
            self._greeting_in_progress = False
        if self._greeting_in_progress:
            if isinstance(frame, (StartInterruptionFrame, InterruptionFrame, UserStartedSpeakingFrame)):
                return
        await self.push_frame(frame, direction)
