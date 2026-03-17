
import time
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    BotStoppedSpeakingFrame,
    StartInterruptionFrame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# Max time to block interruptions — safety net in case BotStoppedSpeakingFrame
# never reaches this processor (it travels downstream, not back upstream).
GREETING_MAX_BLOCK_SECS = 5.0


class GreetingInterruptionFilter(FrameProcessor):
    """Filters out interruption frames while the greeting is being played.

    Auto-disables after GREETING_MAX_BLOCK_SECS as a safety net, since
    BotStoppedSpeakingFrame travels downstream and may never reach this
    upstream processor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._greeting_in_progress = False
        self._greeting_start_time = 0.0

    def start_greeting(self):
        self._greeting_in_progress = True
        self._greeting_start_time = time.monotonic()
        logger.debug("Greeting started - interruptions blocked")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._greeting_in_progress:
            # Check timeout — auto-disable if greeting took too long
            elapsed = time.monotonic() - self._greeting_start_time
            if elapsed > GREETING_MAX_BLOCK_SECS:
                self._greeting_in_progress = False
                logger.info(f"Greeting filter auto-disabled after {elapsed:.1f}s timeout")

            # BotStoppedSpeakingFrame may arrive if propagated upstream
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._greeting_in_progress = False
                logger.info("Greeting completed - interruptions enabled")

        if self._greeting_in_progress:
            if isinstance(frame, (StartInterruptionFrame, InterruptionFrame, UserStartedSpeakingFrame)):
                logger.debug(f"Blocked {frame.__class__.__name__} during greeting")
                return

        await self.push_frame(frame, direction)
