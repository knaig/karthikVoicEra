"""IndicConformer REST STT Service for Pipecat — client-side buffering with REST calls."""

import os
import asyncio
import base64
import time
from typing import AsyncGenerator, Optional, Dict, Any
from loguru import logger

from pipecat.frames.frames import (
    Frame, TranscriptionFrame, InterimTranscriptionFrame, ErrorFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.audio.utils import create_stream_resampler
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState

try:
    import aiohttp
except ImportError:
    raise ImportError("aiohttp required for AI4Bharat STT: pip install aiohttp")


class IndicConformerRESTSTTService(STTService):

    def __init__(
        self, *, language_id: str = "hi", sample_rate: int = 16000,
        input_sample_rate: int = 8000, server_url: str = None,
        vad_analyzer: Optional[VADAnalyzer] = None, **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._server_url = (server_url or os.getenv("INDIC_STT_SERVER_URL", "")).rstrip("/")
        if not self._server_url:
            raise ValueError("server_url or INDIC_STT_SERVER_URL env var required")
        self._server_url += "/transcribe"

        self._language_id = language_id
        self._sample_rate = sample_rate
        self._input_sample_rate = input_sample_rate
        self._session: Optional[aiohttp.ClientSession] = None
        self._vad_analyzer = vad_analyzer
        self._audio_buffer = b""
        self._text_chunks = []
        self._is_speaking = False
        self._stopping_start_time: Optional[float] = None
        self._stopping_triggered = False
        self._STOPPING_DURATION_MS = 10
        self._resampler = create_stream_resampler()

    async def _transcribe_buffer(self) -> str:
        if not self._audio_buffer or len(self._audio_buffer) < 3200:
            return ""
        try:
            audio_b64 = base64.b64encode(self._audio_buffer).decode("utf-8")
            async with self._session.post(
                self._server_url,
                json={"audio_b64": audio_b64, "language_id": self._language_id},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    return (await response.json()).get("text", "")
                return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def _check_stopping_state(self) -> bool:
        if self._vad_analyzer is None:
            return False
        try:
            vad_state = self._vad_analyzer._vad_state
            if vad_state == VADState.STOPPING:
                now = time.time() * 1000
                if self._stopping_start_time is None:
                    self._stopping_start_time = now
                    return False
                if now - self._stopping_start_time >= self._STOPPING_DURATION_MS and not self._stopping_triggered:
                    self._stopping_triggered = True
                    return True
                return False
            else:
                self._stopping_start_time = None
                self._stopping_triggered = False
                return False
        except AttributeError:
            return False

    async def process_frame(self, frame: Frame, direction):
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._is_speaking = False
            self._stopping_start_time = None
            self._stopping_triggered = False
            if self._text_chunks:
                accumulated = " ".join(self._text_chunks)
                await self.push_frame(TranscriptionFrame(
                    text=accumulated, user_id=self._user_id,
                    timestamp=str(int(time.time() * 1000)),
                ))
                self._text_chunks = []
                self._audio_buffer = b""
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._is_speaking = True
            self._stopping_start_time = None
            self._stopping_triggered = False
            self._audio_buffer = b""
            self._text_chunks = []

    async def start(self, frame: Frame) -> None:
        self._session = aiohttp.ClientSession()
        self._is_speaking = False
        self._audio_buffer = b""
        self._text_chunks = []
        await super().start(frame)

    async def stop(self, frame: Frame) -> None:
        if self._session:
            await self._session.close()
        await super().stop(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not audio:
            return
        try:
            resampled = audio
            if self._input_sample_rate != self._sample_rate:
                resampled = await self._resampler.resample(audio, self._input_sample_rate, self._sample_rate)
            self._audio_buffer += resampled
            if self._check_stopping_state():
                text = await self._transcribe_buffer()
                if text:
                    self._text_chunks.append(text)
                    accumulated = " ".join(self._text_chunks)
                    yield InterimTranscriptionFrame(
                        text=accumulated, user_id=self._user_id,
                        timestamp=str(int(time.time() * 1000)),
                    )
                self._audio_buffer = b""
        except Exception as e:
            logger.error(f"STT processing error: {e}")
            yield ErrorFrame(f"STT processing failed: {e}")

    async def set_language(self, language_id: str):
        self._language_id = language_id

    def can_generate_metrics(self) -> bool:
        return True
