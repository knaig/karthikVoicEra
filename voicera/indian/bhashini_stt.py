"""Bhashini Socket.IO STT Service for Pipecat — 22 Indian languages."""

import asyncio
import os
from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import (
    Frame, TranscriptionFrame, InterimTranscriptionFrame, ErrorFrame,
    StartFrame, EndFrame, CancelFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

try:
    import socketio
except ModuleNotFoundError:
    raise ImportError("python-socketio required: pip install python-socketio[asyncio_client] aiohttp")


class BhashiniSTTService(STTService):

    def __init__(
        self, *, api_key: str, socket_url: str = None,
        service_id: str = "bhashini/ai4bharat/conformer-multilingual-asr",
        language: str = "hi", sample_rate: int = 16000,
        response_frequency_secs: float = 1.0, **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_key = api_key
        self._socket_url = socket_url or os.getenv("BHASHINI_SOCKET_URL", "wss://dhruva-api.bhashini.gov.in")
        self._service_id = service_id
        self._language = language
        self._response_frequency_secs = response_frequency_secs
        self._sio: Optional[socketio.AsyncClient] = None
        self._is_connected = False
        self._is_ready = False
        self._is_speaking = False
        self._ready_event: Optional[asyncio.Event] = None

    def _build_task_sequence(self) -> list:
        return [{"taskType": "asr", "config": {
            "serviceId": self._service_id,
            "language": {"sourceLanguage": self._language},
            "samplingRate": self.sample_rate, "audioFormat": "wav",
        }}]

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._send_end_of_stream()
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._disconnect()
        await super().cancel(frame)

    async def _connect(self):
        self._ready_event = asyncio.Event()
        self._sio = socketio.AsyncClient(reconnection_attempts=5)

        @self._sio.event
        async def connect():
            self._is_connected = True
            await self._sio.emit("start", (
                self._build_task_sequence(),
                {"responseFrequencyInSecs": self._response_frequency_secs},
            ))

        @self._sio.event
        async def connect_error(data):
            await self.push_error(ErrorFrame(error=f"Connection error: {data}"))

        @self._sio.on("ready")
        async def on_ready():
            self._is_ready = True
            self._ready_event.set()

        @self._sio.on("response")
        async def on_response(response, streaming_status):
            await self._handle_response(response, streaming_status)

        @self._sio.on("abort")
        async def on_abort(message):
            await self.push_error(ErrorFrame(error=f"Aborted: {message}"))

        @self._sio.on("terminate")
        async def on_terminate():
            self._is_ready = False
            self._is_connected = False

        @self._sio.event
        async def disconnect():
            self._is_connected = False
            self._is_ready = False

        try:
            await self._sio.connect(
                url=self._socket_url, transports=["websocket", "polling"],
                socketio_path="/socket.io", auth={"authorization": self._api_key},
            )
            await asyncio.wait_for(self._ready_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            await self.push_error(ErrorFrame(error="Connection timeout"))
        except Exception as e:
            await self.push_error(ErrorFrame(error=str(e)))

    async def _disconnect(self):
        if self._sio:
            self._is_ready = False
            self._is_connected = False
            try:
                await self._sio.disconnect()
            except Exception:
                pass
            self._sio = None

    async def _send_end_of_stream(self):
        if not self._sio or not self._is_connected:
            return
        try:
            await self._sio.emit("data", (None, None, True, False))
            await self._sio.emit("data", (None, None, True, True))
        except Exception:
            pass

    async def _handle_response(self, response: dict, streaming_status: dict):
        try:
            is_interim = streaming_status.get("isIntermediateResult", True)
            pipeline_response = response.get("pipelineResponse", [])
            if not pipeline_response:
                return
            outputs = pipeline_response[0].get("output", [])
            if not outputs:
                return
            if is_interim:
                transcript = outputs[0].get("source", "")
            else:
                transcript = ". ".join(
                    c.get("source", "") for c in outputs if c.get("source", "").strip()
                )
            if not transcript.strip():
                return
            frame_cls = InterimTranscriptionFrame if is_interim else TranscriptionFrame
            await self.push_frame(frame_cls(
                text=transcript, user_id=self._user_id, timestamp=time_now_iso8601(),
            ))
        except Exception as e:
            logger.error(f"Response handling error: {e}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._is_ready or not audio:
            yield None
            return
        try:
            await self._sio.emit("data", (
                {"audio": [{"audioContent": audio}]}, {}, False, False,
            ))
        except Exception as e:
            yield ErrorFrame(error=str(e))
            return
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._is_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_speaking = False
            if self._sio and self._is_ready:
                try:
                    await self._sio.emit("data", (None, None, True, False))
                except Exception:
                    pass

    async def set_language(self, language: str):
        self._language = language
        await self._disconnect()
        await self._connect()

    def can_generate_metrics(self) -> bool:
        return True
