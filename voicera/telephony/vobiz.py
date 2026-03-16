"""Vobiz telephony — dial, webhook, WebSocket, frame serializer."""

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
import aiohttp
from loguru import logger
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response

from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.frames.frames import AudioRawFrame, InputAudioRawFrame, Frame


# ── Vobiz Frame Serializer ──────────────────────────────────────────

class VobizFrameSerializer(PlivoFrameSerializer):
    """Plivo-compatible serializer with 16kHz L16 support for Vobiz."""

    class InputParams(PlivoFrameSerializer.InputParams):
        def __init__(self, vobiz_sample_rate: int = 8000, sample_rate: int = None, auto_hang_up: bool = True):
            super().__init__(plivo_sample_rate=vobiz_sample_rate, sample_rate=sample_rate, auto_hang_up=auto_hang_up)

    def __init__(self, stream_sid: str, call_sid: str, params: InputParams = None):
        super().__init__(stream_id=stream_sid, call_id=call_sid, params=params or self.InputParams())

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if self._plivo_sample_rate == 16000 and isinstance(frame, AudioRawFrame):
            data = frame.audio
            if frame.sample_rate != 16000:
                data = await self._output_resampler.resample(data, frame.sample_rate, 16000)
            payload = base64.b64encode(data).decode("utf-8")
            return json.dumps({
                "event": "playAudio",
                "media": {"contentType": "audio/x-l16", "sampleRate": 16000, "payload": payload},
                "streamId": self._stream_id,
            })
        return await super().serialize(frame)

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if self._plivo_sample_rate == 16000:
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                return None
            if message.get("event") == "media":
                payload_b64 = message.get("media", {}).get("payload")
                if payload_b64:
                    return InputAudioRawFrame(
                        audio=base64.b64decode(payload_b64), num_channels=1, sample_rate=16000,
                    )
        return await super().deserialize(data)


# ── Vobiz Connection Result ─────────────────────────────────────────

@dataclass
class VobizConnection:
    """Returned when Vobiz connects WebSocket audio."""
    websocket: WebSocket
    stream_sid: str
    call_sid: str
    serializer: VobizFrameSerializer


# ── Vobiz Telephony Provider ────────────────────────────────────────

class VobizTelephony:
    """Manages Vobiz outbound calls — dial, webhook, WebSocket."""

    def __init__(
        self, *,
        auth_id: str = None, auth_token: str = None, caller_id: str = None,
        server_url: str = None, websocket_url: str = None,
        api_base: str = None, sample_rate: int = 8000,
    ):
        self.auth_id = auth_id or os.getenv("VOBIZ_AUTH_ID", "")
        self.auth_token = auth_token or os.getenv("VOBIZ_AUTH_TOKEN", "")
        self.caller_id = caller_id or os.getenv("VOBIZ_CALLER_ID", "")
        self.server_url = server_url or os.getenv("VOICERA_SERVER_URL", "")
        self.websocket_url = websocket_url or os.getenv("VOICERA_WEBSOCKET_URL", "")
        self.api_base = api_base or os.getenv("VOBIZ_API_BASE", "https://api.vobiz.ai/api/v1")
        self.sample_rate = sample_rate

        self._app: Optional[FastAPI] = None
        self._server_task: Optional[asyncio.Task] = None
        self._pending: dict[str, asyncio.Future] = {}

    def _create_app(self) -> FastAPI:
        """Create a minimal FastAPI app for Vobiz webhooks."""
        app = FastAPI()

        @app.api_route("/answer", methods=["GET", "POST"])
        async def answer(request: Request):
            call_id = request.query_params.get("call_id", "unknown")
            form_data = dict(await request.form()) if request.method == "POST" else {}
            event = form_data.get("Event", "unknown")

            if event == "StartApp":
                ws_url = self.websocket_url or self.server_url.replace("https://", "wss://").replace("http://", "ws://")
                websocket_url = f"{ws_url}/ws/{call_id}"
                if self.sample_rate == 16000:
                    content_type = "audio/x-l16;rate=16000"
                else:
                    content_type = f"audio/x-mulaw;rate={self.sample_rate}"
                xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="{content_type}">
        {websocket_url}
    </Stream>
</Response>'''
                return Response(content=xml, media_type="application/xml")

            elif event == "Hangup":
                hangup_cause = form_data.get("HangupCause", "")
                if hangup_cause in ("USER_BUSY", "NO_ANSWER", "CALL_REJECTED"):
                    fut = self._pending.pop(call_id, None)
                    if fut and not fut.done():
                        fut.set_exception(ConnectionError(f"Call not answered: {hangup_cause}"))

            return Response(status_code=200)

        @app.websocket("/ws/{call_id}")
        async def ws_endpoint(websocket: WebSocket, call_id: str):
            await websocket.accept()
            logger.info(f"Vobiz WebSocket connected: call_id={call_id}")

            try:
                first_message = await websocket.receive_text()
                data = json.loads(first_message)
                if data.get("event") != "start":
                    logger.warning(f"Expected 'start', got: {data.get('event')}")
                    return

                start_info = data.get("start", {})
                stream_sid = start_info.get("streamSid") or start_info.get("streamId", call_id)

                serializer = VobizFrameSerializer(
                    stream_sid=stream_sid, call_sid=call_id,
                    params=VobizFrameSerializer.InputParams(
                        vobiz_sample_rate=self.sample_rate, sample_rate=self.sample_rate,
                    ),
                )
                connection = VobizConnection(
                    websocket=websocket, stream_sid=stream_sid,
                    call_sid=call_id, serializer=serializer,
                )

                fut = self._pending.get(call_id)
                if fut and not fut.done():
                    fut.set_result(connection)

                # Keep WebSocket alive until the pipeline closes it
                while True:
                    try:
                        await asyncio.wait_for(websocket.receive(), timeout=600)
                    except Exception:
                        break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

        return app

    async def start_server(self, host: str = "0.0.0.0", port: int = 7860):
        """Start the webhook/WebSocket server in background."""
        import uvicorn
        self._app = self._create_app()
        config = uvicorn.Config(self._app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())

    async def dial(self, phone: str, call_id: str = None) -> VobizConnection:
        """Dial a phone number via Vobiz and wait for WebSocket connection."""
        if not self.auth_id or not self.auth_token:
            raise ValueError("Vobiz credentials not configured")
        if not self.server_url:
            raise ValueError("VOICERA_SERVER_URL not configured")

        call_id = call_id or f"vc_{int(time.time() * 1000)}"

        # Create a future for this call
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending[call_id] = fut

        # Dial
        vobiz_url = f"{self.api_base}/Account/{self.auth_id}/Call/"
        payload = {
            "from": self.caller_id,
            "to": phone,
            "answer_url": f"{self.server_url}/answer?call_id={call_id}",
            "answer_method": "POST",
        }

        logger.info(f"Dialing {phone} (call_id={call_id})")
        response = requests.post(
            vobiz_url, json=payload,
            headers={"X-Auth-ID": self.auth_id, "X-Auth-Token": self.auth_token, "Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()

        # Wait for Vobiz to connect WebSocket (timeout 60s)
        try:
            connection = await asyncio.wait_for(fut, timeout=60)
            return connection
        except asyncio.TimeoutError:
            self._pending.pop(call_id, None)
            raise TimeoutError(f"Vobiz did not connect WebSocket for call {call_id}")

    async def shutdown(self):
        """Stop the webhook server."""
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
