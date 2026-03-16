"""VoicERA Server — Thin Pipecat voice server with telephony integration.

Exposes two interfaces:
1. POST /call/outbound — Calling app requests an outbound call
2. WebSocket /ws/{call_id} — Telephony provider connects audio stream

Post-call, sends results to the calling app's webhook URL.
"""

import os
import json
import socket
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import requests
from loguru import logger
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .bot import handle_call


# ============================================================================
# CONFIG
# ============================================================================

VOBIZ_API_BASE = os.getenv("VOBIZ_API_BASE", "https://api.vobiz.in/v1")
VOBIZ_AUTH_ID = os.getenv("VOBIZ_AUTH_ID", "")
VOBIZ_AUTH_TOKEN = os.getenv("VOBIZ_AUTH_TOKEN", "")
VOBIZ_CALLER_ID = os.getenv("VOBIZ_CALLER_ID", "")
SERVER_URL = os.getenv("VOICERA_SERVER_URL", "")  # Public URL of this server
WEBSOCKET_URL = os.getenv("VOICERA_WEBSOCKET_URL", "")  # WSS URL of this server
API_KEY = os.getenv("VOICERA_API_KEY", "")  # Simple API key auth

# In-memory call config store (call_id → config)
# In production, use Redis for multi-process support
_pending_calls: dict[str, dict] = {}


# ============================================================================
# MODELS
# ============================================================================

class OutboundCallRequest(BaseModel):
    """Request from the calling application to initiate a call."""
    phone: str = Field(..., description="E.164 phone number to call")
    systemPrompt: str = Field(..., description="System prompt with {{variable}} placeholders")
    variables: dict[str, str] = Field(default_factory=dict, description="Variables to inject into prompt")
    greeting: str = Field(default="", description="First message Mira speaks")
    webhookUrl: str = Field(default="", description="URL to POST call results to when call ends")
    maxDurationSeconds: int = Field(default=600, description="Max call duration in seconds")
    callerId: Optional[str] = Field(default=None, description="Override caller ID")
    # Provider config
    llm: dict = Field(default_factory=lambda: {"provider": "openai", "model": "gpt-4o-mini"})
    stt: dict = Field(default_factory=lambda: {"provider": "deepgram", "language": "English"})
    tts: dict = Field(default_factory=lambda: {"provider": "cartesia", "args": {"voice_id": "95d51f79-c397-46f9-b49a-23763d3eaa2d"}})
    # Metadata — passed through to webhook, not used by voice server
    metadata: dict = Field(default_factory=dict)


# ============================================================================
# AUTH
# ============================================================================

def verify_api_key(request: Request) -> bool:
    # Auth disabled for now — re-enable once API key is properly configured via env vars
    return True


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="VoicERA Server",
    description="Thin Pipecat voice server with Vobiz telephony",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "voicera-server"}


@app.post("/call/outbound")
async def outbound_call(request: Request, body: OutboundCallRequest):
    """Initiate an outbound phone call.

    The calling app sends the system prompt, variables, and provider config.
    We dial via Vobiz, run the Pipecat pipeline, and POST results to webhookUrl.
    """
    if not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not VOBIZ_AUTH_ID or not VOBIZ_AUTH_TOKEN:
        raise HTTPException(status_code=500, detail="Vobiz credentials not configured")

    if not SERVER_URL:
        raise HTTPException(status_code=500, detail="VOICERA_SERVER_URL not configured")

    # Generate call ID
    call_id = f"vc_{int(time.time() * 1000)}"

    # Store call config for when Vobiz connects the WebSocket
    _pending_calls[call_id] = {
        "systemPrompt": body.systemPrompt,
        "variables": body.variables,
        "greeting": body.greeting,
        "webhookUrl": body.webhookUrl,
        "maxDurationSeconds": body.maxDurationSeconds,
        "llm": body.llm,
        "stt": body.stt,
        "tts": body.tts,
        "metadata": body.metadata,
        "phone": body.phone,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }

    # Dial via Vobiz
    try:
        caller_id = body.callerId or VOBIZ_CALLER_ID
        if not caller_id:
            raise ValueError("No caller ID configured")

        vobiz_url = f"{VOBIZ_API_BASE}/Account/{VOBIZ_AUTH_ID}/Call/"
        payload = {
            "from": caller_id,
            "to": body.phone,
            "answer_url": f"{SERVER_URL}/answer?call_id={call_id}",
            "answer_method": "POST",
        }

        logger.info(f"Dialing {body.phone} (call_id={call_id})")
        response = requests.post(
            vobiz_url,
            json=payload,
            headers={
                "X-Auth-ID": VOBIZ_AUTH_ID,
                "X-Auth-Token": VOBIZ_AUTH_TOKEN,
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        return JSONResponse(content={
            "success": True,
            "callId": call_id,
            "vobizCallId": result.get("call_uuid"),
            "phone": body.phone,
        })

    except requests.exceptions.HTTPError as e:
        _pending_calls.pop(call_id, None)
        error_body = ""
        try:
            error_body = e.response.text
        except Exception:
            pass
        logger.error(f"Outbound call failed: {e} | Vobiz response: {error_body}")
        raise HTTPException(status_code=500, detail=f"{e} | {error_body}")

    except Exception as e:
        _pending_calls.pop(call_id, None)
        logger.error(f"Outbound call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/answer", methods=["GET", "POST"])
async def vobiz_answer_webhook(request: Request):
    """Vobiz calls this when the user picks up.

    Returns XML instructing Vobiz to connect WebSocket audio to our /ws/{call_id}.
    """
    call_id = request.query_params.get("call_id", "unknown")
    form_data = dict(await request.form()) if request.method == "POST" else {}
    event = form_data.get("Event", "unknown")
    hangup_cause = form_data.get("HangupCause", "")

    if event == "StartApp":
        ws_url = WEBSOCKET_URL or SERVER_URL.replace("https://", "wss://").replace("http://", "ws://")
        websocket_url = f"{ws_url}/ws/{call_id}"

        sample_rate = int(os.getenv("SAMPLE_RATE", "8000"))
        if sample_rate == 16000:
            content_type = "audio/x-l16;rate=16000"
        else:
            content_type = f"audio/x-mulaw;rate={sample_rate}"

        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="{content_type}">
        {websocket_url}
    </Stream>
</Response>'''
        return Response(content=xml, media_type="application/xml")

    elif event == "Hangup":
        logger.info(f"Call {call_id} hangup: {hangup_cause}")
        # If user was busy/didn't answer, notify webhook
        if hangup_cause in ("USER_BUSY", "NO_ANSWER", "CALL_REJECTED"):
            config = _pending_calls.pop(call_id, None)
            if config and config.get("webhookUrl"):
                await _send_webhook(config["webhookUrl"], {
                    "callId": call_id,
                    "status": "no_answer",
                    "endedReason": hangup_cause,
                    "metadata": config.get("metadata", {}),
                })

    return Response(status_code=200)


@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    """Vobiz connects audio here after the user picks up."""
    await websocket.accept()
    logger.info(f"WebSocket connected: call_id={call_id}")

    config = _pending_calls.pop(call_id, None)
    if not config:
        logger.error(f"No config found for call_id={call_id}")
        await websocket.close(code=1008, reason="Unknown call")
        return

    stream_sid = None

    try:
        # Wait for Vobiz 'start' event
        first_message = await websocket.receive_text()
        data = json.loads(first_message)

        if data.get("event") != "start":
            logger.warning(f"Expected 'start', got: {data.get('event')}")
            return

        start_info = data.get("start", {})
        stream_sid = start_info.get("streamSid") or start_info.get("streamId", call_id)
        vobiz_call_sid = start_info.get("callSid") or start_info.get("callId", call_id)

        logger.info(f"Call started: call_id={call_id}, stream={stream_sid}")

        # Run the voice pipeline
        result = await handle_call(
            websocket_client=websocket,
            stream_sid=stream_sid,
            call_sid=call_id,
            call_config=config,
        )

        # Send results to calling app's webhook
        if config.get("webhookUrl"):
            await _send_webhook(config["webhookUrl"], {
                **result,
                "status": "completed",
                "endedReason": "call_ended",
                "metadata": config.get("metadata", {}),
            })

    except Exception as e:
        logger.error(f"Call {call_id} error: {e}")
        logger.debug(traceback.format_exc())

        if config.get("webhookUrl"):
            await _send_webhook(config["webhookUrl"], {
                "callId": call_id,
                "status": "error",
                "endedReason": str(e),
                "metadata": config.get("metadata", {}),
            })
    finally:
        logger.info(f"WebSocket closed: call_id={call_id}")


# ============================================================================
# WEBHOOK
# ============================================================================

async def _send_webhook(url: str, data: dict) -> None:
    """POST call results to the calling app's webhook."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                logger.info(f"Webhook sent to {url}: status={resp.status}")
    except Exception as e:
        logger.error(f"Webhook failed ({url}): {e}")


# ============================================================================
# SERVER
# ============================================================================

def create_nodelay_websocket_protocol():
    try:
        from uvicorn.protocols.websockets.websockets_impl import WebSocketProtocol

        class NoDelayWebSocketProtocol(WebSocketProtocol):
            def connection_made(self, transport):
                try:
                    sock = transport.get_extra_info("socket")
                    if sock is not None:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
                super().connection_made(transport)

        return NoDelayWebSocketProtocol
    except ImportError:
        return None


def run_server(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        loop="auto",
        ws="websockets",
    )

    nodelay_protocol = create_nodelay_websocket_protocol()
    if nodelay_protocol:
        config.ws_protocol_class = nodelay_protocol
        logger.info("TCP_NODELAY enabled for WebSocket connections")

    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    run_server()
