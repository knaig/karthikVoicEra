"""VoicERA Server — Thin Pipecat voice server with telephony integration.
Exposes two interfaces:
1. POST /call/outbound — Calling app requests an outbound call
2. WebSocket /ws/{call_id} — Telephony provider connects audio stream
3. POST /call/web — Browser initiates a WebRTC call (SDP offer/answer)
4. POST /call/web/patch — ICE candidate trickle for WebRTC

Post-call, sends results to the calling app's webhook URL.
"""
import asyncio
import uuid
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
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequestHandler,
    SmallWebRTCRequest,
    SmallWebRTCPatchRequest,
    IceCandidate,
)

from .bot import handle_call
from .web_bot import handle_web_call

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

# Pre-loaded call sessions (token → config), set by /call/web/prepare
_prepared_sessions: dict[str, dict] = {}

# WebRTC connection manager (handles SDP offer/answer + ICE)
_webrtc_handler = SmallWebRTCRequestHandler()

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


class WebCallRequest(BaseModel):
    """SDP offer from the browser to initiate a WebRTC call."""
    sdp: str = Field(..., description="SDP offer string from browser RTCPeerConnection")
    type: str = Field(default="offer", description="SDP type (always 'offer' for new calls)")
    pc_id: Optional[str] = Field(default=None, description="Peer connection ID (for renegotiation)")
    restart_pc: Optional[bool] = Field(default=None)
    # Call config fields (same as OutboundCallRequest minus phone)
    systemPrompt: str = Field(default="You are Mira, an AI executive coach.")
    variables: dict = Field(default_factory=dict)
    greeting: str = Field(default="")
    webhookUrl: str = Field(default="")
    llm: dict = Field(default_factory=lambda: {"provider": "openai", "model": "gpt-4o-mini"})
    stt: dict = Field(default_factory=lambda: {"provider": "deepgram", "language": "English"})
    tts: dict = Field(default_factory=lambda: {"provider": "openai", "args": {"voice": "nova"}})
    metadata: dict = Field(default_factory=dict)



class PrepareCallRequest(BaseModel):
    """Pre-load context for an upcoming web call (used by Cowork plugin skill)."""
    systemPrompt: str = Field(default="You are Mira, an AI executive coach.")
    greeting: str = Field(default="")
    llm: dict = Field(default_factory=lambda: {"provider": "gemini", "model": "gemini-2.0-flash"})
    stt: dict = Field(default_factory=lambda: {"provider": "openai", "language": "English"})
    tts: dict = Field(default_factory=lambda: {"provider": "openai", "args": {"voice": "nova"}})
    metadata: dict = Field(default_factory=dict)
    ttlSeconds: int = Field(default=300, description="How long to keep the session alive (default 5 min)")

class WebPatchRequest(BaseModel):
    """ICE candidate trickle request."""
    pc_id: str
    candidates: list[dict]


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
    description="Thin Pipecat voice server with Vobiz telephony + WebRTC",
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


@app.get("/debug")
async def debug_logs_endpoint():
    """Return recent pipeline debug logs."""
    from .bot import debug_logs
    return {"logs": list(debug_logs)}


@app.get("/", response_class=HTMLResponse)
async def serve_web_client():
    """Serve the web call client HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), "web_client.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>VoicERA Web Client not found</h1><p>Place web_client.html in the api/ directory.</p>")


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



@app.post("/call/web/prepare")
async def prepare_web_call(request: Request, body: PrepareCallRequest):
    """Pre-load a call config from the Cowork plugin skill.
    Returns a session URL the skill can open in the browser.
    The web client fetches the config from /call/web/session/{token}/config.
    """
    token = uuid.uuid4().hex
    _prepared_sessions[token] = {
        "systemPrompt": body.systemPrompt,
        "greeting": body.greeting,
        "llm": body.llm,
        "stt": body.stt,
        "tts": body.tts,
        "metadata": body.metadata,
        "createdAt": time.time(),
        "ttlSeconds": body.ttlSeconds,
    }
    base_url = SERVER_URL or "http://localhost:7860"
    session_url = f"{base_url}/call/web/session/{token}"
    logger.info(f"Prepared call session {token} (expires in {body.ttlSeconds}s)")
    return JSONResponse(content={"token": token, "url": session_url})


@app.get("/call/web/session/{token}/config")
async def get_session_config(token: str):
    """Fetch pre-loaded config for a call session (called by the web client JS)."""
    session = _prepared_sessions.get(token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    # Check TTL
    age = time.time() - session["createdAt"]
    if age > session["ttlSeconds"]:
        _prepared_sessions.pop(token, None)
        raise HTTPException(status_code=410, detail="Session expired")
    return JSONResponse(content=session)


@app.get("/call/web/session/{token}", response_class=HTMLResponse)
async def serve_session_client(token: str):
    """Serve the web client for a specific pre-loaded session."""
    html_path = os.path.join(os.path.dirname(__file__), "web_client.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="<h1>Web client not found</h1>")
    with open(html_path) as f:
        html = f.read()
    # Inject the session token so the client knows which config to fetch
    html = html.replace("</head>", f'<script>window.MIRA_SESSION_TOKEN = "{token}";</script></head>')
    return HTMLResponse(content=html)

@app.post("/call/web")
async def web_call(request: Request, body: WebCallRequest):
    """Initiate a browser WebRTC call.

    Browser sends an SDP offer; we return an SDP answer.
    The Pipecat pipeline starts running in the background on the same event loop.
    """
    if not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    call_id = f"wc_{int(time.time() * 1000)}"
    call_config = {
        "callId": call_id,
        "systemPrompt": body.systemPrompt,
        "variables": body.variables,
        "greeting": body.greeting,
        "webhookUrl": body.webhookUrl,
        "llm": body.llm,
        "stt": body.stt,
        "tts": body.tts,
        "metadata": body.metadata,
    }

    logger.info(f"Web call {call_id}: SDP offer received, starting pipeline")

    webrtc_request = SmallWebRTCRequest(
        sdp=body.sdp,
        type=body.type,
        pc_id=body.pc_id,
        restart_pc=body.restart_pc,
    )

    async def on_webrtc_connection(webrtc_connection):
        """Called by SmallWebRTCRequestHandler once the peer connection is set up.
        Kick off the Pipecat pipeline as a background task so we can return the
        SDP answer immediately.
        """
        async def run_pipeline():
            try:
                result = await handle_web_call(webrtc_connection, call_config)
                if call_config.get("webhookUrl"):
                    await _send_webhook(call_config["webhookUrl"], {
                        **result,
                        "endedReason": "call_ended",
                        "metadata": call_config.get("metadata", {}),
                    })
            except Exception as e:
                logger.error(f"Web call pipeline error: {e}")
                logger.debug(traceback.format_exc())

        asyncio.create_task(run_pipeline())

    answer = await _webrtc_handler.handle_web_request(webrtc_request, on_webrtc_connection)
    return JSONResponse(content=answer)


@app.post("/call/web/patch")
async def web_call_patch(request: Request, body: WebPatchRequest):
    """Add ICE candidates to an existing WebRTC peer connection (trickle ICE)."""
    ice_candidates = [
        IceCandidate(
            candidate=c["candidate"],
            sdp_mid=c.get("sdpMid", ""),
            sdp_mline_index=c.get("sdpMLineIndex", 0),
        )
        for c in body.candidates
        if c.get("candidate")
    ]
    from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCPatchRequest as _PatchReq
    patch_req = _PatchReq(pc_id=body.pc_id, candidates=ice_candidates)
    await _webrtc_handler.handle_patch_request(patch_req)
    return JSONResponse(content={"ok": True})


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

        xml = f'''<?xml version="1.0" encoding="UTF-8"?><Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="{content_type}">
        {websocket_url}
    </Stream>
</Response>'''
        return Response(content=xml, media_type="application/xml")

    elif event == "Hangup":
        logger.info(f"Call {call_id} hangup: {hangup_cause}")
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
