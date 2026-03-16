"""VoiceCall — the main SDK interface."""

import asyncio
import os
import time
from typing import Callable, Optional
from dataclasses import dataclass

from loguru import logger

from voicera.tools import ToolRegistry
from voicera.audio.optimizations import inject_variables


@dataclass
class CallResult:
    """Result returned after a call ends."""
    call_id: str
    transcript: str
    transcript_lines: list[str]
    duration: int
    started_at: str
    ended_at: str


class VoiceCall:
    """Make AI voice calls with tool calling support.

    Usage:
        call = VoiceCall(
            phone="+919886974008",
            system_prompt="You are Mira, an AI coach...",
            greeting="Hey, it's Mira.",
            llm={"provider": "gemini", "model": "gemini-2.0-flash"},
            stt={"provider": "openai", "language": "English"},
            tts={"provider": "openai", "args": {"voice": "nova"}},
            telephony={"provider": "vobiz"},
        )

        @call.tool("lookup")
        def lookup(name: str) -> str:
            return db.query(name)

        @call.on("call_ended")
        def on_end(result):
            print(f"Done: {result.duration}s")

        await call.start()
    """

    def __init__(
        self,
        phone: str,
        system_prompt: str,
        greeting: str = "",
        stt: dict = None,
        tts: dict = None,
        llm: dict = None,
        telephony: dict = None,
        max_duration: int = 600,
        variables: dict = None,
        sample_rate: int = 8000,
    ):
        self.phone = phone
        self.system_prompt = system_prompt
        self.greeting = greeting
        self.stt_config = stt or {"provider": "deepgram", "language": "English"}
        self.tts_config = tts or {"provider": "cartesia"}
        self.llm_config = llm or {"provider": "gemini", "model": "gemini-2.0-flash"}
        self.telephony_config = telephony or {"provider": "vobiz"}
        self.max_duration = max_duration
        self.variables = variables or {}
        self.sample_rate = sample_rate

        self._tool_registry = ToolRegistry()
        self._callbacks: dict[str, list[Callable]] = {}
        self._result: Optional[CallResult] = None

    def tool(self, name: str):
        """Decorator to register a function as a callable tool.

        The function runs in the caller's process, not over HTTP.

            @call.tool("check_calendar")
            def check_calendar(date: str) -> str:
                return json.dumps(get_meetings(date))
        """
        def decorator(func):
            self._tool_registry.register(name, func)
            return func
        return decorator

    def on(self, event: str):
        """Decorator to register an event callback.

        Events: "call_ended", "transcript_update"

            @call.on("call_ended")
            def on_end(result):
                print(result.duration)
        """
        def decorator(func):
            self._callbacks.setdefault(event, []).append(func)
            return func
        return decorator

    async def _fire_callbacks(self, event: str, *args):
        for cb in self._callbacks.get(event, []):
            try:
                result = cb(*args)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.error(f"Callback error ({event}): {e}")

    def _is_remote_server(self) -> bool:
        """Check if server_url points to a remote deployment (not localhost)."""
        url = self.telephony_config.get("server_url", "")
        return url.startswith("https://") or (url.startswith("http://") and "localhost" not in url and "127.0.0.1" not in url)

    async def _start_remote(self) -> CallResult:
        """Delegate call to a deployed voice server via its REST API."""
        import aiohttp

        prompt = self.system_prompt
        if self.variables:
            prompt = inject_variables(prompt, self.variables)
            # Also inject into greeting
            greeting = inject_variables(self.greeting, self.variables) if self.greeting else ""
        else:
            greeting = self.greeting

        server_url = self.telephony_config["server_url"].rstrip("/")
        api_key = self.telephony_config.get("api_key") or os.getenv("VOICERA_API_KEY", "")

        payload = {
            "phone": self.phone,
            "systemPrompt": prompt,
            "variables": {},  # Already injected
            "greeting": greeting,
            "llm": self.llm_config,
            "stt": self.stt_config,
            "tts": self.tts_config,
            "maxDurationSeconds": self.max_duration,
            "metadata": {},
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/call/outbound",
                json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Server returned {resp.status}: {body}")
                data = await resp.json()

        call_id = data.get("callId", "unknown")
        self._result = CallResult(
            call_id=call_id, transcript="(call in progress on remote server)",
            transcript_lines=[], duration=0,
            started_at="", ended_at="",
        )

        logger.info(f"Call initiated on remote server: {call_id}")
        await self._fire_callbacks("call_ended", self._result)
        return self._result

    async def _start_local(self) -> CallResult:
        """Run the full pipeline locally (requires public URL for webhooks)."""
        from voicera.pipeline import run_voice_pipeline
        from voicera.telephony.vobiz import VobizTelephony

        prompt = self.system_prompt
        if self.variables:
            prompt = inject_variables(prompt, self.variables)

        telephony = VobizTelephony(
            auth_id=self.telephony_config.get("auth_id"),
            auth_token=self.telephony_config.get("auth_token"),
            caller_id=self.telephony_config.get("caller_id"),
            server_url=self.telephony_config.get("server_url"),
            websocket_url=self.telephony_config.get("websocket_url"),
            api_base=self.telephony_config.get("api_base"),
            sample_rate=self.sample_rate,
        )

        server_port = self.telephony_config.get("port", 7860)
        await telephony.start_server(port=server_port)
        await asyncio.sleep(0.5)

        call_id = f"vc_{int(time.time() * 1000)}"

        try:
            connection = await telephony.dial(self.phone, call_id=call_id)

            on_transcript = None
            if "transcript_update" in self._callbacks:
                async def on_transcript(msg):
                    await self._fire_callbacks("transcript_update", msg)

            result_dict = await run_voice_pipeline(
                websocket=connection.websocket,
                stream_sid=connection.stream_sid,
                call_sid=call_id,
                system_prompt=prompt,
                greeting=self.greeting,
                llm_config=self.llm_config,
                stt_config=self.stt_config,
                tts_config=self.tts_config,
                tools=self._tool_registry.get_openai_tools() if self._tool_registry.has_tools() else None,
                tool_handler=self._tool_registry.execute if self._tool_registry.has_tools() else None,
                on_transcript_update=on_transcript,
                max_duration=self.max_duration,
                sample_rate=self.sample_rate,
                serializer=connection.serializer,
            )

            self._result = CallResult(**result_dict)

        except Exception as e:
            logger.error(f"Call failed: {e}")
            self._result = CallResult(
                call_id=call_id, transcript="", transcript_lines=[],
                duration=0, started_at="", ended_at="",
            )
            raise

        finally:
            await telephony.shutdown()

        await self._fire_callbacks("call_ended", self._result)
        return self._result

    async def start(self) -> CallResult:
        """Start the voice call.

        If server_url is a remote deployment (https://), delegates to the
        deployed server. Otherwise runs the pipeline locally (requires a
        publicly accessible server_url for Vobiz webhooks).
        """
        if self._is_remote_server():
            return await self._start_remote()
        return await self._start_local()
