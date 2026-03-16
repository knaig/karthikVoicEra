"""Voice bot pipeline — Pipecat STT → LLM → TTS with variable injection.

The calling application sends a system prompt with {{variable}} placeholders.
This module replaces them with provided values before starting the pipeline.
"""

import os
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame, TTSStartedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator, Aggregation, AggregationType
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from serializer.vobiz_serializer import VobizFrameSerializer
from services.audio.greeting_interruption_filter import GreetingInterruptionFilter
from .services import create_llm_service, create_stt_service, create_tts_service


# ============================================================================
# VARIABLE INJECTION
# ============================================================================

def inject_variables(template: str, variables: dict[str, str]) -> str:
    """Replace {{variableName}} placeholders with values.

    This is how the calling app injects context into the system prompt —
    meeting data, user info, coaching posture, etc.
    """
    def replacer(match: re.Match) -> str:
        key = match.group(1).strip()
        return variables.get(key, match.group(0))

    return re.sub(r'\{\{(\s*\w+\s*)\}\}', replacer, template)


# ============================================================================
# LATENCY OPTIMIZATIONS (from VoicERA)
# ============================================================================

# Monkey-patch SOXR resampler for lower latency
try:
    from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
    import soxr

    def patched_initialize(self, in_rate: float, out_rate: float):
        self._in_rate = in_rate
        self._out_rate = out_rate
        self._last_resample_time = time.time()
        self._soxr_stream = soxr.ResampleStream(
            in_rate=in_rate, out_rate=out_rate, num_channels=1, quality="QQ", dtype="int16"
        )

    SOXRStreamAudioResampler._initialize = patched_initialize
    logger.info("SOXR resampler patched for low latency (Quick quality)")
except Exception as e:
    logger.warning(f"Failed to patch SOXR resampler: {e}")


class FastPunctuationAggregator(BaseTextAggregator):
    """Send text on punctuation immediately — no NLTK lookahead."""

    def __init__(self):
        self._text = ""

    @property
    def text(self):
        return Aggregation(text=self._text.strip(), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str):
        for char in text:
            self._text += char
            if char in '.!?,':
                if self._text.strip():
                    yield Aggregation(self._text.strip(), AggregationType.SENTENCE)
                    self._text = ""

    async def flush(self):
        if self._text.strip():
            result = self._text.strip()
            self._text = ""
            return Aggregation(result, AggregationType.SENTENCE)
        return None

    async def handle_interruption(self):
        self._text = ""

    async def reset(self):
        self._text = ""


def patch_immediate_first_chunk(transport):
    """Send first audio chunk immediately with zero delay."""
    output = transport.output()
    output._send_interval = 0
    output._first_chunk_sent = False

    _orig_write = output.write_audio_frame

    async def _write_immediate(frame):
        if not output._first_chunk_sent:
            output._first_chunk_sent = True
            output._next_send_time = time.monotonic() - 0.001
        await _orig_write(frame)

    output.write_audio_frame = _write_immediate

    _orig_process = output.process_frame

    async def _reset_on_tts(frame, direction):
        if isinstance(frame, TTSStartedFrame):
            output._first_chunk_sent = False
        await _orig_process(frame, direction)

    output.process_frame = _reset_on_tts


# ============================================================================
# PIPELINE
# ============================================================================

def _get_sample_rate() -> int:
    return int(os.getenv("SAMPLE_RATE", "8000"))


async def run_pipeline(
    transport: FastAPIWebsocketTransport,
    system_prompt: str,
    greeting: str,
    llm_config: dict,
    stt_config: dict,
    tts_config: dict,
    audiobuffer: AudioBufferProcessor,
    transcript: TranscriptProcessor,
    vad_analyzer: Any = None,
    max_duration_seconds: int = 600,
) -> None:
    """Run the Pipecat voice pipeline."""
    sample_rate = _get_sample_rate()

    try:
        llm = create_llm_service(llm_config)
        stt = create_stt_service(stt_config, sample_rate, vad_analyzer=vad_analyzer)
        tts = create_tts_service(tts_config, sample_rate)

        # Fast punctuation aggregation for lower TTS latency
        tts._aggregate_sentences = True
        tts._text_aggregator = FastPunctuationAggregator()

        context = OpenAILLMContext([{"role": "system", "content": system_prompt}])

        user_params = getattr(llm, "_user_aggregator_params", None)
        if user_params:
            context_aggregator = llm.create_context_aggregator(context, user_params=user_params)
        else:
            context_aggregator = llm.create_context_aggregator(context)

        greeting_filter = GreetingInterruptionFilter()

        pipeline = Pipeline([
            transport.input(),
            greeting_filter,
            stt,
            transcript.user(),
            context_aggregator.user(),
            llm,
            tts,
            transcript.assistant(),
            audiobuffer,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=True),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            await audiobuffer.start_recording()
            if greeting and greeting.strip():
                greeting_filter.start_greeting()
                await task.queue_frames([TTSSpeakFrame(greeting)])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)

    except Exception as e:
        logger.error(f"Pipeline error: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# CALL ENTRY POINT
# ============================================================================

async def handle_call(
    websocket_client,
    stream_sid: str,
    call_sid: str,
    call_config: dict,
) -> dict:
    """Main entry point for a voice call.

    Args:
        websocket_client: WebSocket connection from telephony provider
        stream_sid: Stream identifier
        call_sid: Call identifier
        call_config: Call configuration from the calling app:
            {
                "systemPrompt": "You are Mira...",
                "variables": {"userName": "Karthik", ...},
                "greeting": "Hey Karthik, it's Mira.",
                "llm": {"provider": "openai", "model": "gpt-4o-mini"},
                "stt": {"provider": "deepgram", "language": "English", "args": {"keywords": [...]}},
                "tts": {"provider": "cartesia", "args": {"voice_id": "..."}},
                "maxDurationSeconds": 300,
            }

    Returns:
        dict with transcript_lines and call duration
    """
    sample_rate = _get_sample_rate()
    max_duration = call_config.get("maxDurationSeconds", 600)
    call_start = time.monotonic()
    start_time_utc = datetime.now(timezone.utc).isoformat()

    # Inject variables into system prompt
    system_prompt = call_config.get("systemPrompt", "")
    variables = call_config.get("variables", {})
    if variables:
        system_prompt = inject_variables(system_prompt, variables)

    greeting = call_config.get("greeting", "")

    # Provider configs
    llm_config = call_config.get("llm", {"provider": "openai", "model": "gpt-4o-mini"})
    stt_config = call_config.get("stt", {"provider": "deepgram", "language": "English"})
    tts_config = call_config.get("tts", {"provider": "cartesia"})

    # Transport setup
    serializer = VobizFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        params=VobizFrameSerializer.InputParams(
            vobiz_sample_rate=sample_rate,
            sample_rate=sample_rate,
        ),
    )

    vad_analyzer = SileroVADAnalyzer(
        sample_rate=sample_rate,
        params=VADParams(
            stop_secs=0.35,
            min_volume=0.3,
            confidence=0.4,
            start_secs=0.1,
        ),
    )
    vad_analyzer._smoothing_factor = 0.1

    import pipecat.transports.base_input
    pipecat.transports.base_input.AUDIO_INPUT_TIMEOUT_SECS = 0.1
    import pipecat.transports.base_output
    pipecat.transports.base_output.BOT_VAD_STOP_SECS = 0.2

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=vad_analyzer,
            serializer=serializer,
            audio_in_passthrough=True,
            session_timeout=max_duration,
            audio_out_10ms_chunks=2,
        ),
    )

    patch_immediate_first_chunk(transport)

    audiobuffer = AudioBufferProcessor()
    transcript = TranscriptProcessor()

    # Accumulate transcript in memory
    transcript_lines: list[str] = []

    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        for message in frame.messages:
            ts = f"[{message.timestamp}] " if message.timestamp else ""
            line = f"{ts}{message.role}: {message.content}"
            transcript_lines.append(line)

    try:
        await run_pipeline(
            transport=transport,
            system_prompt=system_prompt,
            greeting=greeting,
            llm_config=llm_config,
            stt_config=stt_config,
            tts_config=tts_config,
            audiobuffer=audiobuffer,
            transcript=transcript,
            vad_analyzer=vad_analyzer,
            max_duration_seconds=max_duration,
        )
    finally:
        duration = time.monotonic() - call_start
        logger.info(f"Call {call_sid} ended after {duration:.1f}s, {len(transcript_lines)} transcript lines")

    return {
        "callId": call_sid,
        "transcript": "\n".join(transcript_lines),
        "transcriptLines": transcript_lines,
        "durationSeconds": int(time.monotonic() - call_start),
        "startedAt": start_time_utc,
        "endedAt": datetime.now(timezone.utc).isoformat(),
    }
