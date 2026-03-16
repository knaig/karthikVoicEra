"""Pipecat pipeline wiring — STT → LLM → TTS with tool calling."""

import time
import traceback
from typing import Any, Callable
from datetime import datetime, timezone

from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams, FastAPIWebsocketTransport,
)

from voicera.providers.llm import create_llm_service
from voicera.providers.stt import create_stt_service
from voicera.providers.tts import create_tts_service
from voicera.audio.greeting_filter import GreetingInterruptionFilter
from voicera.audio.optimizations import (
    patch_soxr_resampler, FastPunctuationAggregator, patch_immediate_first_chunk,
)
from voicera.telephony.vobiz import VobizFrameSerializer

# Apply SOXR patch once on import
patch_soxr_resampler()


async def run_voice_pipeline(
    websocket,
    stream_sid: str,
    call_sid: str,
    system_prompt: str,
    greeting: str,
    llm_config: dict,
    stt_config: dict,
    tts_config: dict,
    tools: list[dict] | None = None,
    tool_handler: Callable | None = None,
    on_transcript_update: Callable | None = None,
    max_duration: int = 600,
    sample_rate: int = 8000,
    serializer=None,
) -> dict:
    """Run the full voice pipeline and return call results.

    Args:
        tools: OpenAI function-calling tool schemas
        tool_handler: async callable(name, args) -> str for tool execution
        on_transcript_update: async callable(message_dict) for live transcript
    """
    call_start = time.monotonic()
    start_time_utc = datetime.now(timezone.utc).isoformat()

    # Serializer
    if serializer is None:
        serializer = VobizFrameSerializer(
            stream_sid=stream_sid, call_sid=call_sid,
            params=VobizFrameSerializer.InputParams(
                vobiz_sample_rate=sample_rate, sample_rate=sample_rate,
            ),
        )

    # VAD
    vad_analyzer = SileroVADAnalyzer(
        sample_rate=sample_rate,
        params=VADParams(stop_secs=0.35, min_volume=0.3, confidence=0.4, start_secs=0.1),
    )
    vad_analyzer._smoothing_factor = 0.1

    # Reduce audio input/output delays
    import pipecat.transports.base_input
    pipecat.transports.base_input.AUDIO_INPUT_TIMEOUT_SECS = 0.1
    import pipecat.transports.base_output
    pipecat.transports.base_output.BOT_VAD_STOP_SECS = 0.2

    # Transport
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False,
            vad_analyzer=vad_analyzer, serializer=serializer,
            audio_in_passthrough=True, session_timeout=max_duration,
            audio_out_10ms_chunks=2,
        ),
    )
    patch_immediate_first_chunk(transport)

    # Services
    llm = create_llm_service(llm_config)
    stt = create_stt_service(stt_config, sample_rate, vad_analyzer=vad_analyzer)
    tts = create_tts_service(tts_config, sample_rate)

    # Fast punctuation for lower TTS latency
    tts._aggregate_sentences = True
    tts._text_aggregator = FastPunctuationAggregator()

    # LLM context with tools
    messages = [{"role": "system", "content": system_prompt}]
    context = OpenAILLMContext(messages, tools=tools or None)

    user_params = getattr(llm, "_user_aggregator_params", None)
    if user_params:
        context_aggregator = llm.create_context_aggregator(context, user_params=user_params)
    else:
        context_aggregator = llm.create_context_aggregator(context)

    # Register tool handler on LLM if tools provided
    if tools and tool_handler:
        for tool_def in tools:
            func_name = tool_def["function"]["name"]

            # Create a closure that captures func_name
            async def _make_handler(fn=func_name):
                async def _handler(function_name, tool_call_id, arguments, llm_instance, context, result_callback):
                    result = await tool_handler(fn, arguments)
                    await result_callback(result)
                return _handler

            llm.register_function(func_name, await _make_handler(func_name))

    greeting_filter = GreetingInterruptionFilter()
    audiobuffer = AudioBufferProcessor()
    transcript = TranscriptProcessor()

    # Transcript accumulation
    transcript_lines: list[str] = []

    @transcript.event_handler("on_transcript_update")
    async def _on_transcript(processor, frame):
        for message in frame.messages:
            ts = f"[{message.timestamp}] " if message.timestamp else ""
            line = f"{ts}{message.role}: {message.content}"
            transcript_lines.append(line)
            if on_transcript_update:
                try:
                    msg = {"role": message.role, "content": message.content, "timestamp": message.timestamp}
                    result = on_transcript_update(msg)
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    pass

    # Pipeline
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

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

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

    try:
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)
    except Exception as e:
        logger.error(f"Pipeline error: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())

    duration = time.monotonic() - call_start
    return {
        "call_id": call_sid,
        "transcript": "\n".join(transcript_lines),
        "transcript_lines": transcript_lines,
        "duration": int(duration),
        "started_at": start_time_utc,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
