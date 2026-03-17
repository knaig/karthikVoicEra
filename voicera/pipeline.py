"""Pipecat pipeline wiring — STT → LLM → TTS with production-grade call quality.

Call quality engineering:
- Smart Turn v3: ML-based end-of-turn detection (not raw VAD)
- Silero VAD: telephony-tuned params from Silero's repo
- 300ms minimum speech gate: filters phone clicks/noise
- Deepgram utterance_end_ms: noise-resistant endpointing
- Voicemail detection: STT-based "leave a message" matching
"""

import time
import traceback
from typing import Any, Callable
from datetime import datetime, timezone

from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame, Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
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
from voicera.audio.noise_filter import NoiseGateFilter, EchoCancellationFilter
from voicera.audio.call_quality import CallQualityAdapter
from voicera.telephony.vobiz import VobizFrameSerializer

# Apply SOXR patch once on import
patch_soxr_resampler()

# Try to import Smart Turn v3 (optional — falls back to raw VAD if not available)
_smart_turn_available = False
try:
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    _smart_turn_available = True
    logger.info("Smart Turn v3 available — using ML-based turn detection")
except ImportError:
    logger.warning("Smart Turn v3 not available — using raw VAD (install pipecat-ai[smart-turn])")


# ============================================================================
# VOICEMAIL DETECTOR
# ============================================================================

class VoicemailDetector(FrameProcessor):
    """Detect voicemail greetings and end the call.

    Listens for the first 6 seconds of speech. If the transcription contains
    voicemail phrases, cancels the pipeline.
    """

    VOICEMAIL_PHRASES = [
        "leave a message", "leave your message", "after the tone",
        "after the beep", "not available", "cannot take your call",
        "please record", "voicemail", "mailbox",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._call_start = time.monotonic()
        self._detection_window = 8.0  # seconds
        self._detected = False
        self._first_user_speech = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Only check during the first N seconds
        elapsed = time.monotonic() - self._call_start
        if elapsed > self._detection_window or self._detected:
            await self.push_frame(frame, direction)
            return

        # Check transcription frames for voicemail phrases
        if isinstance(frame, TranscriptionFrame) and frame.text:
            text_lower = frame.text.lower()
            for phrase in self.VOICEMAIL_PHRASES:
                if phrase in text_lower:
                    logger.info(f"Voicemail detected: '{frame.text}' — ending call")
                    self._detected = True
                    # Don't push the frame — let the pipeline end
                    return

        await self.push_frame(frame, direction)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

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
    """Run the full voice pipeline with production-grade call quality."""
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

    # ================================================================
    # VAD — Silero telephony-tuned params
    # Source: Silero VAD repo recommended presets for 8kHz telephony
    # ================================================================
    vad_analyzer = SileroVADAnalyzer(
        sample_rate=16000,  # Silero + Smart Turn expect 16kHz (serializer upsamples)
        params=VADParams(
            confidence=0.6,     # Silero confidence threshold
            start_secs=0.3,     # 300ms min speech — filters phone clicks/noise
            stop_secs=0.5,      # 500ms silence before considering speech ended
            min_volume=0.5,     # Filters low-level line noise
        ),
    )

    # Reduce audio timeouts for responsiveness
    import pipecat.transports.base_input
    pipecat.transports.base_input.AUDIO_INPUT_TIMEOUT_SECS = 0.2
    import pipecat.transports.base_output
    pipecat.transports.base_output.BOT_VAD_STOP_SECS = 0.3

    # ================================================================
    # Transport
    # ================================================================
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
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

    # ================================================================
    # Services
    # ================================================================
    llm = create_llm_service(llm_config)
    stt = create_stt_service(stt_config, sample_rate, vad_analyzer=vad_analyzer)
    tts = create_tts_service(tts_config, sample_rate)

    # Fast punctuation for lower TTS latency
    tts._aggregate_sentences = True
    tts._text_aggregator = FastPunctuationAggregator()

    # ================================================================
    # LLM context with tools
    # ================================================================
    messages = [{"role": "system", "content": system_prompt}]
    context = OpenAILLMContext(messages, tools=tools or None)

    # Smart Turn: use ML-based turn detection if available
    if _smart_turn_available:
        try:
            smart_turn = LocalSmartTurnAnalyzerV3()
            from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams
            user_params = LLMUserAggregatorParams(
                smart_turn_analyzer=smart_turn,
                aggregation_timeout=3.0,  # Max wait after VAD silence before forcing end-of-turn
            )
            context_aggregator = llm.create_context_aggregator(context, user_params=user_params)
            logger.info("Using Smart Turn v3 for turn detection")
        except Exception as e:
            logger.warning(f"Smart Turn init failed, using default: {e}")
            user_params = getattr(llm, "_user_aggregator_params", None)
            if user_params:
                context_aggregator = llm.create_context_aggregator(context, user_params=user_params)
            else:
                context_aggregator = llm.create_context_aggregator(context)
    else:
        user_params = getattr(llm, "_user_aggregator_params", None)
        if user_params:
            context_aggregator = llm.create_context_aggregator(context, user_params=user_params)
        else:
            context_aggregator = llm.create_context_aggregator(context)

    # Register tool handler on LLM if tools provided
    if tools and tool_handler:
        for tool_def in tools:
            func_name = tool_def["function"]["name"]

            async def _make_handler(fn=func_name):
                async def _handler(function_name, tool_call_id, arguments, llm_instance, context, result_callback):
                    result = await tool_handler(fn, arguments)
                    await result_callback(result)
                return _handler

            llm.register_function(func_name, await _make_handler(func_name))

    # ================================================================
    # Pipeline processors
    # ================================================================
    greeting_filter = GreetingInterruptionFilter()
    voicemail_detector = VoicemailDetector()
    noise_filter = NoiseGateFilter(gate_threshold=0.02, sample_rate=16000)
    echo_filter = EchoCancellationFilter(suppression_factor=0.1)
    quality_adapter = CallQualityAdapter(
        vad_analyzer=vad_analyzer,
        noise_filter=noise_filter,
        echo_filter=echo_filter,
        sample_rate=16000,
    )
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

    # ================================================================
    # Pipeline — order matters
    # ================================================================
    pipeline = Pipeline([
        transport.input(),
        quality_adapter,         # Measure SNR, auto-adjust VAD/noise/echo params
        noise_filter,            # Remove phone line hiss/noise before VAD sees it
        echo_filter,             # Suppress echo when bot is speaking
        greeting_filter,         # Block interruptions during greeting
        stt,                     # Deepgram STT (with utterance_end_ms)
        voicemail_detector,      # Detect voicemail in first 8 seconds
        transcript.user(),
        context_aggregator.user(),  # Smart Turn decides end-of-turn here
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
        echo_filter.set_bot_speaking(True)  # Suppress echo during greeting
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

    # Check if voicemail was detected
    voicemail = voicemail_detector._detected if voicemail_detector else False

    return {
        "call_id": call_sid,
        "transcript": "\n".join(transcript_lines),
        "transcript_lines": transcript_lines,
        "duration": int(duration),
        "started_at": start_time_utc,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "voicemail_detected": voicemail,
    }
