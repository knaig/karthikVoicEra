"""Web call handler — Pipecat pipeline over SmallWebRTC (browser WebRTC)."""
import asyncio
import time
import traceback
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
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from .services import create_llm_service, create_stt_service, create_tts_service
from .bot import inject_variables, FastPunctuationAggregator


async def handle_web_call(webrtc_connection, call_config: dict) -> dict:
    """Run a Pipecat pipeline over a SmallWebRTC connection (browser call).

    Args:
        webrtc_connection: SmallWebRTCConnection instance (from request handler callback).
        call_config: Same schema as phone call config — systemPrompt, variables,
                     greeting, llm, stt, tts, etc.
    Returns:
        dict with callId, transcript, durationSeconds, status
    """
    call_id = call_config.get("callId", f"web_{int(time.time() * 1000)}")
    start_time_utc = datetime.now(timezone.utc).isoformat()
    call_start = time.monotonic()

    # Inject variables into system prompt + greeting
    system_prompt = call_config.get("systemPrompt", "You are Mira, an AI executive coach.")
    variables = call_config.get("variables", {})
    if variables:
        system_prompt = inject_variables(system_prompt, variables)

    greeting = call_config.get("greeting", "")
    if variables and greeting:
        greeting = inject_variables(greeting, variables)

    llm_config = call_config.get("llm", {"provider": "gemini", "model": "gemini-2.0-flash"})
    stt_config = call_config.get("stt", {"provider": "deepgram", "language": "English"})
    tts_config = call_config.get("tts", {"provider": "openai", "args": {"voice": "nova"}})

    SAMPLE_RATE = 16000

    vad_analyzer = SileroVADAnalyzer(
        sample_rate=SAMPLE_RATE,
        params=VADParams(
            confidence=0.6,
            start_secs=0.3,
            stop_secs=0.5,
            min_volume=0.4,
        ),
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
        ),
    )

    audiobuffer = AudioBufferProcessor()
    transcript_processor = TranscriptProcessor()
    transcript_lines = []

    @transcript_processor.event_handler("on_transcript_update")
    async def on_update(processor, frame):
        for m in frame.messages:
            transcript_lines.append(f"{m.role}: {m.content}")

    try:
        llm = create_llm_service(llm_config)
        stt = create_stt_service(stt_config, SAMPLE_RATE, vad_analyzer=vad_analyzer)
        tts = create_tts_service(tts_config, SAMPLE_RATE)

        # Speed optimisation: aggregate sentences before sending to TTS
        tts._aggregate_sentences = True
        tts._text_aggregator = FastPunctuationAggregator()

        context = OpenAILLMContext([{"role": "system", "content": system_prompt}])
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                transcript_processor.user(),
                context_aggregator.user(),
                llm,
                tts,
                transcript_processor.assistant(),
                audiobuffer,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_client_connected")
        async def on_connected(transport, conn):
            await audiobuffer.start_recording()
            if greeting and greeting.strip():
                await task.queue_frames([TTSSpeakFrame(greeting)])
            logger.info(f"Web call {call_id}: client connected")

        @transport.event_handler("on_client_disconnected")
        async def on_disconnected(transport, conn):
            logger.info(f"Web call {call_id}: client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)

    except Exception as e:
        logger.error(f"Web call {call_id} error: {e}")
        logger.debug(traceback.format_exc())

    duration = int(time.monotonic() - call_start)
    logger.info(f"Web call {call_id} ended — {duration}s, {len(transcript_lines)} turns")

    return {
        "callId": call_id,
        "transcript": "\n".join(transcript_lines),
        "transcriptLines": transcript_lines,
        "durationSeconds": duration,
        "startedAt": start_time_utc,
        "endedAt": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
    }
