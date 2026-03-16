"""Latency optimizations — SOXR patch, fast punctuation, immediate first chunk."""

import re
import time
from loguru import logger
from pipecat.frames.frames import TTSStartedFrame
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator, Aggregation, AggregationType


def inject_variables(template: str, variables: dict[str, str]) -> str:
    """Replace {{variableName}} placeholders with values."""
    def replacer(match: re.Match) -> str:
        key = match.group(1).strip()
        return variables.get(key, match.group(0))
    return re.sub(r'\{\{(\s*\w+\s*)\}\}', replacer, template)


def patch_soxr_resampler():
    """Monkey-patch SOXR resampler for Quick Quality (lower latency)."""
    try:
        from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
        import soxr

        def patched_initialize(self, in_rate: float, out_rate: float):
            self._in_rate = in_rate
            self._out_rate = out_rate
            self._last_resample_time = time.time()
            self._soxr_stream = soxr.ResampleStream(
                in_rate=in_rate, out_rate=out_rate, num_channels=1, quality="QQ", dtype="int16",
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
            if char in ".!?,":
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
