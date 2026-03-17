"""Call quality adaptation — auto-adjust params based on audio conditions.

Computes SNR from first 3 seconds of audio and adjusts:
- VAD confidence threshold (noisy line = higher threshold)
- Noise filter aggressiveness
- Echo suppression factor

Also monitors ongoing call quality and adapts mid-call.
"""

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.audio.vad.vad_analyzer import VADParams


class CallQualityAdapter(FrameProcessor):
    """Monitors audio quality and adapts pipeline params in real-time.

    First 3 seconds: measures SNR to set initial thresholds.
    Ongoing: rolling SNR check every 10 seconds, adjusts if conditions change.
    """

    def __init__(
        self,
        vad_analyzer=None,
        noise_filter=None,
        echo_filter=None,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vad = vad_analyzer
        self._noise_filter = noise_filter
        self._echo_filter = echo_filter
        self._sample_rate = sample_rate

        # Calibration
        self._calibration_frames: list[np.ndarray] = []
        self._calibration_duration = 0.0
        self._calibration_target = 3.0  # seconds
        self._calibrated = False

        # Ongoing monitoring
        self._monitor_buffer: list[np.ndarray] = []
        self._monitor_duration = 0.0
        self._monitor_interval = 10.0  # re-check every 10s
        self._last_snr = None

    def _compute_snr(self, audio: np.ndarray) -> float:
        """Estimate SNR from audio. Higher = cleaner signal."""
        if len(audio) < 1600:
            return 15.0  # default mid-range

        # Simple SNR: ratio of signal power to noise floor
        # Use top 10% as "signal" and bottom 50% as "noise"
        abs_audio = np.abs(audio)
        sorted_vals = np.sort(abs_audio)
        n = len(sorted_vals)

        noise_floor = np.mean(sorted_vals[:n // 2]) + 1e-10
        signal_level = np.mean(sorted_vals[int(n * 0.9):]) + 1e-10

        snr_db = 20 * np.log10(signal_level / noise_floor)
        return float(np.clip(snr_db, 0, 40))

    def _adapt_params(self, snr_db: float):
        """Adjust pipeline params based on measured SNR."""
        if snr_db < 10:
            # Noisy line — be conservative
            vad_confidence = 0.7
            vad_min_volume = 0.6
            noise_gate = 0.04
            echo_suppression = 0.05
            quality = "noisy"
        elif snr_db < 20:
            # Moderate — balanced
            vad_confidence = 0.6
            vad_min_volume = 0.5
            noise_gate = 0.02
            echo_suppression = 0.1
            quality = "moderate"
        else:
            # Clean line — responsive
            vad_confidence = 0.5
            vad_min_volume = 0.3
            noise_gate = 0.01
            echo_suppression = 0.15
            quality = "clean"

        # Apply to VAD
        if self._vad:
            try:
                self._vad._params._confidence = vad_confidence
                self._vad._params._min_volume = vad_min_volume
            except Exception:
                pass

        # Apply to noise filter
        if self._noise_filter:
            try:
                self._noise_filter._gate_threshold = noise_gate
            except Exception:
                pass

        # Apply to echo filter
        if self._echo_filter:
            try:
                self._echo_filter._suppression_factor = echo_suppression
            except Exception:
                pass

        self._last_snr = snr_db
        logger.info(f"[CallQuality] SNR={snr_db:.1f}dB ({quality}) → VAD conf={vad_confidence}, vol={vad_min_volume}, gate={noise_gate}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            audio = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            frame_duration = len(audio) / self._sample_rate

            if not self._calibrated:
                # Calibration phase — collect first 3 seconds
                self._calibration_frames.append(audio)
                self._calibration_duration += frame_duration

                if self._calibration_duration >= self._calibration_target:
                    full_audio = np.concatenate(self._calibration_frames)
                    snr = self._compute_snr(full_audio)
                    self._adapt_params(snr)
                    self._calibrated = True
                    self._calibration_frames = []  # free memory
                    logger.info(f"[CallQuality] Initial calibration complete: SNR={snr:.1f}dB")

            else:
                # Ongoing monitoring
                self._monitor_buffer.append(audio)
                self._monitor_duration += frame_duration

                if self._monitor_duration >= self._monitor_interval:
                    full_audio = np.concatenate(self._monitor_buffer)
                    snr = self._compute_snr(full_audio)

                    # Only re-adapt if SNR changed significantly (>5dB)
                    if self._last_snr is None or abs(snr - self._last_snr) > 5:
                        self._adapt_params(snr)

                    self._monitor_buffer = []
                    self._monitor_duration = 0.0

        await self.push_frame(frame, direction)
