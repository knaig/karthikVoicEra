"""Audio noise reduction and echo cancellation for phone calls.

Sits in the pipeline before VAD/STT to clean up phone line noise.
Uses noisereduce (stationary noise removal) for line hiss/HVAC/car noise.
"""

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

_noisereduce_available = False
try:
    import noisereduce as nr
    _noisereduce_available = True
except ImportError:
    logger.warning("noisereduce not installed — noise filtering disabled (pip install noisereduce)")


class NoiseGateFilter(FrameProcessor):
    """Simple noise gate + optional noisereduce for phone audio.

    Two-stage approach:
    1. Noise gate: zero out audio below a volume threshold (kills constant line hiss)
    2. noisereduce: stationary noise removal (kills HVAC, car noise, fan)

    Runs on every audio frame — must be fast.
    """

    def __init__(
        self,
        gate_threshold: float = 0.02,      # Volume below this is zeroed (0-1 scale)
        use_noisereduce: bool = True,       # Use noisereduce library
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gate_threshold = gate_threshold
        self._use_noisereduce = use_noisereduce and _noisereduce_available
        self._sample_rate = sample_rate
        self._noise_profile: np.ndarray | None = None
        self._frames_seen = 0
        self._profile_frames = 5  # Use first N frames to build noise profile

        if self._use_noisereduce:
            logger.info("NoiseGateFilter: noisereduce enabled")
        else:
            logger.info("NoiseGateFilter: using simple noise gate only")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, InputAudioRawFrame):
            await self.push_frame(frame, direction)
            return

        try:
            # Convert to float
            audio = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Build noise profile from first few frames (assumed to be silence/noise)
            self._frames_seen += 1
            if self._frames_seen <= self._profile_frames:
                if self._noise_profile is None:
                    self._noise_profile = audio.copy()
                else:
                    # Running average
                    self._noise_profile = 0.7 * self._noise_profile + 0.3 * audio[:len(self._noise_profile)]

            # Stage 1: Simple noise gate
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < self._gate_threshold:
                # Below threshold — output silence
                cleaned = np.zeros_like(audio)
            elif self._use_noisereduce and self._noise_profile is not None and self._frames_seen > self._profile_frames:
                # Stage 2: noisereduce stationary noise removal
                cleaned = nr.reduce_noise(
                    y=audio,
                    sr=self._sample_rate,
                    y_noise=self._noise_profile,
                    stationary=True,
                    prop_decrease=0.8,  # How much to reduce noise (0-1)
                    n_fft=512,          # Smaller FFT for lower latency
                    hop_length=128,
                )
            else:
                cleaned = audio

            # Convert back to int16
            cleaned_int16 = (np.clip(cleaned, -1.0, 1.0) * 32767).astype(np.int16)

            # Create new frame with cleaned audio
            new_frame = InputAudioRawFrame(
                audio=cleaned_int16.tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            await self.push_frame(new_frame, direction)

        except Exception as e:
            # On any error, pass through original audio
            logger.debug(f"NoiseGateFilter error (passing through): {e}")
            await self.push_frame(frame, direction)


class EchoCancellationFilter(FrameProcessor):
    """Simple echo suppression for phone calls.

    When TTS audio is playing (bot is speaking), suppress any audio
    that looks like echo by reducing gain. Not true AEC — but prevents
    the most common case where TTS audio feeds back through the phone
    and triggers false speech detection.

    Works with the greeting_filter: during greeting, all audio is suppressed.
    After greeting, monitors bot speaking state and reduces gain during overlap.
    """

    def __init__(self, suppression_factor: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._bot_speaking = False
        self._suppression_factor = suppression_factor
        logger.info(f"EchoCancellationFilter: suppression={suppression_factor}")

    def set_bot_speaking(self, speaking: bool):
        self._bot_speaking = speaking

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame) and self._bot_speaking:
            # Bot is speaking — suppress mic audio to prevent echo
            audio = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
            audio *= self._suppression_factor
            suppressed = audio.astype(np.int16)
            new_frame = InputAudioRawFrame(
                audio=suppressed.tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            await self.push_frame(new_frame, direction)
        else:
            await self.push_frame(frame, direction)
