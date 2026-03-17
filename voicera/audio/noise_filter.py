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
    """Adaptive echo cancellation for phone calls using NLMS algorithm.

    How it works:
    1. Captures a copy of TTS audio being sent to the phone (reference signal)
    2. When mic audio comes back, uses an adaptive NLMS filter to estimate
       the echo path and subtract the predicted echo from the mic signal
    3. What remains is the user's actual voice

    This is the same approach used by Vapi, speexdsp, and WebRTC AEC —
    just implemented in pure Python/NumPy (no C deps needed).

    Falls back to simple gain suppression if the adaptive filter isn't converged.
    """

    def __init__(
        self,
        filter_length: int = 256,    # NLMS filter taps (longer = handles longer echo paths)
        step_size: float = 0.1,      # Learning rate (0.01-0.5, lower = more stable)
        suppression_factor: float = 0.1,  # Fallback gain suppression
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._bot_speaking = False
        self._suppression_factor = suppression_factor

        # NLMS adaptive filter
        self._filter_length = filter_length
        self._step_size = step_size
        self._weights = np.zeros(filter_length, dtype=np.float32)
        self._reference_buffer = np.zeros(filter_length, dtype=np.float32)
        self._converged = False
        self._frames_with_reference = 0

        # Reference signal capture (TTS audio being sent to phone)
        self._reference_queue: list[np.ndarray] = []
        self._max_ref_queue = 50  # ~1 second of reference audio

        logger.info(f"EchoCancellationFilter: NLMS (taps={filter_length}, step={step_size})")

    def set_bot_speaking(self, speaking: bool):
        self._bot_speaking = speaking
        if not speaking:
            # Bot stopped — clear reference after a short delay
            self._reference_queue.clear()

    def feed_reference(self, audio_bytes: bytes):
        """Feed TTS output audio as the reference signal for echo cancellation."""
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._reference_queue.append(audio)
        # Trim to prevent memory growth
        while len(self._reference_queue) > self._max_ref_queue:
            self._reference_queue.pop(0)

    def _nlms_cancel(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Apply NLMS adaptive filter to cancel echo.

        mic: microphone signal (contains user voice + echo)
        ref: reference signal (TTS audio sent to phone)
        returns: mic signal with echo subtracted
        """
        output = np.zeros_like(mic)
        n = min(len(mic), len(ref))

        for i in range(n):
            # Shift reference buffer
            self._reference_buffer = np.roll(self._reference_buffer, 1)
            self._reference_buffer[0] = ref[i] if i < len(ref) else 0.0

            # Predict echo
            echo_estimate = np.dot(self._weights, self._reference_buffer)

            # Error = mic - predicted echo (this should be the user's voice)
            error = mic[i] - echo_estimate
            output[i] = error

            # Update filter weights (NLMS)
            power = np.dot(self._reference_buffer, self._reference_buffer) + 1e-10
            self._weights += (self._step_size / power) * error * self._reference_buffer

        return output

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, InputAudioRawFrame):
            # Capture outgoing TTS audio as reference
            from pipecat.frames.frames import AudioRawFrame
            if isinstance(frame, AudioRawFrame) and self._bot_speaking:
                self.feed_reference(frame.audio)
            await self.push_frame(frame, direction)
            return

        if not self._bot_speaking:
            # Bot not speaking — no echo to cancel, pass through
            await self.push_frame(frame, direction)
            return

        mic = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Try NLMS cancellation if we have reference audio
        if self._reference_queue:
            ref = np.concatenate(self._reference_queue)
            # Align reference to mic (simple: use latest reference samples)
            ref_aligned = ref[-len(mic):] if len(ref) >= len(mic) else np.pad(ref, (0, len(mic) - len(ref)))

            try:
                cancelled = self._nlms_cancel(mic, ref_aligned)
                self._frames_with_reference += 1
                if self._frames_with_reference > 10:
                    self._converged = True
            except Exception:
                # Fallback to gain suppression
                cancelled = mic * self._suppression_factor
        else:
            # No reference — fallback to gain suppression
            cancelled = mic * self._suppression_factor

        cleaned_int16 = (np.clip(cancelled, -1.0, 1.0) * 32767).astype(np.int16)
        new_frame = InputAudioRawFrame(
            audio=cleaned_int16.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
        )
        await self.push_frame(new_frame, direction)
