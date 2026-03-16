"""STT service factory — Deepgram, Google, OpenAI, Sarvam, AI4Bharat, Bhashini."""

import os
from typing import Any, Optional

from loguru import logger

from voicera.indian.language_maps import STT_LANGUAGE_MAP


_PROVIDER_KEY = {
    "deepgram": "Deepgram", "google": "Google", "openai": "OpenAI",
    "sarvam": "Sarvam", "ai4bharat": "AI4Bharat", "bhashini": "Bhashini",
}


def create_stt_service(config: dict, sample_rate: int, vad_analyzer: Any = None) -> Any:
    """Create an STT service from config dict.

    Config keys: provider/name, language, api_key, args
    """
    provider = (config.get("provider") or config.get("name") or "deepgram").lower()
    language = config.get("language", "English")
    api_key = config.get("api_key")
    args = config.get("args", {})
    pk = _PROVIDER_KEY.get(provider, provider)

    if provider == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService
        try:
            from deepgram import LiveOptions
        except ImportError:
            from deepgram import DeepgramClientOptions as LiveOptions
        return DeepgramSTTService(
            api_key=api_key or os.getenv("DEEPGRAM_API_KEY"),
            sample_rate=sample_rate,
            live_options=LiveOptions(
                model=args.get("model", "nova-2"),
                language=STT_LANGUAGE_MAP.get(pk, {}).get(language, "en-US"),
                channels=1, encoding="linear16", sample_rate=sample_rate,
                interim_results=True, endpointing=150, smart_format=True,
                punctuate=True, keywords=args.get("keywords", []),
            ),
        )

    elif provider == "google":
        from pipecat.services.google.stt import GoogleSTTService
        return GoogleSTTService(
            credentials_path=args.get("credentials_path") or os.getenv("GOOGLE_STT_CREDENTIALS_PATH", "credentials/google_stt.json"),
            sample_rate=sample_rate,
            params=GoogleSTTService.InputParams(
                languages=[STT_LANGUAGE_MAP.get(pk, {}).get(language, "en-US")]
            ),
        )

    elif provider == "openai":
        from pipecat.services.openai.stt import OpenAISTTService
        return OpenAISTTService(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            language=STT_LANGUAGE_MAP.get(pk, {}).get(language, "en"),
        )

    elif provider == "ai4bharat":
        from voicera.indian.ai4bharat_stt import IndicConformerRESTSTTService
        return IndicConformerRESTSTTService(
            language_id=STT_LANGUAGE_MAP.get(pk, {}).get(language, "hi"),
            sample_rate=16000, input_sample_rate=sample_rate,
            server_url=args.get("server_url"), vad_analyzer=vad_analyzer,
        )

    elif provider == "bhashini":
        from voicera.indian.bhashini_stt import BhashiniSTTService
        return BhashiniSTTService(
            api_key=api_key or os.getenv("BHASHINI_API_KEY"),
            language=STT_LANGUAGE_MAP.get(pk, {}).get(language, "hi"),
            service_id=args.get("model", "bhashini/ai4bharat/conformer-multilingual-asr"),
            sample_rate=sample_rate,
        )

    elif provider == "sarvam":
        from pipecat.services.sarvam.stt import SarvamSTTService
        return SarvamSTTService(
            api_key=api_key or os.getenv("SARVAM_API_KEY"),
            language=STT_LANGUAGE_MAP.get(pk, {}).get(language, "hi-IN"),
            model=args.get("model"), sample_rate=sample_rate,
        )

    else:
        raise ValueError(f"Unknown STT provider: {provider}")
