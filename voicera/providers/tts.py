"""TTS service factory — Cartesia, Google, OpenAI, Sarvam, AI4Bharat, Bhashini."""

import os
from typing import Any

from loguru import logger

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.sarvam.tts import SarvamTTSService

from voicera.indian.language_maps import TTS_LANGUAGE_MAP


_PROVIDER_KEY = {
    "cartesia": "Cartesia", "google": "Google", "openai": "OpenAI",
    "sarvam": "Sarvam", "ai4bharat": "AI4Bharat", "bhashini": "Bhashini",
}


def create_tts_service(config: dict, sample_rate: int) -> Any:
    """Create a TTS service from config dict.

    Config keys: provider/name, language, api_key, voice_id, args
    """
    provider = (config.get("provider") or config.get("name") or "cartesia").lower()
    language = config.get("language", "English")
    api_key = config.get("api_key")
    args = config.get("args", {})
    voice_id = config.get("voice_id") or args.get("voice_id")
    pk = _PROVIDER_KEY.get(provider, provider)

    if provider == "cartesia":
        return CartesiaTTSService(
            api_key=api_key or os.getenv("CARTESIA_API_KEY"),
            model=args.get("model"), encoding="pcm_s16le", voice_id=voice_id,
        )

    elif provider == "google":
        return GoogleTTSService(
            credentials_path=args.get("credentials_path") or os.getenv("GOOGLE_TTS_CREDENTIALS_PATH", "credentials/google_tts.json"),
            voice_id=voice_id or args.get("voice_id"),
            params=GoogleTTSService.InputParams(
                language=TTS_LANGUAGE_MAP.get(pk, {}).get(language, "en-US")
            ),
        )

    elif provider == "openai":
        return OpenAITTSService(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            voice=args.get("voice", "alloy"),
        )

    elif provider == "ai4bharat":
        from voicera.indian.ai4bharat_tts import IndicParlerRESTTTSService
        return IndicParlerRESTTTSService(
            speaker=args.get("speaker", "Divya"),
            description=args.get("description", "A clear, natural voice with good audio quality."),
            sample_rate=sample_rate, server_url=args.get("server_url"),
        )

    elif provider == "bhashini":
        from voicera.indian.bhashini_tts import BhashiniTTSService
        return BhashiniTTSService(
            speaker=args.get("speaker", "Divya"),
            description=args.get("description", "A clear, natural voice with good audio quality."),
            sample_rate=44100, server_url=args.get("server_url"),
            auth_token=api_key or args.get("auth_token"),
        )

    elif provider == "sarvam":
        return SarvamTTSService(
            api_key=api_key or os.getenv("SARVAM_API_KEY"),
            target_language_code=TTS_LANGUAGE_MAP.get(pk, {}).get(language, "hi-IN"),
            model=args.get("model"), speaker=args.get("speaker"),
            pitch=args.get("pitch"), pace=args.get("pace"), loudness=args.get("loudness"),
        )

    else:
        raise ValueError(f"Unknown TTS provider: {provider}")
