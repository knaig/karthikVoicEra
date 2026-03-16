"""Service factory — creates LLM, STT, and TTS services from config.

Supports 6 STT providers, 5 TTS providers, and 3 LLM providers.
Add a new provider = add one elif block.
"""

import os
from typing import Any, Optional

from loguru import logger
from deepgram import LiveOptions

# Pipecat built-in services
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

# Custom Indian language services
from services.ai4bharat.tts import IndicParlerRESTTTSService
from services.ai4bharat.stt import IndicConformerRESTSTTService
from services.bhashini.stt import BhashiniSTTService
from services.bhashini.tts import BhashiniTTSService

from config.stt_mappings import STT_LANGUAGE_MAP
from config.tts_mappings import TTS_LANGUAGE_MAP


class ServiceCreationError(Exception):
    pass


# ============================================================================
# LLM
# ============================================================================

def create_llm_service(llm_config: dict) -> Any:
    provider = (llm_config.get("provider") or llm_config.get("name") or "openai").lower()
    model = llm_config.get("model") or llm_config.get("args", {}).get("model")
    args = llm_config.get("args", {})

    if provider == "openai":
        user_aggregator_params = LLMUserAggregatorParams(
            aggregation_timeout=args.get("aggregation_timeout", 0.05)
        )
        service = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model or "gpt-4o-mini",
        )
        service._user_aggregator_params = user_aggregator_params
        return service

    elif provider == "gemini" or provider == "google":
        # Use OpenAI-compatible endpoint for Gemini
        service = OpenAILLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=model or "gemini-2.0-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return service

    elif provider == "anthropic":
        # Anthropic via OpenAI-compatible proxy or direct
        service = OpenAILLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=model or "claude-sonnet-4-20250514",
            base_url="https://api.anthropic.com/v1/",
        )
        return service

    else:
        raise ServiceCreationError(f"Unknown LLM provider: {provider}")


# ============================================================================
# STT
# ============================================================================

def create_stt_service(stt_config: dict, sample_rate: int, vad_analyzer: Any = None) -> Any:
    provider = (stt_config.get("name") or stt_config.get("provider") or "deepgram").lower()
    language = stt_config.get("language", "English")
    args = stt_config.get("args", {})

    # Normalize provider name for language map lookup
    provider_key = {
        "deepgram": "Deepgram", "google": "Google", "openai": "OpenAI",
        "sarvam": "Sarvam", "ai4bharat": "AI4Bharat", "bhashini": "Bhashini",
    }.get(provider, provider)

    if provider == "deepgram":
        return DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            sample_rate=sample_rate,
            live_options=LiveOptions(
                model=args.get("model", "nova-2"),
                language=STT_LANGUAGE_MAP.get(provider_key, {}).get(language, "en-US"),
                channels=1,
                encoding="linear16",
                sample_rate=sample_rate,
                interim_results=True,
                endpointing=150,
                smart_format=True,
                punctuate=True,
                keywords=args.get("keywords", []),
            )
        )

    elif provider == "google":
        return GoogleSTTService(
            credentials_path=os.getenv("GOOGLE_STT_CREDENTIALS_PATH", "credentials/google_stt.json"),
            sample_rate=sample_rate,
            params=GoogleSTTService.InputParams(
                languages=[STT_LANGUAGE_MAP.get(provider_key, {}).get(language, "en-US")]
            )
        )

    elif provider == "openai":
        return OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            language=STT_LANGUAGE_MAP.get(provider_key, {}).get(language, "en"),
        )

    elif provider == "ai4bharat":
        return IndicConformerRESTSTTService(
            language_id=STT_LANGUAGE_MAP.get(provider_key, {}).get(language, "hi"),
            sample_rate=16000,
            input_sample_rate=sample_rate,
            vad_analyzer=vad_analyzer,
        )

    elif provider == "bhashini":
        return BhashiniSTTService(
            api_key=os.getenv("BHASHINI_API_KEY"),
            language=STT_LANGUAGE_MAP.get(provider_key, {}).get(language, "hi"),
            service_id=args.get("model", "bhashini/ai4bharat/conformer-multilingual-asr"),
            sample_rate=sample_rate,
        )

    elif provider == "sarvam":
        return SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            language=STT_LANGUAGE_MAP.get(provider_key, {}).get(language, "hi-IN"),
            model=args.get("model"),
            sample_rate=sample_rate,
        )

    else:
        raise ServiceCreationError(f"Unknown STT provider: {provider}")


# ============================================================================
# TTS
# ============================================================================

def create_tts_service(tts_config: dict, sample_rate: int) -> Any:
    provider = (tts_config.get("name") or tts_config.get("provider") or "cartesia").lower()
    language = tts_config.get("language", "English")
    args = tts_config.get("args", {})

    provider_key = {
        "cartesia": "Cartesia", "google": "Google", "openai": "OpenAI",
        "sarvam": "Sarvam", "ai4bharat": "AI4Bharat", "bhashini": "Bhashini",
    }.get(provider, provider)

    if provider == "cartesia":
        return CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            model=args.get("model"),
            encoding="pcm_s16le",
            voice_id=args.get("voice_id"),
        )

    elif provider == "google":
        return GoogleTTSService(
            credentials_path=os.getenv("GOOGLE_TTS_CREDENTIALS_PATH", "credentials/google_tts.json"),
            voice_id=args.get("voice_id"),
            params=GoogleTTSService.InputParams(
                language=TTS_LANGUAGE_MAP.get(provider_key, {}).get(language, "en-US")
            )
        )

    elif provider == "openai":
        return OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice=args.get("voice", "alloy"),
        )

    elif provider == "ai4bharat":
        return IndicParlerRESTTTSService(
            speaker=args.get("speaker", "Divya"),
            description=args.get("description", "A clear, natural voice with good audio quality."),
            sample_rate=sample_rate,
        )

    elif provider == "bhashini":
        return BhashiniTTSService(
            speaker=args.get("speaker", "Divya"),
            description=args.get("description", "A clear, natural voice with good audio quality."),
            sample_rate=44100,
        )

    elif provider == "sarvam":
        return SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            target_language_code=TTS_LANGUAGE_MAP.get(provider_key, {}).get(language, "hi-IN"),
            model=args.get("model"),
            speaker=args.get("speaker"),
            pitch=args.get("pitch"),
            pace=args.get("pace"),
            loudness=args.get("loudness"),
        )

    else:
        raise ServiceCreationError(f"Unknown TTS provider: {provider}")
