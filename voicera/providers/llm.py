"""LLM service factory — OpenAI, Gemini, Anthropic."""

import os
from typing import Any

from loguru import logger
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams


def create_llm_service(config: dict) -> Any:
    """Create an LLM service from config dict.

    Config keys: provider, model, api_key, args
    """
    provider = (config.get("provider") or "openai").lower()
    model = config.get("model") or config.get("args", {}).get("model")
    api_key = config.get("api_key")
    args = config.get("args", {})

    if provider == "openai":
        user_aggregator_params = LLMUserAggregatorParams(
            aggregation_timeout=args.get("aggregation_timeout", 0.05)
        )
        service = OpenAILLMService(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model or "gpt-4o-mini",
        )
        service._user_aggregator_params = user_aggregator_params
        return service

    elif provider in ("gemini", "google"):
        return OpenAILLMService(
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
            model=model or "gemini-2.0-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    elif provider == "anthropic":
        return OpenAILLMService(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            model=model or "claude-sonnet-4-20250514",
            base_url="https://api.anthropic.com/v1/",
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
