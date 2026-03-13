"""Generation helpers for ParlerTTSForConditionalGeneration (encoder/decoder prep, cache, etc.)."""

import copy
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import BaseModelOutput


def get_decoder_start_token_id(
    model,
    decoder_start_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
) -> int:
    """Resolve decoder start token id from config or args."""
    decoder_start_token_id = (
        decoder_start_token_id
        or getattr(model.generation_config, "decoder_start_token_id", None)
    )
    bos_token_id = bos_token_id or getattr(model.generation_config, "bos_token_id", None)
    if decoder_start_token_id is not None:
        return decoder_start_token_id
    if bos_token_id is not None:
        return bos_token_id
    raise ValueError("decoder_start_token_id or bos_token_id must be set for encoder-decoder generation.")


def maybe_initialize_input_ids_for_generation(
    model,
    inputs: Optional[torch.Tensor],
    bos_token_id: Optional[int],
    model_kwargs: Dict[str, Any],
) -> torch.LongTensor:
    """Return input_ids for generation; use dummy -100 if encoder_outputs already provided."""
    if inputs is not None:
        return inputs
    encoder_outputs = model_kwargs.get("encoder_outputs")
    if encoder_outputs is not None:
        shape = encoder_outputs[0].size()[:-1]
        return torch.ones(shape, dtype=torch.long, device=model.device) * -100
    if bos_token_id is None:
        raise ValueError("bos_token_id must be defined when input_ids are not provided.")
    batch_size = 1
    for v in model_kwargs.values():
        if isinstance(v, torch.Tensor):
            batch_size = v.shape[0]
            break
    return torch.ones((batch_size, 1), dtype=torch.long, device=model.device) * bos_token_id


def prepare_decoder_input_ids_for_generation(
    model,
    batch_size: int,
    model_input_name: str,
    model_kwargs: Dict[str, Any],
    decoder_start_token_id: Optional[int],
    bos_token_id: Optional[int],
    device: Optional[torch.device],
) -> Tuple[torch.LongTensor, Dict[str, Any]]:
    """Build decoder input ids for first step; prepend decoder_start_token_id and prompt embeddings."""
    if model_kwargs.get("decoder_input_ids") is not None:
        decoder_input_ids = model_kwargs.pop("decoder_input_ids")
    elif "input_ids" in model_kwargs and model_input_name != "input_ids":
        decoder_input_ids = model_kwargs.pop("input_ids")
    else:
        decoder_input_ids = None

    start_id = get_decoder_start_token_id(model, decoder_start_token_id, bos_token_id)
    device = device or model.device
    start_tensor = torch.ones(
        (batch_size * model.decoder.num_codebooks, 1), dtype=torch.long, device=device
    ) * start_id

    if decoder_input_ids is None:
        decoder_input_ids = start_tensor
    elif (decoder_input_ids[..., 0] != start_id).all().item():
        decoder_input_ids = torch.cat([start_tensor, decoder_input_ids], dim=-1)
        if "decoder_attention_mask" in model_kwargs:
            mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                (torch.ones_like(mask)[:, :1], mask), dim=-1
            )

    prompt_hidden_states = model_kwargs.get("prompt_hidden_states")
    num_codebooks = model.decoder.num_codebooks
    input_reshaped = decoder_input_ids.reshape(-1, num_codebooks, decoder_input_ids.shape[-1])
    inputs_embeds = sum(
        model.decoder.model.decoder.embed_tokens[c](input_reshaped[:, c])
        for c in range(num_codebooks)
    )
    if prompt_hidden_states is not None:
        inputs_embeds = torch.cat([prompt_hidden_states, inputs_embeds], dim=1)
    model_kwargs["inputs_embeds"] = inputs_embeds
    return decoder_input_ids, model_kwargs


def prepare_text_encoder_kwargs_for_generation(
    model,
    inputs_tensor: torch.Tensor,
    model_kwargs: Dict[str, Any],
    model_input_name: Optional[str],
    generation_config: GenerationConfig,
) -> Dict[str, Any]:
    """Run text encoder and store encoder_outputs in model_kwargs."""
    encoder = model.get_text_encoder()
    if hasattr(encoder, "_hf_hook"):
        encoder._hf_hook.io_same_device = True
    irrelevant = ["decoder_", "cross_attn", "prompt_", "use_cache", "labels"]
    encoder_kwargs = {
        k: v for k, v in model_kwargs.items()
        if not any(k.startswith(p) for p in irrelevant)
    }
    sig = set(inspect.signature(encoder.forward).parameters)
    if "kwargs" not in sig and "model_kwargs" not in sig:
        encoder_kwargs = {k: v for k, v in encoder_kwargs.items() if k in sig}
    encoder_kwargs["output_attentions"] = generation_config.output_attentions
    encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    model_input_name = model_input_name or model.text_encoder.main_input_name
    encoder_kwargs["return_dict"] = True
    encoder_kwargs[model_input_name] = inputs_tensor
    last_hidden = encoder(**encoder_kwargs).last_hidden_state
    encoder_hidden = last_hidden
    if (
        model.text_encoder.config.hidden_size != model.decoder.config.hidden_size
        and model.decoder.config.cross_attention_hidden_size is None
    ):
        encoder_hidden = model.enc_to_dec_proj(encoder_hidden)
    if model_kwargs.get("attention_mask") is not None:
        encoder_hidden = encoder_hidden * model_kwargs["attention_mask"][..., None]
    model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=encoder_hidden)
    return model_kwargs


def prepare_prompt_kwargs_for_generation(model, prompt_input_ids: torch.Tensor, model_kwargs: Dict) -> Dict:
    """Embed prompt tokens and set prompt_hidden_states in model_kwargs."""
    model_kwargs["prompt_hidden_states"] = model.embed_prompts(prompt_input_ids)
    return model_kwargs


def prepare_audio_encoder_kwargs_for_generation(
    model,
    input_values: torch.Tensor,
    model_kwargs: Dict[str, Any],
    model_input_name: Optional[str],
) -> Dict[str, Any]:
    """Run audio encoder and set decoder_input_ids (and optional audio_scales) in model_kwargs."""
    encoder = model.get_audio_encoder()
    if hasattr(encoder, "_hf_hook"):
        encoder._hf_hook.io_same_device = True
    irrelevant = ["decoder_", "cross_attn", "use_cache"]
    encoder_kwargs = {
        k: v for k, v in model_kwargs.items()
        if not any(k.startswith(p) for p in irrelevant)
    }
    sig = set(inspect.signature(encoder.forward).parameters)
    if "kwargs" not in sig and "model_kwargs" not in sig:
        encoder_kwargs = {k: v for k, v in encoder_kwargs.items() if k in sig}
    model_input_name = model_input_name or model.audio_encoder.main_input_name
    encoder_kwargs["return_dict"] = True
    if "num_quantizers" in sig:
        encoder_kwargs["num_quantizers"] = model.config.decoder.num_codebooks
    elif "num_codebooks" in sig:
        encoder_kwargs["num_codebooks"] = model.config.decoder.num_codebooks
    elif "n_quantizers" in sig:
        encoder_kwargs["n_quantizers"] = model.config.decoder.num_codebooks
    encoder_kwargs[model_input_name] = input_values
    out = encoder.encode(**encoder_kwargs)
    audio_codes = out.audio_codes
    audio_scales = getattr(out, "audio_scales", None)
    if audio_codes.ndim == 3:
        bsz, _, seq_len = audio_codes.shape
        decoder_input_ids = audio_codes.reshape(bsz * model.decoder.num_codebooks, seq_len)
    else:
        frames, bsz, _, seq_len = audio_codes.shape
        if frames != 1:
            raise ValueError(
                "Expected 1 frame from audio encoder; set chunk_length=None to disable chunking."
            )
        decoder_input_ids = audio_codes[0, ...].reshape(bsz * model.decoder.num_codebooks, seq_len)
    model_kwargs["decoder_input_ids"] = decoder_input_ids
    if audio_scales is not None:
        model_kwargs["audio_scales"] = audio_scales
    return model_kwargs


def get_cache(
    model,
    cache_classes_mapping: Dict[str, type],
    cache_implementation: str,
    max_batch_size: int,
    max_cache_len: int,
    model_kwargs: Dict[str, Any],
) -> Cache:
    """Create or reuse KV cache for generation (static/sliding_window)."""
    cache_cls = cache_classes_mapping[cache_implementation]
    needs_cross = getattr(model.config, "is_encoder_decoder", True) or model_kwargs.get("encoder_outputs") is not None
    cache_to_check = None
    if hasattr(model, "_cache"):
        cache_to_check = model._cache.self_attention_cache if needs_cross else model._cache
    if cache_implementation == "sliding_window":
        max_cache_len = min(getattr(model.config, "sliding_window", max_cache_len), max_cache_len)
    need_new = (
        not hasattr(model, "_cache")
        or cache_to_check is None
        or not isinstance(cache_to_check, cache_cls)
        or getattr(cache_to_check, "max_batch_size", 0) != max_batch_size
        or getattr(cache_to_check, "max_cache_len", 0) < max_cache_len
    )
    if needs_cross and hasattr(model, "_cache"):
        enc_len = model_kwargs["encoder_outputs"][0].shape[1]
        need_new = need_new or model._cache.cross_attention_cache.max_cache_len != enc_len
    if need_new:
        cache_dtype = getattr(model.config, "_pre_quantization_dtype", model.dtype)
        cache_kwargs = {
            "config": model.config.decoder,
            "max_batch_size": max_batch_size,
            "max_cache_len": max_cache_len,
            "device": model.device,
            "dtype": cache_dtype,
        }
        model._cache = cache_cls(**cache_kwargs)
        if needs_cross:
            enc_kw = cache_kwargs.copy()
            enc_kw["max_cache_len"] = model_kwargs["encoder_outputs"][0].shape[1]
            cfg_cross = copy.deepcopy(model.config.decoder)
            cfg_cross.update({"num_key_value_heads": model.config.decoder.num_cross_attention_key_value_heads})
            enc_kw["config"] = cfg_cross
            model._cache = EncoderDecoderCache(model._cache, cache_cls(**enc_kw))
    else:
        model._cache.reset()
    return model._cache


def get_initial_cache_position(model, input_ids: torch.Tensor, model_kwargs: Dict) -> Dict:
    """Set cache_position in model_kwargs for pre-fill step."""
    from transformers.utils import is_torchdynamo_compiling
    if "inputs_embeds" in model_kwargs:
        cache_position = torch.ones_like(
            model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64
        ).cumsum(0) - 1
    else:
        cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
    past_length = 0
    if model_kwargs.get("past_key_values") is not None:
        cache = model_kwargs["past_key_values"]
        if not isinstance(cache, Cache):
            past_length = cache[0][0].shape[2]
        elif getattr(cache, "get_seq_length", None) is not None:
            past_length = cache.get_seq_length()
        if not is_torchdynamo_compiling():
            cache_position = cache_position[past_length:]
    model_kwargs["cache_position"] = cache_position
    return model_kwargs
