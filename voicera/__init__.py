"""voicera — Make AI voice calls with tool calling support.

    from voicera import VoiceCall

    call = VoiceCall(
        phone="+919886974008",
        system_prompt="You are Mira, an AI coach...",
        llm={"provider": "gemini", "model": "gemini-2.0-flash"},
    )
    await call.start()
"""

from voicera.call import VoiceCall, CallResult

__version__ = "0.1.0"
__all__ = ["VoiceCall", "CallResult"]
