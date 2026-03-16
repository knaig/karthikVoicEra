"""Minimal example — make a voice call with Gemini Flash."""

import asyncio
from voicera import VoiceCall


async def main():
    call = VoiceCall(
        phone="+919886974008",
        system_prompt="You are Mira, a friendly AI assistant. Keep responses to 1-2 sentences.",
        greeting="Hi! This is Mira. How can I help you today?",
        llm={"provider": "gemini", "model": "gemini-2.0-flash"},
        stt={"provider": "openai", "language": "English"},
        tts={"provider": "openai", "args": {"voice": "nova"}},
        max_duration=120,
    )

    @call.on("call_ended")
    def on_end(result):
        print(f"\nCall ended: {result.duration}s, {len(result.transcript_lines)} lines")
        print(result.transcript)

    result = await call.start()


if __name__ == "__main__":
    asyncio.run(main())
