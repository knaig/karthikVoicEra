"""Mira-style coaching call with variable injection."""

import asyncio
from voicera import VoiceCall


async def main():
    call = VoiceCall(
        phone="+919886974008",
        system_prompt="""You are Mira, an executive leadership coach.
You help {{userName}} develop their leadership skills through reflective conversation.
You are warm, insightful, and ask powerful questions.
Keep responses concise (2-3 sentences).
Focus on active listening and helping them find their own answers.
Current coaching focus: {{coachingFocus}}.""",
        greeting="Hi {{userName}}, it's Mira, your leadership coach. How are you doing today?",
        variables={
            "userName": "Karthik",
            "coachingFocus": "delegation and trust-building with direct reports",
        },
        llm={"provider": "gemini", "model": "gemini-2.0-flash"},
        stt={"provider": "openai", "language": "English"},
        tts={"provider": "openai", "args": {"voice": "nova"}},
        max_duration=300,
    )

    @call.on("call_ended")
    def on_end(result):
        print(f"\nCoaching session ended: {result.duration}s")
        print(f"Transcript:\n{result.transcript}")

    await call.start()


if __name__ == "__main__":
    asyncio.run(main())
