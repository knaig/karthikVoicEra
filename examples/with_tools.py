"""Function calling example — tools run in the caller's process."""

import asyncio
import json
from voicera import VoiceCall


async def main():
    call = VoiceCall(
        phone="+919886974008",
        system_prompt="""You are Mira, an executive leadership coach.
You have access to tools to look up stakeholder information and manage commitments.
Use them when the user mentions names or wants to create action items.
Keep responses to 2-3 sentences.""",
        greeting="Hi Karthik, it's Mira. Let's work on your leadership goals today.",
        llm={"provider": "gemini", "model": "gemini-2.0-flash"},
        stt={"provider": "openai", "language": "English"},
        tts={"provider": "openai", "args": {"voice": "nova"}},
        max_duration=300,
    )

    @call.tool("lookup_stakeholder")
    def lookup_stakeholder(name: str) -> str:
        """Look up a stakeholder's profile and relationship context."""
        # In production, this would query your database
        profiles = {
            "ravi": {"name": "Ravi Kumar", "role": "VP Engineering", "relationship": "direct report"},
            "priya": {"name": "Priya Shah", "role": "CTO", "relationship": "skip-level manager"},
        }
        match = profiles.get(name.lower(), {"name": name, "role": "Unknown", "relationship": "Unknown"})
        return json.dumps(match)

    @call.tool("create_commitment")
    def create_commitment(text: str, due_date: str) -> str:
        """Create a new commitment or action item from the conversation."""
        print(f"[COMMITMENT] {text} (due: {due_date})")
        return f"Commitment created: {text}"

    @call.on("transcript_update")
    def on_transcript(message):
        print(f"  {message['role']}: {message['content']}")

    @call.on("call_ended")
    def on_end(result):
        print(f"\nCall ended: {result.duration}s")

    await call.start()


if __name__ == "__main__":
    asyncio.run(main())
