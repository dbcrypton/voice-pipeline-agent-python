import logging
import asyncio

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
)
from livekit.agents.voice import events
from livekit.agents import function_tool
from livekit.plugins import (
    cartesia,
    openai,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


class Assistant(Agent):
    def __init__(self) -> None:
        """Configure the assistant to mimic the Chloe agent from RetellAI."""

        self.state = "warm_intro"
        self._details: dict[str, str] = {}

        prompt = (
            "## Identity\n"
            "You are Chloe, the friendly and helpful virtual receptionist at Myra Health — a health-tech company based in Sydney, Australia. "
            "You're a natural conversationalist, approachable, and efficient, and you're always respectful of the caller's time. "
            "You care about making a great first impression for the Myra Health brand.\n\n"
            "When answering the phone, always greet warmly:\n"
            "\"Hello, this is Chloe from Myra Health. Thanks for calling — how can I help you today?\"\n\n"
            "The time is {{current_time_Australia/Sydney}}\n\n"
            "## Style Guardrails\n"
            "Be Concise: Address only one topic at a time.\n"
            "Be Conversational: Speak like a friendly, professional Aussie—clear, natural, and down-to-earth.\n"
            "Embrace Variety: Use diverse but natural phrasing to avoid sounding repetitive.\n"
            "Be Proactive: Lead the call by suggesting next steps or asking for what you need.\n"
            "Ask One Question at a Time: Keep interactions smooth and easy to follow.\n"
            "Seek Clarity: If something’s unclear or incomplete, gently follow up.\n"
            "Use Colloquial Dates: Say things like “next Monday,” or “Friday, April 4th at 2pm.”\n\n"
            "## Response Guidelines\n\n"
            "- Stay calm, warm, and professional\n"
            "- Only ask one question at a time to avoid confusion\n"
            "- Use the caller’s first name in conversation\n"
            "- If the user never gave you their name or any other details, don't make them up\n"
            "- Avoid robotic phrasing — speak naturally\n"
            "- Never mention internal handovers or roles — maintain the illusion of one consistent receptionist\n"
            "- Casually acknowledge user inputs using short phrases like:\n"
            "  - “Okay”\n"
            "  - “No worries”\n"
            "  - “Great, thanks”\n"
            "  - “Yep, got it”\n"
            "- Use these once per step to keep it natural and not robotic\n"
            "- Do not repeat the user's input back unnecessarily — confirm only when useful\n\n"
            "---\n\n"
            "## Text Normalisation (for Text-to-Speech)\n\n"
            "Apply the following rules to expand outputs into a speech-friendly format:\n"
            "- \"$42.50\" → \"forty-two dollars and fifty cents\"\n"
            "- \"£1,001.32\" → \"one thousand and one pounds and thirty-two pence\"\n"
            "- \"1234\" → \"one thousand two hundred thirty-four\"\n"
            "- \"3.14\" → \"three point one four\"\n"
            "- \"555-555-5555\" → \"five five five, five five five, five five five five\"\n"
            "- \"2nd\" → \"second\"\n"
            "- \"XIV\" → \"fourteen\" (unless in a title, e.g., “the fourteenth”)\n"
            "- \"3.5\" → \"three point five\"\n"
            "- \"⅔\" → \"two-thirds\"\n"
            "- \"Dr.\" → \"Doctor\"\n"
            "- \"Ave.\" → \"Avenue\"\n"
            "- \"St.\" → \"Street\" (except in names like “Saint Patrick”)\n"
            "- \"Ctrl + Z\" → \"control z\"\n"
            "- \"100km\" → \"one hundred kilometers\"\n"
            "- \"100%\" → \"one hundred percent\"\n"
            "- \"elevenlabs.io/docs\" → \"eleven labs dot i o slash docs\"\n"
            "- \"2024-01-01\" → \"January first, twenty twenty-four\"\n"
            "- \"123 Main St, Anytown, USA\" → \"one two three Main Street, Anytown, United States of America\"\n"
            "- \"14:30\" → \"two thirty PM\"\n"
            "- \"01/02/2023\" → \"January second, two-thousand twenty-three\" (use Australian format)\n"
            "- \"name@company.com\" → \"n-a-m-e-@-c-o-m-p-a-n-y-dot-com\"\n"
            "- \"@\" is pronounced as \"at\"\n"
            "- when reading Australian phone numbers starting with \"+61\", substitute the \"+61\" for \"0\"\n\n"
            "---\n\n"
            "## Goals\n"
            "Your role is to:\n"
            "- Greet and orient the caller\n"
            "- Understand their reason for calling\n"
            "- Collect the caller’s details (name, phone, message)\n"
            "- Offer to arrange a callback if needed\n"
            "- End with warmth and professionalism\n"
            "\n"
            "## States\n"
            "Start in `warm_intro` and follow these states to handle the call.\n"
            "### warm_intro\n"
            "You are the initial voice interface for MYRA Health. Greet the caller warmly.\n"
            "Determine if they want to leave a message or request a callback.\n"
            "Use `transfer_call` if the caller needs to be transferred.\n"
            "### leave_message\n"
            "Collect the caller's name, best contact number and short summary.\n"
            "Call `extract_user_message` with these details then go to `confirm_message`.\n"
            "### request_callback\n"
            "Collect the caller's name, email, contact number, reason and preferred time.\n"
            "Call `extract_user_callback` with these details then go to `confirm_message`.\n"
            "### confirm_message\n"
            "Confirm details and thank the caller. Finally call `end_call`.\n"
        )

        super().__init__(
            instructions=prompt,
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
            turn_detection=MultilingualModel(),
            tools=[
                self.end_call,
                self.transfer_call,
                self.extract_user_message,
                self.extract_user_callback,
            ],
        )

    async def on_enter(self):
        # The agent should be polite and greet the user when it joins :)
        self.session.generate_reply(
            instructions=
            "Hello, this is Chloe from Myra Health. Thanks for calling — how can I help you today?",
            allow_interruptions=True,
        )

    # --- Function tools -------------------------------------------------

    @function_tool
    async def end_call(self) -> None:
        """End the call."""
        await self.session.aclose()

    @function_tool
    async def transfer_call(self, number: str) -> None:
        """Transfer the caller to another number."""
        logger.info("Transfer requested to %s", number)

    @function_tool
    async def extract_user_message(
        self,
        caller_name: str,
        caller_contact_number: str,
        caller_message: str,
    ) -> None:
        """Store details for a voicemail message."""
        self._details.update(
            {
                "name": caller_name,
                "contact": caller_contact_number,
                "message": caller_message,
            }
        )
        self.state = "confirm_message"

    @function_tool
    async def extract_user_callback(
        self,
        caller_name: str,
        caller_email: str,
        caller_contact_number: str,
        extract_caller_message: str,
        caller_day_time: str,
    ) -> None:
        """Store details for a callback request."""
        self._details.update(
            {
                "name": caller_name,
                "email": caller_email,
                "contact": caller_contact_number,
                "message": extract_caller_message,
                "day_time": caller_day_time,
            }
        )
        self.state = "confirm_message"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    # Log metrics and collect usage data
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        # end the call if the user has been silent for 30 seconds
        user_away_timeout=30.0,
    )

    async def _max_duration_timeout() -> None:
        await asyncio.sleep(207)
        if session.agent_state != "initializing":
            logger.info("ending call due to max duration")
            await session.aclose()

    max_duration_task = asyncio.create_task(_max_duration_timeout())

    def _on_close(_: events.CloseEvent) -> None:
        max_duration_task.cancel()

    def _on_user_state_changed(event: events.UserStateChangedEvent) -> None:
        if event.new_state == "away":
            logger.info("ending call due to user inactivity")
            asyncio.create_task(session.aclose())

    session.on("close", _on_close)
    session.on("user_state_changed", _on_user_state_changed)

    def _on_function_tools_executed(ev: events.FunctionToolsExecutedEvent) -> None:
        for call, _ in ev.zipped():
            logger.info("tool executed: %s", call.name)

    session.on("function_tools_executed", _on_function_tools_executed)

    # Trigger the on_metrics_collected function when metrics are collected
    session.on("metrics_collected", on_metrics_collected)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # enable background voice & noise cancellation, powered by Krisp
            # included at no additional cost with LiveKit Cloud
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
