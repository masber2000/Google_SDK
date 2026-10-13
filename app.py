"""Gradio chat UI for the ADK weather agent.

Run locally:      python app.py            (opens http://127.0.0.1:7860)
Run in Colab:     %cd /content/Google_SDK
                  from app import demo; demo.launch(debug=True)

The agent stays loaded for the life of the process, so each question costs
only the model and weather calls, not ADK start-up. Model credentials come
from weather_agent/.env, the same file `adk run` / `adk web` use.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load credentials before importing the agent: it reads WEATHER_AGENT_MODEL at import.
load_dotenv(Path(__file__).resolve().parent / "weather_agent" / ".env")

import gradio as gr  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from weather_agent.agent import root_agent  # noqa: E402

APP_NAME = "weather_agent"
NO_REPLY_MESSAGE = "Sorry, I didn't get a response. Please try again."

logger = logging.getLogger(__name__)

runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
_session_ids: dict[str, str] = {}  # browser session -> ADK session


async def _get_session_id(user_id: str) -> str:
    """Returns the ADK session for this browser tab, creating it on first use."""
    if user_id not in _session_ids:
        session = await runner.session_service.create_session(
            app_name=APP_NAME, user_id=user_id
        )
        _session_ids[user_id] = session.id
    return _session_ids[user_id]


async def respond(message: str, history: list, request: gr.Request):
    """Streams the agent's reply, showing a status line while tools run.

    Conversation memory lives in the ADK session, so Gradio's history is unused.
    """
    user_id = (request.session_hash if request else None) or "local"
    session_id = await _get_session_id(user_id)
    content = types.Content(role="user", parts=[types.Part(text=message)])

    status_lines: list[str] = []
    final_text = ""
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            for call in event.get_function_calls() or []:
                city = (call.args or {}).get("city", "")
                verb = "Forecasting" if call.name == "get_forecast" else "Checking"
                status_lines.append(f"_{verb} {city}…_")
                yield "\n\n".join(status_lines)
            if event.is_final_response() and event.content and event.content.parts:
                text = "".join(part.text or "" for part in event.content.parts).strip()
                if text:
                    final_text = text
        yield final_text or NO_REPLY_MESSAGE
    except Exception as exc:  # surface model/auth failures in the chat, not a stack trace
        logger.exception("Agent run failed")
        yield f"Something went wrong talking to the model: {type(exc).__name__}: {exc}"


demo = gr.ChatInterface(
    fn=respond,
    title="Weather Agent",
    description=(
        "Ask about current weather or forecasts for any city. "
        "Add a state or country to be specific, e.g. *Springfield, Illinois*."
    ),
    examples=[
        "What's the weather in Tokyo right now?",
        "Will it rain in London this weekend?",
        "What's the weather in Springfield?",
        "Compare Miami and Seattle right now.",
    ],
)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    demo.launch()


if __name__ == "__main__":
    main()
