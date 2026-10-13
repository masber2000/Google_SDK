"""Gradio chat UI for the ADK weather agent.

Run locally:      python app.py            (opens http://127.0.0.1:7860)
Run in Colab:     %cd /content/Google_SDK
                  from app import demo; demo.launch(debug=True)

Agents stay loaded for the life of the process, and a warm-up runs as soon as
the page opens (weather API connections, model credentials and connection),
so the first question doesn't pay those start-up costs. Credentials come from
weather_agent/.env, the same file `adk run` / `adk web` use. Set
WEATHER_AGENT_WARMUP=0 to skip the warm-up's one small model call.
"""

import logging
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

# Load credentials before importing the agent: it reads WEATHER_AGENT_MODEL at import.
load_dotenv(Path(__file__).resolve().parent / "weather_agent" / ".env")

import gradio as gr  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from weather_agent import tools  # noqa: E402
from weather_agent.agent import CLAUDE_MODEL, DEFAULT_MODEL, MODEL, build_agent  # noqa: E402

APP_NAME = "weather_agent"
NO_REPLY_MESSAGE = "Sorry, I didn't get a response. Please try again."
MODEL_CHOICES = list(dict.fromkeys([MODEL, DEFAULT_MODEL, CLAUDE_MODEL]))
WARMUP_ENABLED = os.getenv("WEATHER_AGENT_WARMUP", "1") != "0"
WARMUP_CITY = "London"

logger = logging.getLogger(__name__)

_runners: dict[str, InMemoryRunner] = {}  # model name -> runner
_session_ids: dict[tuple[str, str], str] = {}  # (browser session, model) -> ADK session
_warmed_models: set[str] = set()


def _get_runner(model_name: str) -> InMemoryRunner:
    """Returns the runner for a model, building its agent on first use."""
    if model_name not in _runners:
        _runners[model_name] = InMemoryRunner(agent=build_agent(model_name), app_name=APP_NAME)
    return _runners[model_name]


async def _get_session_id(runner: InMemoryRunner, user_id: str, model_name: str) -> str:
    """Returns the ADK session for this browser tab and model, creating it on first use."""
    key = (user_id, model_name)
    if key not in _session_ids:
        session = await runner.session_service.create_session(
            app_name=APP_NAME, user_id=user_id
        )
        _session_ids[key] = session.id
    return _session_ids[key]


def _warm_weather_api() -> None:
    """Opens the pooled connections to both Open-Meteo hosts."""
    started = time.perf_counter()
    result = tools.get_forecast(WARMUP_CITY, days=1)
    logger.info(
        "Weather API warm-up: %s in %.1fs", result["status"], time.perf_counter() - started
    )


async def warm_up_model(model_name: str) -> None:
    """Makes one tiny model call so credentials and the connection are ready.

    Runs on Gradio's event loop (page load / model switch), the same loop that
    serves chat requests, so the warmed client is the one questions will use.
    """
    if not WARMUP_ENABLED or not model_name or model_name in _warmed_models:
        return
    _warmed_models.add(model_name)
    agent = _get_runner(model_name).agent
    llm = agent.canonical_model
    config = agent.generate_content_config or types.GenerateContentConfig()
    request = LlmRequest(
        model=llm.model,
        contents=[types.Content(role="user", parts=[types.Part(text="Reply with OK.")])],
        config=config.model_copy(deep=True),
    )
    started = time.perf_counter()
    try:
        async for _ in llm.generate_content_async(request):
            pass
        logger.info("Model warm-up (%s): %.1fs", model_name, time.perf_counter() - started)
    except Exception as exc:  # a failed warm-up must never block the UI
        _warmed_models.discard(model_name)
        logger.warning("Model warm-up (%s) failed: %s: %s", model_name, type(exc).__name__, exc)


async def respond(message: str, history: list, model_name: str, request: gr.Request):
    """Streams the agent's reply, showing a status line while tools run.

    Conversation memory lives in the ADK session, so Gradio's history is unused.
    Each model keeps its own session; switching models starts a fresh conversation.
    """
    model_name = (model_name or MODEL).strip()
    user_id = (request.session_hash if request else None) or "local"
    status_lines: list[str] = []
    final_text = ""
    try:
        runner = _get_runner(model_name)
        session_id = await _get_session_id(runner, user_id, model_name)
        content = types.Content(role="user", parts=[types.Part(text=message)])
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
        logger.exception("Agent run failed (%s)", model_name)
        yield f"Something went wrong talking to {model_name}: {type(exc).__name__}: {exc}"


with gr.Blocks(title="Weather Agent") as demo:
    model_choice = gr.Dropdown(
        choices=MODEL_CHOICES,
        value=MODEL,
        label="Model",
        info="Gemini, or Claude (needs ANTHROPIC_API_KEY, or Claude enabled in Agent Platform). "
        "You can type another model name.",
        allow_custom_value=True,
    )
    gr.ChatInterface(
        fn=respond,
        additional_inputs=[model_choice],
        title="Weather Agent",
        description=(
            "Ask about current weather or forecasts for any city. "
            "Add a state or country to be specific, e.g. *Springfield, Illinois*."
        ),
        examples=[
            ["What's the weather in Tokyo right now?"],
            ["Will it rain in London this weekend?"],
            ["What's the weather in Springfield?"],
            ["Compare Miami and Seattle right now."],
        ],
    )
    demo.load(warm_up_model, inputs=model_choice)
    model_choice.change(warm_up_model, inputs=model_choice)

if WARMUP_ENABLED:
    threading.Thread(target=_warm_weather_api, name="weather-warmup", daemon=True).start()


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    demo.launch()


if __name__ == "__main__":
    main()
