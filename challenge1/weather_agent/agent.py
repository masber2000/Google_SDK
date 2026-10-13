"""ADK weather agent: answers current-weather and forecast questions for any city.

Model selection (WEATHER_AGENT_MODEL, default gemini-2.5-flash):
  * Gemini model names run on the Gemini API or Agent Platform, per the
    GOOGLE_* settings in weather_agent/.env.
  * Claude model names (e.g. claude-opus-5-5) run on the Anthropic API when
    ANTHROPIC_API_KEY is set, otherwise on Claude in Agent Platform using
    GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION. Requires the `anthropic` package.
"""

import os

from google.adk.agents import Agent
from google.genai import types

from .tools import get_forecast, get_weather

DEFAULT_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-opus-5-5"
MODEL = os.getenv("WEATHER_AGENT_MODEL", DEFAULT_MODEL)

# Weather lookups are simple tool routing, so Claude runs at low effort for speed.
CLAUDE_EFFORT = os.getenv("WEATHER_AGENT_CLAUDE_EFFORT", "low")

INSTRUCTION = (
    "You are a concise, friendly weather assistant.\n"
    "- For current conditions, call get_weather.\n"
    "- For future days, call get_forecast. You do not know today's date on "
    "your own: the forecast's today_local/today_weekday and each entry's "
    "weekday tell you. For 'tomorrow', 'this weekend', 'next few days' or any "
    "named weekday, request days=7 and pick the matching entries by weekday. "
    "Never ask the user what today's date is.\n"
    "- If the user asks about several cities, call the tool once for every "
    "city and cover all of them in your answer.\n"
    "- Always state the resolved location. If other_matches is non-empty, say "
    "which place you used and list the alternatives so the user can pick one; "
    "they can reply with e.g. 'Springfield, Illinois'.\n"
    "- Give temperatures in both Celsius and Fahrenheit.\n"
    "- If a tool returns status 'error', explain the problem plainly and do not "
    "invent weather data.\n"
    "- Only answer weather questions; politely decline anything else."
)


def _model_and_config(model_name: str):
    """Returns the ADK model (name or BaseLlm) and generation config for a model name."""
    if model_name.startswith("claude-"):
        # Imported lazily so Gemini-only setups don't need the anthropic package.
        from google.adk.models.anthropic_llm import (
            AnthropicGenerateContentConfig,
            AnthropicLlm,
            Claude,
        )

        llm_class = AnthropicLlm if os.getenv("ANTHROPIC_API_KEY") else Claude
        return llm_class(model=model_name), AnthropicGenerateContentConfig(effort=CLAUDE_EFFORT)

    if "2.5-flash" in model_name:
        # Gemini 2.5 Flash "thinks" before every reply by default, which roughly
        # doubles latency for a task this simple. Budget 0 turns it off. Other
        # Gemini models keep their defaults because some reject a zero budget.
        thinking_off = types.ThinkingConfig(thinking_budget=0)
        return model_name, types.GenerateContentConfig(thinking_config=thinking_off)

    return model_name, None


def build_agent(model_name: str = MODEL) -> Agent:
    """Builds the weather agent on the given model."""
    model, config = _model_and_config(model_name)
    return Agent(
        name="weather_agent",
        model=model,
        description="Answers questions about current weather and forecasts for cities worldwide.",
        instruction=INSTRUCTION,
        tools=[get_weather, get_forecast],
        generate_content_config=config,
    )


root_agent = build_agent()
