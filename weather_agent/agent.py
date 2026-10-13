"""ADK weather agent: answers current-weather and forecast questions for any city."""

import os

from google.adk.agents import Agent
from google.genai import types

from .tools import get_forecast, get_weather

MODEL = os.getenv("WEATHER_AGENT_MODEL", "gemini-2.5-flash")

# Gemini 2.5 Flash "thinks" before every reply by default, which roughly
# doubles latency for a task this simple. Budget 0 turns it off. Other models
# keep their defaults because some reject a zero budget.
GENERATE_CONTENT_CONFIG = (
    types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0))
    if "2.5-flash" in MODEL
    else None
)

root_agent = Agent(
    name="weather_agent",
    model=MODEL,
    description="Answers questions about current weather and forecasts for cities worldwide.",
    instruction=(
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
    ),
    tools=[get_weather, get_forecast],
    generate_content_config=GENERATE_CONTENT_CONFIG,
)
