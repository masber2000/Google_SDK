"""ADK weather agent: answers current-weather and forecast questions for any city."""

import os

from google.adk.agents import Agent

from .tools import get_forecast, get_weather

MODEL = os.getenv("WEATHER_AGENT_MODEL", "gemini-2.5-flash")

root_agent = Agent(
    name="weather_agent",
    model=MODEL,
    description="Answers questions about current weather and forecasts for cities worldwide.",
    instruction=(
        "You are a concise, friendly weather assistant.\n"
        "- For current conditions, call get_weather.\n"
        "- For future days (tomorrow, this weekend, next week), call get_forecast "
        "with the number of days needed to cover the request (max 16).\n"
        "- Always state the resolved location from the tool result so the user can "
        "catch a wrong match (for example Paris, Texas vs Paris, France). If the "
        "user's city is ambiguous and the match looks wrong, ask them to clarify.\n"
        "- Give temperatures in both Celsius and Fahrenheit.\n"
        "- If a tool returns status 'error', explain the problem plainly and do not "
        "invent weather data.\n"
        "- Only answer weather questions; politely decline anything else."
    ),
    tools=[get_weather, get_forecast],
)
