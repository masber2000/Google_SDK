# Weather Agent (Google ADK)

A single-agent weather bot built with the [Agent Development Kit](https://adk.dev). It answers questions about current conditions and multi-day forecasts for any city, using live data from [Open-Meteo](https://open-meteo.com) (free, no API key).

## How it works

`weather_agent/agent.py` defines `root_agent`, an ADK `Agent` running Gemini (`gemini-2.5-flash` by default; override with `WEATHER_AGENT_MODEL`) with two function tools from `weather_agent/tools.py`:

| Tool | What it does |
|------|--------------|
| `get_weather(city)` | Geocodes the city, then returns current conditions, temperature and feels-like (C and F), humidity, and wind. |
| `get_forecast(city, days=3)` | Returns a daily forecast (1-16 days): conditions, high/low (C and F), and precipitation probability. |

City names can carry a region or country (`Springfield, Illinois`, `Paris, Texas`, `London, UK`). When a bare name matches several comparably sized places, the tool returns `other_matches` and the agent names the one it used and offers the alternatives. Forecast days include weekday names, so "this weekend" and "tomorrow" resolve without the model knowing today's date.

Each tool returns `{"status": "success", ...}` or `{"status": "error", "error_message": ...}`. The agent's instructions tell it to relay errors instead of inventing data, and to state the resolved location so a wrong geocoding match (Paris, Texas vs Paris, France) is visible to the user.

## Setup

Requires Python 3.10+. Pinned to `google-adk==2.7.0` to match the course environment.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp weather_agent/.env.example weather_agent/.env
```

Then put your Gemini API key (from [Google AI Studio](https://aistudio.google.com/app/apikey)) in `weather_agent/.env`.

## Run

### Chat GUI (recommended)

```bash
python app.py    # opens http://127.0.0.1:7860
```

In Colab, run this in a Python cell from the repo root instead; it renders inline and prints a public share link:

```python
from app import demo
demo.launch(debug=True)
```

The GUI keeps the agent loaded between questions and shows a status line while a tool is running.

### ADK CLI and dev UI

From the repository root (the folder that contains `weather_agent/`):

```bash
adk web          # browser dev UI at http://localhost:8000, pick "weather_agent"
adk run weather_agent   # terminal chat
```

On Windows, if `adk web` raises `_make_subprocess_transport NotImplementedError`, use `adk web --no-reload`.

Example prompts:

- What's the weather in Tokyo right now?
- Will it rain in London this weekend?
- Give me a 5-day forecast for Denver.

## Tests

```bash
pytest
```

The tests stub the HTTP layer, so they run offline and need no API keys.
