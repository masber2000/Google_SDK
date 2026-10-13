"""Weather tools for the ADK weather agent, backed by the Open-Meteo API.

Open-Meteo is free and needs no API key, so the repo carries no weather
credentials. Each tool returns a dict with a "status" key: "success" with
the data, or "error" with an "error_message" the model can relay to the user.
"""

import datetime
import functools
import logging

import httpx

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Short timeouts plus one retry: a stalled request is retried quickly instead of
# leaving the agent waiting on a single slow connection.
REQUEST_TIMEOUT = httpx.Timeout(4.0, connect=2.0)
REQUEST_ATTEMPTS = 2
MAX_FORECAST_DAYS = 16  # Open-Meteo's forecast horizon
GEOCODE_CANDIDATES = 10
MAX_OTHER_MATCHES = 3
AMBIGUITY_POPULATION_RATIO = 0.25  # alternative must be >= 25% of the top match
COUNTRY_ALIASES = {"usa": "us", "u.s.": "us", "u.s.a.": "us", "uk": "gb"}

# One pooled client so repeat calls reuse the TLS connection instead of
# paying a fresh handshake on every request.
_http_client = httpx.Client(timeout=REQUEST_TIMEOUT)

# WMO weather interpretation codes used by Open-Meteo.
WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherLookupError(Exception):
    """Raised when a location or weather lookup cannot be completed."""


def describe_weather_code(code: int) -> str:
    """Maps a WMO weather code to a human-readable description."""
    return WMO_WEATHER_CODES.get(code, f"Unknown conditions (code {code})")


def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts Celsius to Fahrenheit, rounded to one decimal place."""
    return round(celsius * 9 / 5 + 32, 1)


def weekday_name(iso_date_or_datetime: str) -> str:
    """Returns the weekday for an ISO date or datetime string, e.g. 'Tuesday'."""
    return datetime.date.fromisoformat(iso_date_or_datetime[:10]).strftime("%A")


def _get_json(url: str, params: dict) -> dict:
    """Performs a GET request and returns the decoded JSON body.

    Timeouts and connection failures are retried once; HTTP error statuses are not.
    """
    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            response = _http_client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt < REQUEST_ATTEMPTS:
                logger.info("Retrying %s after %s (attempt %d)", url, type(exc).__name__, attempt)
                continue
            if isinstance(exc, httpx.TimeoutException):
                raise WeatherLookupError("The weather service timed out.") from exc
            raise WeatherLookupError(f"Could not reach the weather service: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise WeatherLookupError(
                f"The weather service returned HTTP {exc.response.status_code}."
            ) from exc
        except ValueError as exc:
            raise WeatherLookupError(f"The weather service sent invalid JSON: {exc}") from exc
    raise AssertionError("unreachable")


@functools.lru_cache(maxsize=256)
def _geocode_candidates(name: str) -> tuple:
    """Returns geocoding candidates for a place name, most prominent first.

    Cached because city coordinates never change; failures raise and are
    therefore not cached.
    """
    data = _get_json(
        GEOCODING_URL,
        {"name": name, "count": GEOCODE_CANDIDATES, "language": "en", "format": "json"},
    )
    return tuple(data.get("results") or [])


def _matches_qualifier(place: dict, qualifier: str) -> bool:
    """True if a qualifier like 'Illinois', 'France' or 'US' fits the place."""
    wanted = qualifier.casefold()
    wanted = COUNTRY_ALIASES.get(wanted, wanted)
    fields = (place.get("admin1"), place.get("country"), place.get("country_code"))
    return any(field and field.casefold().startswith(wanted) for field in fields)


def _location_label(place: dict) -> str:
    """Builds a readable label like 'Paris, Ile-de-France, France'."""
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    return ", ".join(part for part in parts if part)


def resolve_location(city: str) -> tuple[dict, list[str]]:
    """Resolves 'City' or 'City, Region/Country' to a place plus other candidates.

    Returns the best match and labels of other same-named places, so the
    agent can tell the user when a name like 'Springfield' is ambiguous.
    """
    name, *qualifiers = [part.strip() for part in city.split(",") if part.strip()] or [""]
    if not name:
        raise WeatherLookupError("A city name is required.")

    candidates = _geocode_candidates(name.casefold())
    same_name = [p for p in candidates if p.get("name", "").casefold() == name.casefold()]
    pool = same_name or list(candidates)
    matches = [p for p in pool if all(_matches_qualifier(p, q) for q in qualifiers)]
    if not matches:
        raise WeatherLookupError(f"Could not find a location named '{city}'.")

    best = matches[0]
    best_population = best.get("population") or 0
    # Only flag genuinely comparable places: Springfield, IL vs MO is ambiguous,
    # London, Ontario vs London, England is not worth interrupting the user for.
    comparable = [
        p
        for p in matches[1:]
        if not best_population
        or (p.get("population") or 0) >= best_population * AMBIGUITY_POPULATION_RATIO
    ]
    others = [_location_label(p) for p in comparable[:MAX_OTHER_MATCHES]]
    return best, others


def _error(message: str) -> dict:
    return {"status": "error", "error_message": message}


def get_weather(city: str) -> dict:
    """Retrieves the current weather for a city.

    Args:
        city: The city name, optionally followed by a region or country to
            disambiguate, for example "Paris", "Paris, Texas" or
            "Springfield, Illinois".

    Returns:
        dict: On success, status "success" with the resolved location, any
        other_matches (other places with the same name), local observation
        time and weekday, conditions, temperature in Celsius and Fahrenheit,
        feels-like temperature, relative humidity, and wind speed. On failure,
        status "error" with an error_message.
    """
    if not city or not city.strip():
        return _error("A city name is required.")

    try:
        place, other_matches = resolve_location(city)
        data = _get_json(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
        )
        current = data["current"]
        temperature_c = current["temperature_2m"]
        feels_like_c = current["apparent_temperature"]
        return {
            "status": "success",
            "location": _location_label(place),
            "other_matches": other_matches,
            "observed_at_local": current["time"],
            "local_weekday": weekday_name(current["time"]),
            "timezone": data.get("timezone", "unknown"),
            "conditions": describe_weather_code(current["weather_code"]),
            "temperature_c": temperature_c,
            "temperature_f": celsius_to_fahrenheit(temperature_c),
            "feels_like_c": feels_like_c,
            "feels_like_f": celsius_to_fahrenheit(feels_like_c),
            "relative_humidity_pct": current["relative_humidity_2m"],
            "wind_speed_kmh": current["wind_speed_10m"],
        }
    except WeatherLookupError as exc:
        logger.warning("get_weather failed for %r: %s", city, exc)
        return _error(str(exc))
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Unexpected Open-Meteo response shape for %r: %r", city, exc)
        return _error("The weather service returned data in an unexpected format.")


def get_forecast(city: str, days: int = 3) -> dict:
    """Retrieves a daily weather forecast for a city.

    The first entry is today in the city's local time zone; each entry has a
    weekday so relative requests ("tomorrow", "this weekend") can be matched.

    Args:
        city: The city name, optionally followed by a region or country to
            disambiguate, for example "Paris, Texas".
        days: Number of forecast days to return, from 1 to 16. Defaults to 3.
            Use 7 to cover any day in the coming week, including the weekend.

    Returns:
        dict: On success, status "success" with the resolved location, any
        other_matches, today's local date and weekday, and a list of daily
        entries (date, weekday, conditions, high and low in Celsius and
        Fahrenheit, maximum precipitation probability). On failure, status
        "error" with an error_message.
    """
    if not city or not city.strip():
        return _error("A city name is required.")
    if not isinstance(days, int) or not 1 <= days <= MAX_FORECAST_DAYS:
        return _error(f"days must be a whole number from 1 to {MAX_FORECAST_DAYS}.")

    try:
        place, other_matches = resolve_location(city)
        data = _get_json(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "forecast_days": days,
                "timezone": "auto",
            },
        )
        daily = data["daily"]
        dates = daily["time"]
        series = (
            daily["weather_code"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_probability_max"],
        )
        if not dates or any(len(values) != len(dates) for values in series):
            raise KeyError("daily arrays are empty or have mismatched lengths")

        forecast = []
        for date, code, high_c, low_c, precip_pct in zip(dates, *series):
            forecast.append(
                {
                    "date": date,
                    "weekday": weekday_name(date),
                    "conditions": describe_weather_code(code),
                    "high_c": high_c,
                    "high_f": celsius_to_fahrenheit(high_c),
                    "low_c": low_c,
                    "low_f": celsius_to_fahrenheit(low_c),
                    "precipitation_probability_pct": precip_pct,
                }
            )
        return {
            "status": "success",
            "location": _location_label(place),
            "other_matches": other_matches,
            "today_local": dates[0],
            "today_weekday": weekday_name(dates[0]),
            "timezone": data.get("timezone", "unknown"),
            "forecast": forecast,
        }
    except WeatherLookupError as exc:
        logger.warning("get_forecast failed for %r: %s", city, exc)
        return _error(str(exc))
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Unexpected Open-Meteo response shape for %r: %r", city, exc)
        return _error("The weather service returned data in an unexpected format.")
