"""Weather tools for the ADK weather agent, backed by the Open-Meteo API.

Open-Meteo is free and needs no API key, so the repo carries no weather
credentials. Each tool returns a dict with a "status" key: "success" with
the data, or "error" with an "error_message" the model can relay to the user.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_FORECAST_DAYS = 16  # Open-Meteo's forecast horizon

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


def _get_json(url: str, params: dict) -> dict:
    """Performs a GET request and returns the decoded JSON body."""
    try:
        response = httpx.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as exc:
        raise WeatherLookupError("The weather service timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise WeatherLookupError(
            f"The weather service returned HTTP {exc.response.status_code}."
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise WeatherLookupError(f"The weather service request failed: {exc}") from exc


def _geocode(city: str) -> dict:
    """Resolves a city name to its best-matching location record."""
    data = _get_json(
        GEOCODING_URL, {"name": city, "count": 1, "language": "en", "format": "json"}
    )
    results = data.get("results") or []
    if not results:
        raise WeatherLookupError(f"Could not find a location named '{city}'.")
    return results[0]


def _location_label(place: dict) -> str:
    """Builds a readable label like 'Paris, Ile-de-France, France'."""
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    return ", ".join(part for part in parts if part)


def _error(message: str) -> dict:
    return {"status": "error", "error_message": message}


def get_weather(city: str) -> dict:
    """Retrieves the current weather for a city.

    Args:
        city: The city name, for example "Paris" or "Tokyo". When several
            places share a name, the most prominent match is used and the
            resolved location is returned so it can be confirmed.

    Returns:
        dict: On success, status "success" with the resolved location, local
        observation time, conditions, temperature in Celsius and Fahrenheit,
        feels-like temperature, relative humidity, and wind speed. On failure,
        status "error" with an error_message.
    """
    if not city or not city.strip():
        return _error("A city name is required.")

    try:
        place = _geocode(city.strip())
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
            "observed_at_local": current["time"],
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
    except (KeyError, TypeError) as exc:
        logger.error("Unexpected Open-Meteo response shape for %r: %r", city, exc)
        return _error("The weather service returned data in an unexpected format.")


def get_forecast(city: str, days: int = 3) -> dict:
    """Retrieves a daily weather forecast for a city.

    Args:
        city: The city name, for example "Paris" or "Tokyo".
        days: Number of forecast days to return, from 1 to 16. Defaults to 3.

    Returns:
        dict: On success, status "success" with the resolved location and a
        list of daily entries (date, conditions, high and low in Celsius and
        Fahrenheit, maximum precipitation probability). On failure, status
        "error" with an error_message.
    """
    if not city or not city.strip():
        return _error("A city name is required.")
    if not isinstance(days, int) or not 1 <= days <= MAX_FORECAST_DAYS:
        return _error(f"days must be a whole number from 1 to {MAX_FORECAST_DAYS}.")

    try:
        place = _geocode(city.strip())
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
        if any(len(values) != len(dates) for values in series):
            raise KeyError("daily arrays have mismatched lengths")

        forecast = []
        for date, code, high_c, low_c, precip_pct in zip(dates, *series):
            forecast.append(
                {
                    "date": date,
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
            "timezone": data.get("timezone", "unknown"),
            "forecast": forecast,
        }
    except WeatherLookupError as exc:
        logger.warning("get_forecast failed for %r: %s", city, exc)
        return _error(str(exc))
    except (KeyError, TypeError) as exc:
        logger.error("Unexpected Open-Meteo response shape for %r: %r", city, exc)
        return _error("The weather service returned data in an unexpected format.")
