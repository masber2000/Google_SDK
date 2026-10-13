"""Unit tests for the weather tools. Network calls are stubbed; no API keys needed."""

import pytest

from weather_agent import tools

PARIS = {
    "name": "Paris",
    "admin1": "Ile-de-France",
    "country": "France",
    "latitude": 48.85341,
    "longitude": 2.3488,
}


def make_fake_get_json(forecast_payload, geocode_results=(PARIS,)):
    def fake_get_json(url, params):
        if url == tools.GEOCODING_URL:
            return {"results": list(geocode_results)} if geocode_results else {}
        if url == tools.FORECAST_URL:
            return forecast_payload
        raise AssertionError(f"unexpected URL {url}")

    return fake_get_json


def test_celsius_to_fahrenheit_known_values():
    assert tools.celsius_to_fahrenheit(0) == 32.0
    assert tools.celsius_to_fahrenheit(100) == 212.0
    assert tools.celsius_to_fahrenheit(-40) == -40.0
    assert tools.celsius_to_fahrenheit(21.5) == 70.7


def test_describe_weather_code_known_and_unknown():
    assert tools.describe_weather_code(0) == "Clear sky"
    assert tools.describe_weather_code(95) == "Thunderstorm"
    assert tools.describe_weather_code(42) == "Unknown conditions (code 42)"


def test_get_weather_success(monkeypatch):
    payload = {
        "timezone": "Europe/Paris",
        "current": {
            "time": "2026-10-13T15:00",
            "temperature_2m": 20.0,
            "apparent_temperature": 18.0,
            "relative_humidity_2m": 60,
            "weather_code": 2,
            "wind_speed_10m": 12.5,
        },
    }
    monkeypatch.setattr(tools, "_get_json", make_fake_get_json(payload))

    result = tools.get_weather("Paris")

    assert result["status"] == "success"
    assert result["location"] == "Paris, Ile-de-France, France"
    assert result["conditions"] == "Partly cloudy"
    assert result["temperature_c"] == 20.0
    assert result["temperature_f"] == 68.0
    assert result["feels_like_f"] == 64.4
    assert result["relative_humidity_pct"] == 60
    assert result["timezone"] == "Europe/Paris"


def test_get_weather_unknown_city(monkeypatch):
    monkeypatch.setattr(tools, "_get_json", make_fake_get_json({}, geocode_results=()))

    result = tools.get_weather("Notarealplace")

    assert result["status"] == "error"
    assert "Notarealplace" in result["error_message"]


@pytest.mark.parametrize("city", ["", "   "])
def test_get_weather_blank_city(city):
    assert tools.get_weather(city)["status"] == "error"


def test_get_weather_service_failure(monkeypatch):
    def failing_get_json(url, params):
        raise tools.WeatherLookupError("The weather service timed out.")

    monkeypatch.setattr(tools, "_get_json", failing_get_json)

    result = tools.get_weather("Paris")

    assert result == {"status": "error", "error_message": "The weather service timed out."}


def test_get_weather_malformed_response(monkeypatch):
    monkeypatch.setattr(tools, "_get_json", make_fake_get_json({"unexpected": True}))

    result = tools.get_weather("Paris")

    assert result["status"] == "error"
    assert "unexpected format" in result["error_message"]


def test_get_forecast_success(monkeypatch):
    payload = {
        "timezone": "Europe/Paris",
        "daily": {
            "time": ["2026-10-13", "2026-10-14"],
            "weather_code": [0, 61],
            "temperature_2m_max": [22.0, 18.0],
            "temperature_2m_min": [10.0, 9.0],
            "precipitation_probability_max": [5, 80],
        },
    }
    monkeypatch.setattr(tools, "_get_json", make_fake_get_json(payload))

    result = tools.get_forecast("Paris", days=2)

    assert result["status"] == "success"
    assert len(result["forecast"]) == 2
    first, second = result["forecast"]
    assert first == {
        "date": "2026-10-13",
        "conditions": "Clear sky",
        "high_c": 22.0,
        "high_f": 71.6,
        "low_c": 10.0,
        "low_f": 50.0,
        "precipitation_probability_pct": 5,
    }
    assert second["conditions"] == "Slight rain"
    assert second["precipitation_probability_pct"] == 80


def test_get_forecast_mismatched_arrays(monkeypatch):
    payload = {
        "daily": {
            "time": ["2026-10-13", "2026-10-14"],
            "weather_code": [0],
            "temperature_2m_max": [22.0, 18.0],
            "temperature_2m_min": [10.0, 9.0],
            "precipitation_probability_max": [5, 80],
        },
    }
    monkeypatch.setattr(tools, "_get_json", make_fake_get_json(payload))

    assert tools.get_forecast("Paris", days=2)["status"] == "error"


@pytest.mark.parametrize("days", [0, 17, -1, 2.5])
def test_get_forecast_rejects_out_of_range_days(days):
    result = tools.get_forecast("Paris", days=days)

    assert result["status"] == "error"
    assert "1 to 16" in result["error_message"]
