"""Unit tests for the weather tools. Network calls are stubbed; no API keys needed."""

import pytest

from weather_agent import tools

PARIS = {
    "name": "Paris",
    "admin1": "Ile-de-France",
    "country": "France",
    "country_code": "FR",
    "latitude": 48.85341,
    "longitude": 2.3488,
}


def springfield(state: str) -> dict:
    return {
        "name": "Springfield",
        "admin1": state,
        "country": "United States",
        "country_code": "US",
        "latitude": 39.0,
        "longitude": -90.0,
    }


SPRINGFIELDS = (
    springfield("Missouri"),
    springfield("Illinois"),
    {**springfield("Missouri"), "name": "Palmyra"},  # alternate-name hit from the geocoder
    springfield("Massachusetts"),
    springfield("Ohio"),
    springfield("Tennessee"),
)

CURRENT_PAYLOAD = {
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


@pytest.fixture(autouse=True)
def clear_geocode_cache():
    tools._geocode_candidates.cache_clear()
    yield
    tools._geocode_candidates.cache_clear()


def make_fake_get_json(forecast_payload, geocode_results=(PARIS,), calls=None):
    def fake_get_json(url, params):
        if calls is not None:
            calls.append(url)
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


def test_weekday_name_known_dates():
    assert tools.weekday_name("2026-10-13") == "Tuesday"
    assert tools.weekday_name("2026-10-17T09:30") == "Saturday"


def test_get_weather_success(monkeypatch):
    monkeypatch.setattr(tools, "_get_json", make_fake_get_json(CURRENT_PAYLOAD))

    result = tools.get_weather("Paris")

    assert result["status"] == "success"
    assert result["location"] == "Paris, Ile-de-France, France"
    assert result["other_matches"] == []
    assert result["local_weekday"] == "Tuesday"
    assert result["conditions"] == "Partly cloudy"
    assert result["temperature_c"] == 20.0
    assert result["temperature_f"] == 68.0
    assert result["feels_like_f"] == 64.4
    assert result["relative_humidity_pct"] == 60
    assert result["timezone"] == "Europe/Paris"


def test_geocoding_is_cached_across_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_get_json", make_fake_get_json(CURRENT_PAYLOAD, calls=calls))

    tools.get_weather("Paris")
    tools.get_weather("paris")

    assert calls.count(tools.GEOCODING_URL) == 1
    assert calls.count(tools.FORECAST_URL) == 2


def test_ambiguous_city_reports_other_matches(monkeypatch):
    monkeypatch.setattr(
        tools, "_get_json", make_fake_get_json(CURRENT_PAYLOAD, geocode_results=SPRINGFIELDS)
    )

    result = tools.get_weather("Springfield")

    assert result["location"] == "Springfield, Missouri, United States"
    assert result["other_matches"] == [
        "Springfield, Illinois, United States",
        "Springfield, Massachusetts, United States",
        "Springfield, Ohio, United States",
    ]


@pytest.mark.parametrize(
    "query, expected_state",
    [
        ("Springfield, Illinois", "Illinois"),
        ("springfield, ohio", "Ohio"),
        ("Springfield, Mass", "Massachusetts"),
        ("Springfield, Illinois, USA", "Illinois"),
    ],
)
def test_qualifier_selects_region(monkeypatch, query, expected_state):
    monkeypatch.setattr(
        tools, "_get_json", make_fake_get_json(CURRENT_PAYLOAD, geocode_results=SPRINGFIELDS)
    )

    result = tools.get_weather(query)

    assert result["status"] == "success"
    assert result["location"] == f"Springfield, {expected_state}, United States"


def test_minor_namesakes_are_not_flagged(monkeypatch):
    london = {**PARIS, "name": "London", "admin1": "England", "country": "United Kingdom",
              "country_code": "GB", "population": 8_961_989}
    london_ontario = {**london, "admin1": "Ontario", "country": "Canada",
                      "country_code": "CA", "population": 422_324}
    monkeypatch.setattr(
        tools, "_get_json",
        make_fake_get_json(CURRENT_PAYLOAD, geocode_results=(london, london_ontario)),
    )

    result = tools.get_weather("London")

    assert result["location"] == "London, England, United Kingdom"
    assert result["other_matches"] == []
    assert tools.get_weather("London, Canada")["location"] == "London, Ontario, Canada"


def test_qualifier_with_no_match_is_an_error(monkeypatch):
    monkeypatch.setattr(
        tools, "_get_json", make_fake_get_json(CURRENT_PAYLOAD, geocode_results=SPRINGFIELDS)
    )

    result = tools.get_weather("Springfield, Alaska")

    assert result["status"] == "error"
    assert "Springfield, Alaska" in result["error_message"]


def test_get_weather_unknown_city(monkeypatch):
    monkeypatch.setattr(tools, "_get_json", make_fake_get_json({}, geocode_results=()))

    result = tools.get_weather("Notarealplace")

    assert result["status"] == "error"
    assert "Notarealplace" in result["error_message"]


@pytest.mark.parametrize("city", ["", "   ", ",", " , "])
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
    assert result["today_local"] == "2026-10-13"
    assert result["today_weekday"] == "Tuesday"
    assert len(result["forecast"]) == 2
    first, second = result["forecast"]
    assert first == {
        "date": "2026-10-13",
        "weekday": "Tuesday",
        "conditions": "Clear sky",
        "high_c": 22.0,
        "high_f": 71.6,
        "low_c": 10.0,
        "low_f": 50.0,
        "precipitation_probability_pct": 5,
    }
    assert second["weekday"] == "Wednesday"
    assert second["conditions"] == "Slight rain"
    assert second["precipitation_probability_pct"] == 80


@pytest.mark.parametrize(
    "daily",
    [
        {
            "time": ["2026-10-13", "2026-10-14"],
            "weather_code": [0],
            "temperature_2m_max": [22.0, 18.0],
            "temperature_2m_min": [10.0, 9.0],
            "precipitation_probability_max": [5, 80],
        },
        {
            "time": [],
            "weather_code": [],
            "temperature_2m_max": [],
            "temperature_2m_min": [],
            "precipitation_probability_max": [],
        },
    ],
)
def test_get_forecast_malformed_daily(monkeypatch, daily):
    monkeypatch.setattr(tools, "_get_json", make_fake_get_json({"daily": daily}))

    assert tools.get_forecast("Paris", days=2)["status"] == "error"


@pytest.mark.parametrize("days", [0, 17, -1, 2.5])
def test_get_forecast_rejects_out_of_range_days(days):
    result = tools.get_forecast("Paris", days=days)

    assert result["status"] == "error"
    assert "1 to 16" in result["error_message"]
