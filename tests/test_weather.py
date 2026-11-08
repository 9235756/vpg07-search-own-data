"""WeatherTool: разбор ответа Open-Meteo без сети."""

from unittest.mock import MagicMock

import pytest
import requests

from hay_v2_bot.components import tools as weather

GEO_OK = {
    "results": [
        {"name": "Москва", "country": "Россия", "latitude": 55.75, "longitude": 37.62}
    ]
}
FORECAST_OK = {
    "current": {
        "temperature_2m": -3.4,
        "apparent_temperature": -8.6,
        "wind_speed_10m": 4.27,
        "weather_code": 71,
    }
}


def fake_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def mock_get(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(weather.requests, "get", mock)
    return mock


def test_weather_parses_response(mock_get):
    mock_get.side_effect = [fake_response(GEO_OK), fake_response(FORECAST_OK)]

    result = weather.get_weather(city="Москва")

    assert "Москва" in result
    assert "небольшой снег" in result
    assert "-3°C" in result
    assert "ощущается как -9°C" in result
    assert "4.3 м/с" in result
    forecast_params = mock_get.call_args_list[1].kwargs["params"]
    assert forecast_params["latitude"] == 55.75
    assert forecast_params["longitude"] == 37.62


def test_city_not_found(mock_get):
    mock_get.return_value = fake_response({"generationtime_ms": 0.1})

    result = weather.get_weather(city="Абырвалг")

    assert "не найден" in result
    assert mock_get.call_count == 1


def test_timeout_returns_text(mock_get):
    mock_get.side_effect = requests.Timeout()

    result = weather.get_weather(city="Москва")

    assert "не ответил вовремя" in result


def test_http_error_returns_text(mock_get):
    mock_get.side_effect = requests.ConnectionError("нет сети")

    assert weather.get_weather(city="Москва") == "Не удалось получить данные о погоде."


def test_incomplete_data(mock_get):
    mock_get.side_effect = [fake_response(GEO_OK), fake_response({"current": {}})]

    assert "неполные данные" in weather.get_weather(city="Москва")


def test_empty_city_makes_no_request(mock_get):
    assert "Не указан город" in weather.get_weather(city="  ")
    mock_get.assert_not_called()


def test_unknown_weather_code():
    assert weather.describe_weather_code(12345) == "погодные условия неизвестны"
    assert weather.describe_weather_code(None) == "погодные условия неизвестны"
