"""Инструменты агента: факт о кошках, картинка собаки с определением породы, погода.

Перенесены из hay/tools (VPg06) без изменения поведения.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable

import requests

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = 10  # секунд


def logged_tool(tool_name: str) -> Callable:
    """Логирует вызов инструмента, его аргументы и результат или ошибку."""

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> str:
            logger.info("Вызов инструмента %s, аргументы: %s", tool_name, kwargs or args)
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception("Инструмент %s упал с ошибкой", tool_name)
                return f"Инструмент {tool_name} временно недоступен. Попробуйте позже."
            logger.info("Инструмент %s вернул: %.200s", tool_name, result)
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------- cat_fact

CAT_FACT_URL = "https://catfact.ninja/fact"
CAT_FACT_TOOL = "cat_fact"


@logged_tool(CAT_FACT_TOOL)
def get_cat_fact() -> str:
    """Возвращает факт о кошках (на английском, агент переводит его сам)."""
    try:
        response = requests.get(CAT_FACT_URL, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        fact = response.json().get("fact", "").strip()
    except requests.Timeout:
        logger.warning("catfact.ninja не ответил за %s с", HTTP_TIMEOUT)
        return "Сервис фактов о кошках не ответил вовремя. Попробуйте позже."
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Ошибка запроса к catfact.ninja: %s", exc)
        return "Не удалось получить факт о кошках."
    return fact or "Сервис фактов о кошках вернул пустой ответ."


def create_cat_fact_tool():
    from haystack.tools import Tool

    return Tool(
        name=CAT_FACT_TOOL,
        description=(
            "Возвращает случайный интересный факт о кошках. "
            "Вызывай, когда пользователь просит факт о кошках или что-то интересное про котов. "
            "Факт приходит на английском, переведи его на русский."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        function=get_cat_fact,
    )


# ---------------------------------------------------------------- dog_image, dog_image_analyzer

DOG_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"
DOG_IMAGE_TOOL = "dog_image"
DOG_ANALYZER_TOOL = "dog_image_analyzer"

ANALYZER_PROMPT = (
    "Определи породу собаки на фотографии. Ответь на русском: "
    "название породы и краткая история породы в 2–3 предложениях. "
    "Если порода неочевидна, назови наиболее вероятные варианты."
)


@logged_tool(DOG_IMAGE_TOOL)
def get_dog_image() -> str:
    """Возвращает ссылку на случайную картинку собаки."""
    try:
        response = requests.get(DOG_IMAGE_URL, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.Timeout:
        return "Сервис картинок собак не ответил вовремя. Попробуйте позже."
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Ошибка запроса к dog.ceo: %s", exc)
        return "Не удалось получить картинку собаки."
    url = data.get("message", "")
    if data.get("status") != "success" or not str(url).startswith("http"):
        return "Сервис картинок собак вернул некорректный ответ."
    return url


class DogImageAnalyzer:
    """Описывает собаку на картинке через модель OpenAI с поддержкой изображений."""

    def __init__(self, chat_generator):
        self._generator = chat_generator

    def analyze(self, image_url: str) -> str:
        from haystack.dataclasses import ChatMessage, ImageContent

        if not image_url or not image_url.startswith("http"):
            return "Нужна корректная ссылка на картинку."
        try:
            image = ImageContent.from_url(image_url, detail="low")
            message = ChatMessage.from_user(content_parts=[ANALYZER_PROMPT, image])
            result = self._generator.run(messages=[message])
            text = result["replies"][0].text or ""
        except Exception as exc:
            logger.exception("Ошибка анализа картинки %s", image_url)
            return f"Не удалось проанализировать картинку: {exc}"
        return text.strip() or "Модель не смогла описать картинку."


def create_dog_image_tool():
    from haystack.tools import Tool

    return Tool(
        name=DOG_IMAGE_TOOL,
        description=(
            "Возвращает ссылку на случайную фотографию собаки. "
            "Вызывай, когда пользователь просит показать собаку. "
            "Картинку пользователю отправит бот, в ответе ссылку повторять не нужно."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        function=get_dog_image,
    )


def create_dog_analyzer_tool(chat_generator):
    from haystack.tools import Tool

    analyzer = DogImageAnalyzer(chat_generator)

    @logged_tool(DOG_ANALYZER_TOOL)
    def analyze_dog_image(image_url: str) -> str:
        return analyzer.analyze(image_url)

    return Tool(
        name=DOG_ANALYZER_TOOL,
        description=(
            "Анализирует фотографию собаки по ссылке: определяет породу и кратко рассказывает "
            "её историю. Вызывай после dog_image, если пользователь хочет узнать породу, "
            "или когда пользователь сам прислал ссылку на фото собаки."
        ),
        parameters={
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "Прямая ссылка на изображение собаки (http или https).",
                }
            },
            "required": ["image_url"],
        },
        function=analyze_dog_image,
    )


# ---------------------------------------------------------------- weather (Open-Meteo, без ключа)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TOOL = "weather"

# Коды погоды WMO, которые возвращает Open-Meteo.
WEATHER_CODES = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "туман с изморозью",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "слабая ледяная морось",
    57: "сильная ледяная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "слабый ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    77: "снежные зёрна",
    80: "небольшой ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "небольшой снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с небольшим градом",
    99: "гроза с сильным градом",
}


class WeatherError(Exception):
    """Ошибка, текст которой можно показать пользователю."""


def describe_weather_code(code) -> str:
    try:
        return WEATHER_CODES.get(int(code), "погодные условия неизвестны")
    except (TypeError, ValueError):
        return "погодные условия неизвестны"


def _get_json(url: str, params: dict) -> dict:
    try:
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise WeatherError("Сервис погоды не ответил вовремя. Попробуйте позже.") from exc
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Ошибка запроса к %s: %s", url, exc)
        raise WeatherError("Не удалось получить данные о погоде.") from exc


def geocode_city(city: str) -> dict:
    data = _get_json(
        GEOCODING_URL,
        {"name": city, "count": 1, "language": "ru", "format": "json"},
    )
    results = data.get("results") or []
    if not results:
        raise WeatherError(f"Город «{city}» не найден. Проверьте название.")
    return results[0]


def format_weather(place: dict, current: dict) -> str:
    name = place.get("name", "")
    location = ", ".join(part for part in (name, place.get("country")) if part)
    temperature = current.get("temperature_2m")
    feels_like = current.get("apparent_temperature")
    wind = current.get("wind_speed_10m")
    if temperature is None or feels_like is None or wind is None:
        raise WeatherError("Сервис погоды вернул неполные данные.")
    description = describe_weather_code(current.get("weather_code"))
    return (
        f"Сейчас {location}: {description}, "
        f"температура {round(temperature)}°C, ощущается как {round(feels_like)}°C, "
        f"ветер {round(wind, 1)} м/с."
    )


@logged_tool(WEATHER_TOOL)
def get_weather(city: str) -> str:
    """Возвращает текущую погоду в городе текстом на русском. Не бросает исключений."""
    city = (city or "").strip()
    if not city:
        return "Не указан город. Уточните, для какого города нужна погода."
    try:
        place = geocode_city(city)
        data = _get_json(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,wind_speed_10m,weather_code",
                "wind_speed_unit": "ms",
                "timezone": "auto",
            },
        )
        return format_weather(place, data.get("current") or {})
    except WeatherError as exc:
        logger.info("WeatherTool: %s", exc)
        return str(exc)
    except (KeyError, TypeError) as exc:
        logger.warning("Неожиданный формат ответа Open-Meteo: %s", exc)
        return "Сервис погоды вернул ответ в неожиданном формате."


def create_weather_tool():
    from haystack.tools import Tool

    return Tool(
        name=WEATHER_TOOL,
        description=(
            "Возвращает текущую погоду в указанном городе: описание, температуру, "
            "температуру по ощущениям и скорость ветра. Вызывай, когда пользователь спрашивает "
            "о погоде, температуре, ветре или о том, как одеться в каком-то городе."
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Название города, например «Москва» или «Санкт-Петербург».",
                }
            },
            "required": ["city"],
        },
        function=get_weather,
    )


def create_tools(make_generator: Callable) -> list:
    """Все инструменты агента. make_generator создаёт OpenAIChatGenerator с адресом прокси."""
    return [
        create_cat_fact_tool(),
        create_dog_image_tool(),
        create_dog_analyzer_tool(make_generator()),
        create_weather_tool(),
    ]
