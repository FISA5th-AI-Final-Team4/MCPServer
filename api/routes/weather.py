from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from datetime import datetime, timezone
from typing import Optional
import httpx

from core.config import settings

# 라우터 생성
router = APIRouter(prefix="/weather", tags=["weather"])

# 입출력 스키마 정의 (서비스 엔드포인트 관련 스키마는 /schemas 폴더에 구현)
class WeatherRequest(BaseModel):
    city: str = Field(..., description="City name to use for the weather lookup.")

class WeatherObservation(BaseModel):
    city: str
    temperature_c: float
    weather: str
    humidity: int
    observed_at: datetime

# 현재 날씨 조회 엔드포인트
@router.post(
    "/weather",
    response_model=WeatherObservation,
    summary="Invoke the weather MCP tool",
    operation_id="get_weather",
)
async def get_weather(payload: WeatherRequest) -> WeatherObservation:
    """Expose the weather tool logic over HTTP for MCP integration tests."""
    return await get_weather_snapshot(payload.city)

async def _call_weather_api(city: str) -> Optional[float]:
    """Attempt to fetch the current temperature from the configured weather provider."""

    async with httpx.AsyncClient(
        base_url=settings.WEATHER_API_BASE_URL,
        timeout=httpx.Timeout(10.0),
    ) as client:
        response = await client.get(
            "/weather",
            params={
                "q": city,
                "appid": settings.WEATHER_API_KEY,
                "units": "metric"
            },
        )
        response.raise_for_status()
        payload = response.json()

    try:
        main_block = payload.get("main", {})
        temperature = float(main_block["temp"])
        weather = payload.get("weather", [])[0]["description"]
        humidity = int(main_block["humidity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(
            [{"loc": ("temperature",), "msg": "Malformed weather payload", "type": "value_error"}],
            WeatherObservation,
        ) from exc

    return temperature, weather, humidity


async def get_weather_snapshot(city: str) -> WeatherObservation:
    """Return the current weather observation for the given city."""

    try:
        print("Calling weather API for city:", city)
        temperature, weather, humidity = await _call_weather_api(city)
        print(f"Received weather data: {temperature}°C, {weather}, {humidity}%")
    except httpx.HTTPError as exc:
        print("Error calling weather API:", str(exc))
        raise HTTPException(
            status_code=502,
            detail="Weather provider error"
        ) from exc
    except ValidationError as exc:
        print("Error parsing weather API response:", str(exc))
        raise HTTPException(
            status_code=502,
            detail="Weather provider returned malformed payload"
        ) from exc

    return WeatherObservation(
        city=city,
        temperature_c=temperature,
        weather=weather,
        humidity=humidity,
        observed_at=datetime.now(timezone.utc)
    )