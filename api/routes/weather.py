from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from datetime import datetime, timezone
from typing import Optional
import httpx

from core.config import settings

# SSE, Ollama, fastmcp를 위한 임포트
from sse_starlette.sse import EventSourceResponse
from langchain_ollama import ChatOllama
# QueryRequest 스키마를 LLM 서버에서 가져오거나 여기에 정의해야 합니다.
# 여기서는 간단한 pydantic 모델로 임시 정의합니다.
# from schemas.mcp_router import QueryRequest # LLM 서버의 스키마 재사용 (가정)
class QueryRequest(BaseModel):
    query: str

router = APIRouter(prefix="/tools", tags=["MCP Tools"])

# MCP 서버 내부에 자체 Ollama LLM 인스턴스 생성 (RAG용)
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
OLLAMA_MODEL_NAME = settings.OLLAMA_MODEL_NAME

mcp_llm = ChatOllama(
    model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.3
)

# 입출력 스키마 정의
class WeatherRequest(BaseModel):
    city: str = Field(..., description="City name to use for the weather lookup.")

class WeatherObservation(BaseModel):
    city: str
    temperature_c: float
    weather: str
    humidity: int
    observed_at: datetime

# 날씨 API 호출 헬퍼 함수
# async def _call_weather_api(city: str) -> Optional[float]:
#     """Attempt to fetch the current temperature from the configured weather provider."""

#     async with httpx.AsyncClient(
#         base_url=settings.WEATHER_API_BASE_URL,
#         timeout=httpx.Timeout(10.0),
#     ) as client:
#         response = await client.get(
#             "/weather",
#             params={
#                 "q": city,
#                 "appid": settings.WEATHER_API_KEY,
#                 "units": "metric"
#             },
#         )
#         response.raise_for_status()
#         payload = response.json()

#     try:
#         main_block = payload.get("main", {})
#         temperature = float(main_block["temp"])
#         weather = payload.get("weather", [])[0]["description"]
#         humidity = int(main_block["humidity"])
#     except (KeyError, TypeError, ValueError) as exc:
#         raise ValidationError(
#             [{"loc": ("temperature",), "msg": "Malformed weather payload", "type": "value_error"}],
#             WeatherObservation,
#         ) from exc

#     return temperature, weather, humidity


# async def get_weather_snapshot(city: str) -> WeatherObservation:
#     """Return the current weather observation for the given city."""

#     try:
#         print("Calling weather API for city:", city)
#         temperature, weather, humidity = await _call_weather_api(city)
#         print(f"Received weather data: {temperature}°C, {weather}, {humidity}%")
#     except httpx.HTTPError as exc:
#         print("Error calling weather API:", str(exc))
#         raise HTTPException(
#             status_code=502,
#             detail="Weather provider error"
#         ) from exc
#     except ValidationError as exc:
#         print("Error parsing weather API response:", str(exc))
#         raise HTTPException(
#             status_code=502,
#             detail="Weather provider returned malformed payload"
#         ) from exc

#     return WeatherObservation(
#         city=city,
#         temperature_c=temperature,
#         weather=weather,
#         humidity=humidity,
#         observed_at=datetime.now(timezone.utc)
#     )

# 1. JSON Tool: 현재 날씨 조회
# @router.post(
#     "/weather",
#     response_model=WeatherObservation,
#     summary="Invoke the weather MCP tool",
#     operation_id="get_weather",
# )
# async def get_weather(payload: WeatherRequest) -> WeatherObservation:
#     """Expose the weather tool logic over HTTP for MCP integration tests."""
#     return await get_weather_snapshot(payload.city)


# 2. RAG Streaming Tool: Ollama 응답을 SSE로 스트리밍
# @router.post(
#     "/rag-generation",
#     summary="Invoke the RAG generation tool (SSE)",
#     operation_id="get_rag_response", # LLM 서버가 이 ID로 호출
# )
# async def get_rag_response(req: QueryRequest):
#     """
#     Ollama LLM의 응답을 SSE(Server-Sent Events)로 스트리밍합니다.
#     fastmcp의 SSETransport가 이 엔드포인트와 통신합니다.
#     """
    
#     async def sse_event_generator():
#         """LLM 토큰을 받아 SSE 이벤트 형식으로 yield하는 제너레이터"""
#         print(f"MCP (RAG): Ollama 스트리밍 시작 (Query: {req.query[:20]}...)")
#         try:
#             # LangChain의 astream()은 비동기 제너레이터입니다.
#             async for chunk in mcp_llm.astream(req.query):
#                 # .content가 문자열 토큰입니다.
#                 if chunk.content:
#                     yield {
#                         "event": "rag_chunk", # 이벤트 타입
#                         "data": chunk.content   # 데이터 (토큰)
#                     }
            
#             print("MCP (RAG): Ollama 스트리밍 완료")

#         except Exception as e:
#             print(f"MCP (RAG) -> Ollama 오류: {e}")
#             yield {"event": "error", "data": str(e)}
#         finally:
#             # fastmcp 클라이언트가 스트림 종료를 알 수 있도록
#             # 'end' 이벤트를 전송합니다.
#             yield {"event": "end", "data": "[END_OF_RAG_STREAM]"}

#     return EventSourceResponse(sse_event_generator())