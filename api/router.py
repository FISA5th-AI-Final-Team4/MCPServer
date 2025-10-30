from fastapi import APIRouter

from api.routes import tools, weather


api_router = APIRouter()
api_router.include_router(tools.router) # 추후 개발 예정 엔드포인트
api_router.include_router(weather.router) # 날씨 API 이용한 엔드포인트 (예제)