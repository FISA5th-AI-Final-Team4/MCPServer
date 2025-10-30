from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from api.router import api_router

# 기존 FastAPI 애플리케이션 생성
app = FastAPI()
app.include_router(api_router)

# MCP 서버 생성 (FastAPI 앱 기반 래핑)
mcp = FastApiMCP(app)

# FastAPI 앱에 MCP 서버 마운트
# mcp.mount_http(mount_path="/mcp")
mcp.mount_sse(mount_path="/mcp")