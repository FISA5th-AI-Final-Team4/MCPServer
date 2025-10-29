from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

# 원래 FastAPI 앱
api_app = FastAPI()

# MCP 서버를 위한 별도의 앱
mcp_app = FastAPI()

# API 앱에서 MCP 서버 생성
mcp = FastApiMCP(
    api_app,
    base_url="http://api-host:8001"  # API 앱이 실행되는 URL
)

# 별도의 앱에 MCP 서버 마운트
mcp.mount(mcp_app)