from pydantic import BaseModel
from typing import Optional, List

from langchain_core.tools import BaseTool


class ConsumptionRecommandRequest(BaseModel):
    """ 소비데이터 기반 카드 추천 요청 모델 """
    session_id: str

# MCP 툴 클래스
class ConsumptionRecommandTool(BaseTool):
    name: str = "consumption_recommand"
    # args_schema를 정의하여 session_id가 포함된 요청을 받도록 설정
    args_schema: ConsumptionRecommandRequest

    async def _run(self, args: ConsumptionRecommandRequest):
        # 실제 툴 로직
        return {
            "answer": "로그인이 필요합니다.",
            "login_required": True,
            "card_list": []
        }

class ConsumptionRecommandResponse(BaseModel):
    """ 소비데이터 기반 카드 추천 응답 모델 """
    answer: str
    login_required: bool
    recommended_cards: List[str]