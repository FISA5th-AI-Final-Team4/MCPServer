"""
카드 추천 API 요청/응답 스키마 (GraphRAG 기반)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class CardRecommendationRequest(BaseModel):
    """카드 추천 요청 (GraphRAG 기반)"""
    query: str = Field(
        ..., 
        description="사용자의 카드 추천 질문", 
        example="편의점 할인 많이 받을 수 있는 카드 추천해줘"
    )
    session_id: Optional[str] = Field(
        None, 
        description="세션 ID (멀티턴 대화 지원용)"
    )
    mode: Literal["global", "local"] = Field(
        "global",
        description="검색 모드 (global: 전체 조망, local: 세부 정보)"
    )


class CardRecommendationResponse(BaseModel):
    """카드 추천 응답 (GraphRAG 기반)"""
    answer: str = Field(..., description="생성된 카드 추천 답변")
    card_list: List[str] = Field(default_factory=list, description="추천된 카드명 리스트")
