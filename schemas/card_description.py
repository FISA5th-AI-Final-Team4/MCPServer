"""
카드 설명 API 요청/응답 스키마
"""

from pydantic import BaseModel, Field
from typing import Optional


class CardDescriptionRequest(BaseModel):
    """카드 설명 요청"""
    query: str = Field(
        ..., 
        description="카드명과 질문 내용 (예: '우리카드 7CORE 연회비 얼마야?')", 
        example="카드의정석 every point 혜택 알려줘"
    )
    top_k: Optional[int] = Field(
        6, 
        description="검색할 문서 수", 
        ge=1, 
        le=20
    )


class CardDescriptionResponse(BaseModel):
    """카드 설명 응답"""
    answer: str = Field(..., description="생성된 카드 설명 답변")
    card_id: Optional[str] = Field(None, description="인식된 카드 ID")
