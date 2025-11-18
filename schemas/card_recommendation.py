"""
카드 추천 API 요청/응답 스키마
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class CardRecommendationRequest(BaseModel):
    """카드 추천 요청"""
    query: str = Field(..., description="사용자의 카드 추천 질문", example="편의점 할인 카드 추천")
    retrieve_k: Optional[int] = Field(20, description="초기 검색 문서 수", ge=5, le=50)
    final_k: Optional[int] = Field(5, description="최종 선정 문서 수", ge=1, le=10)


class ContextDocument(BaseModel):
    """컨텍스트 문서"""
    content: str = Field(..., description="문서 내용")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="문서 메타데이터")


class CardRecommendationResponse(BaseModel):
    """카드 추천 응답"""
    query: str = Field(..., description="입력 쿼리")
    retrieved_count: int = Field(..., description="검색된 문서 수")
    final_count: int = Field(..., description="최종 선정 문서 수")
    answer: str = Field(..., description="생성된 카드 추천 답변")
    context_docs: List[ContextDocument] = Field(..., description="사용된 컨텍스트 문서")
