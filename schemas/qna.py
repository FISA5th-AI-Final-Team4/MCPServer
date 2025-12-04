"""
FAQ 및 금융 용어 검색 요청/응답 스키마
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ============================================================================
# FAQ 검색 스키마
# ============================================================================

class FAQRequest(BaseModel):
    """FAQ 검색 요청"""
    query: str = Field(..., description="사용자 질문", min_length=1)
    top_k: int = Field(default=3, description="반환할 최대 FAQ 개수", ge=3, le=10)


class FAQResult(BaseModel):
    """개별 FAQ 결과"""
    faq_id: str
    question: str
    answer: str
    keywords: List[str]
    category_name: str
    similarity: float
    views: int
    priority: int


class FAQResponse(BaseModel):
    """FAQ 검색 응답"""
    success: bool = Field(..., description="검색 성공 여부")
    query: str = Field(..., description="원본 질문")
    answer: str = Field(..., description="생성된 답변")
    results: List[FAQResult] = Field(default=[], description="검색된 FAQ 리스트")
    total_found: int = Field(..., description="검색된 FAQ 총 개수")
    best_similarity: Optional[float] = Field(None, description="최고 유사도")


# ============================================================================
# 금융 용어 검색 스키마
# ============================================================================

class TermRequest(BaseModel):
    """금융 용어 검색 요청"""
    query: str = Field(..., description="검색할 용어", min_length=1)
    top_k: int = Field(default=3, description="반환할 최대 용어 개수", ge=1, le=10)


class TermResponse(BaseModel):
    """
    금융 용어 검색 응답
    
    FAQ와 동일한 형식으로 반환:
    - answer: 최상위 용어 정의
    - relatedQuestions: 유사도 기준 2, 3순위 용어명 리스트
    """
    answer: str = Field(..., description="최상위 용어 정의 및 설명")
    relatedQuestions: List[str] = Field(default=[], description="유사도 순 관련 용어 목록 (최대 2개)")
