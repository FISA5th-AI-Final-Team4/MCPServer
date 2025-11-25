from fastapi import APIRouter, HTTPException

from mcp_module.RAG.faq_rag_based import search_faq
from mcp_module.RAG.retrieval_qdrant import card_desc_hybrid_search_generation
from mcp_module.RAG.term_rag_based import search_term
from schemas.card_recommendation import CardRecommendationRequest
from schemas.qna import FAQRequest, TermRequest


router = APIRouter(prefix="/tools", tags=["mcp-tools"])


# ============================================================================
# 엔드포인트
# ============================================================================

@router.post(
    "/card-recommendation",
    summary="카드 추천 RAG 파이프라인",
    description="사용자 질문을 받아 RAG 파이프라인을 실행하여 카드를 추천합니다.",
    operation_id="get_card_recommendation"
)
async def card_recommendation(request: CardRecommendationRequest):
    """
    카드 추천 RAG 파이프라인 실행
    
    LLM 서버에서 쿼리 라우팅을 통해 호출되며,
    벡터 DB 검색 → 컨텍스트 선정 → 최종 답변 생성을 수행합니다.
    """
    
    try:
        print(f"\n[API] 카드 추천 요청: {request.query}")
        
        answer = card_desc_hybrid_search_generation(request.query)
        
        print(f"[API] 카드 추천 완료\n")
        
        return {"answer": answer}
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"벡터 DB를 찾을 수 없습니다: {str(e)}"
        )
        
    except Exception as e:
        print(f"[API 오류] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"카드 추천 실행 중 오류: {str(e)}"
        )


@router.post(
    "/faq-query",
    summary="FAQ 검색",
    description="사용자 질문에 대한 FAQ를 검색하여 답변을 제공합니다.",
    operation_id="query_faq_database"
)
async def faq_query(request: FAQRequest):
    """
    FAQ 검색 및 답변 생성
    
    LLM 서버에서 query_faq_database Tool을 통해 호출되며,
    PostgreSQL에서 유사도 검색으로 FAQ를 찾아 답변을 생성합니다.
    """
    
    try:
        print(f"\n[API] FAQ 검색 요청: {request.query}")
        
        answer = search_faq(
            question=request.query,
            top_k=request.top_k
        )
        
        print(f"[API] FAQ 검색 완료\n")
        
        return answer
        
    except Exception as e:
        print(f"[API 오류] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"FAQ 검색 중 오류: {str(e)}"
        )


@router.post(
    "/term-query",
    summary="금융 용어 검색",
    description="금융 용어를 검색하여 정의와 관련 정보를 제공합니다.",
    operation_id="query_term_database"
)
async def term_query(request: TermRequest):
    """
    금융 용어 검색 및 설명 생성
    
    LLM 서버에서 query_term_database Tool을 통해 호출되며,
    PostgreSQL에서 용어를 검색하여 정의와 관련 용어를 제공합니다.
    """
    
    try:
        print(f"\n[API] 용어 검색 요청: {request.query}")
        
        answer = search_term(term_query=request.query)
        
        print(f"[API] 용어 검색 완료\n")
        
        return {"answer": answer}
        
    except Exception as e:
        print(f"[API 오류] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"용어 검색 중 오류: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check",
    operation_id="health_check"
)
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "service": "MCP Tools Server",
        "available_tools": [
            "card-recommendation",
            "faq-query",
            "term-query"
        ]
    }
