from fastapi import APIRouter, HTTPException

from mcp_module.RAG.card_desc_rag_based import card_recommendation_rag_pipeline
from schemas.card_recommendation import (
    CardRecommendationRequest,
    CardRecommendationResponse,
    ContextDocument
)


router = APIRouter(prefix="/tools", tags=["mcp-tools"])


# ============================================================================
# 엔드포인트
# ============================================================================

@router.post(
    "/card-recommendation",
    response_model=CardRecommendationResponse,
    summary="카드 추천 RAG 파이프라인",
    description="사용자 질문을 받아 RAG 파이프라인을 실행하여 카드를 추천합니다.",
    operation_id="get_card_recommendation"
)
async def card_recommendation(request: CardRecommendationRequest) -> CardRecommendationResponse:
    """
    카드 추천 RAG 파이프라인 실행
    
    LLM 서버에서 쿼리 라우팅을 통해 호출되며,
    벡터 DB 검색 → 컨텍스트 선정 → 최종 답변 생성을 수행합니다.
    """
    
    try:
        print(f"\n[API] 카드 추천 요청: {request.query}")
        
        # RAG 파이프라인 실행
        result = card_recommendation_rag_pipeline(
            query=request.query,
            retrieve_k=request.retrieve_k,
            final_k=request.final_k
        )
        
        # 응답 변환
        response = CardRecommendationResponse(
            query=result["query"],
            retrieved_count=result["retrieved_count"],
            final_count=result["final_count"],
            answer=result["answer"],
            context_docs=[
                ContextDocument(
                    content=doc["content"],
                    metadata=doc["metadata"]
                )
                for doc in result["context_docs"]
            ]
        )
        
        print(f"[API] 카드 추천 완료\n")
        
        return response
        
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
        "available_tools": ["card-recommendation"]
    }
