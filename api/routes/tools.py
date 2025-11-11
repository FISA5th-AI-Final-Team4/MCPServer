from fastapi import APIRouter, HTTPException

from mcp_module.RAG.card_desc_rag_based import card_recommendation_rag_pipeline
from mcp_module.RAG.faq_rag_based import search_faq
from mcp_module.RAG.term_rag_based import search_term
from schemas.card_recommendation import (
    CardRecommendationRequest,
    CardRecommendationResponse,
    ContextDocument
)
from schemas.qna import (
    FAQRequest,
    FAQResponse,
    FAQResult,
    TermRequest,
    TermResponse,
    TermInfo,
    RelatedTerm
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


@router.post(
    "/faq-query",
    response_model=FAQResponse,
    summary="FAQ 검색",
    description="사용자 질문에 대한 FAQ를 검색하여 답변을 제공합니다.",
    operation_id="query_faq_database"
)
async def faq_query(request: FAQRequest) -> FAQResponse:
    """
    FAQ 검색 및 답변 생성
    
    LLM 서버에서 query_faq_database Tool을 통해 호출되며,
    PostgreSQL에서 유사도 검색으로 FAQ를 찾아 답변을 생성합니다.
    """
    
    try:
        print(f"\n[API] FAQ 검색 요청: {request.query}")
        
        result = search_faq(
            question=request.query,
            top_k=request.top_k
        )
        
        faq_results = []
        if result["success"] and result["results"]:
            faq_results = [
                FAQResult(
                    faq_id=faq["faq_id"],
                    question=faq["question"],
                    answer=faq["answer"],
                    keywords=faq["keywords"],
                    category_name=faq["category_name"],
                    similarity=float(faq["similarity"]),
                    views=faq["views"],
                    priority=faq["priority"]
                )
                for faq in result["results"]
            ]
        
        response = FAQResponse(
            success=result["success"],
            query=result["query"],
            answer=result["answer"],
            results=faq_results,
            total_found=result["total_found"],
            best_similarity=result.get("best_similarity")
        )
        
        print(f"[API] FAQ 검색 완료: {result['total_found']}개 발견\n")
        
        return response
        
    except Exception as e:
        print(f"[API 오류] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"FAQ 검색 중 오류: {str(e)}"
        )


@router.post(
    "/term-query",
    response_model=TermResponse,
    summary="금융 용어 검색",
    description="금융 용어를 검색하여 정의와 관련 정보를 제공합니다.",
    operation_id="query_term_database"
)
async def term_query(request: TermRequest) -> TermResponse:
    """
    금융 용어 검색 및 설명 생성
    
    LLM 서버에서 query_term_database Tool을 통해 호출되며,
    PostgreSQL에서 용어를 검색하여 정의와 관련 용어를 제공합니다.
    """
    
    try:
        print(f"\n[API] 용어 검색 요청: {request.query}")
        
        result = search_term(term_query=request.query)
        
        term_info = None
        related_terms = []
        
        if result["success"] and result["term_info"]:
            term_data = result["term_info"]
            term_info = TermInfo(
                term_id=term_data["term_id"],
                term=term_data["term"],
                definition=term_data["definition"],
                english=term_data.get("english"),
                related_terms=term_data.get("related_terms", []),
                examples=term_data.get("examples"),
                category_name=term_data["category_name"],
                similarity=float(term_data["similarity"])
            )
            
            related_terms = [
                RelatedTerm(term=rt["term"], definition=rt["definition"])
                for rt in result.get("related_terms", [])
            ]
        
        response = TermResponse(
            success=result["success"],
            query=result["query"],
            answer=result["answer"],
            term_info=term_info,
            related_terms=related_terms,
            similarity=result.get("similarity")
        )
        
        print(f"[API] 용어 검색 완료\n")
        
        return response
        
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
