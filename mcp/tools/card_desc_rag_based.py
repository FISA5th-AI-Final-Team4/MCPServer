"""
쿼리 변환 (Query Transformation) - Multi-Query Retrieval

하나의 사용자 쿼리를 여러 관점의 쿼리로 확장하여
벡터 DB 검색의 커버리지를 높이는 기능을 제공합니다.
"""

from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever


# ============================================================================
# 쿼리 변환 (Query Transformation) 
# ============================================================================

def create_multi_query_retriever(base_retriever, llm) -> MultiQueryRetriever:
    """
    원본 쿼리를 3개의 다른 관점 쿼리로 변환하여 총 4개 쿼리로 검색
    검색 결과를 합치고 중복 제거하여 반환
    """
    
    query_transform_prompt = PromptTemplate(
        input_variables=["question"],
        template="""당신은 AI 검색 보조자입니다.
사용자의 질문을 벡터 데이터베이스에서 검색하기 위해
다양한 관점으로 재해석하는 것이 목표입니다.

사용자의 질문을 **서로 다른 3개의 검색 쿼리**로 생성하세요.

규칙:
- 정확히 3개의 쿼리만 생성
- 각 쿼리는 원본 질문의 다른 측면을 포착
- 더 구체적이거나 더 넓은 범위 가능
- 각 쿼리는 한 줄로 작성
- 번호나 기호 없이 쿼리만 출력
- 개행으로 구분

원본 질문: {question}

대체 검색 쿼리:
"""
    )
    
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=query_transform_prompt,
        include_original=True
    )
    
    return multi_query_retriever