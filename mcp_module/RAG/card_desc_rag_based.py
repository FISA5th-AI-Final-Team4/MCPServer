import os
from typing import List, Dict, Any
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from core.config import settings


# 전역 설정
VECTOR_DB_PATH = Path(__file__).parent.parent.parent / "data" / "VectorDB"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================================
# 쿼리 변환 (Query Transformation) 
# ============================================================================
"""
쿼리 변환 (Query Transformation) - Multi-Query Retrieval

하나의 사용자 쿼리를 여러 관점의 쿼리로 확장하여
벡터 DB 검색의 커버리지를 높이는 기능을 제공.
"""

def generate_alternative_queries(query: str, llm) -> List[str]:
    """
    원본 쿼리를 3개의 다른 관점 쿼리로 변환
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
    
    # LLM을 사용하여 대체 쿼리 생성
    chain = query_transform_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": query})
    
    # 결과를 라인별로 분리하여 리스트로 변환
    alternative_queries = [q.strip() for q in result.strip().split('\n') if q.strip()]
    
    # 원본 쿼리 포함하여 반환
    all_queries = [query] + alternative_queries
    
    return all_queries


# ============================================================================
# 검색 실행 (Search Execution)
# ============================================================================

def load_vector_db() -> FAISS:
    """
    FAISS 벡터 DB 로드
    문서 임베딩과 동일한 paraphrase-multilingual-MiniLM-L12-v2 모델 사용
    """
    
    print("\n" + "="*80)
    print("벡터 DB 로드")
    print("="*80)
    
    # HuggingFace 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,  #sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"임베딩 모델: {EMBEDDING_MODEL_NAME}")
    
    # FAISS 인덱스 로드
    faiss_index_path = str(VECTOR_DB_PATH)
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"벡터 DB를 찾을 수 없습니다: {faiss_index_path}")
    
    vectorstore = FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print(f"벡터 DB 로드 완료: {faiss_index_path}")
    print("="*80 + "\n")
    
    return vectorstore


def create_base_retriever(vectorstore: FAISS, k: int = 20):
    """벡터 스토어로부터 기본 리트리버 생성 (Top-K 유사도 검색)"""
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever


def execute_multi_query_search(query: str, k: int = 20) -> List[Document]:
    """
    Multi-Query Retrieval 전체 실행
    
    흐름:
    1. 벡터 DB 로드 (임베딩 모델 포함)
    2. 기본 리트리버 생성
    3. LLM으로 쿼리 변환 (1개 → 4개)
    4. 각 쿼리를 임베딩 → 벡터 검색
    5. 결과 합치기 + 중복 제거
    """
    
    print("\n" + "="*80)
    print("쿼리 변환 및 검색 실행")
    print("="*80)
    print(f"입력 쿼리: {query}")
    print(f"검색 문서 수(K): {k}")
    
    # 벡터 DB 로드 (임베딩 모델 포함)
    vectorstore = load_vector_db()
    
    # 기본 리트리버 생성
    base_retriever = create_base_retriever(vectorstore, k=k)
    
    # 쿼리 변환용 경량 LLM
    query_transform_llm = ChatOllama(
        model=settings.OLLAMA_MODEL_NAME,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.3
    )
    
    # 쿼리 변환 (1개 → 4개)
    print("\n쿼리 변환 중...")
    all_queries = generate_alternative_queries(query, query_transform_llm)
    print(f"생성된 쿼리 수: {len(all_queries)}")
    for i, q in enumerate(all_queries, 1):
        print(f"  {i}. {q}")
    
    # 각 쿼리로 검색 실행
    print("\n멀티 쿼리 검색 실행 중...")
    all_retrieved_docs = []
    seen_contents = set()
    
    for i, q in enumerate(all_queries, 1):
        print(f"  검색 {i}/{len(all_queries)}: {q[:50]}...")
        docs = base_retriever.invoke(q)
        
        # 중복 제거
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_retrieved_docs.append(doc)
    
    print(f"검색 완료: {len(all_retrieved_docs)}개 문서 (중복 제거 후)")
    print("="*80 + "\n")
    
    return all_retrieved_docs


# ============================================================================
# 결과 후처리 및 최종 컨텍스트 선정
# ============================================================================

def postprocess_and_select_documents(
    query: str,
    retrieved_docs: List[Document],
    top_n: int = 5
) -> List[Document]:
    """
    검색된 문서를 후처리하여 최종 컨텍스트 선정
    
    현재는 단순히 상위 N개 선택
    향후 개선 가능:
    - CohereRerank로 재정렬
    - LLMChainExtractor로 핵심 구절 추출
    - 관련성 스코어링
    """
    
    print("\n" + "="*80)
    print("결과 후처리 및 최종 컨텍스트 선정")
    print("="*80)
    print(f"입력 문서 수: {len(retrieved_docs)}")
    print(f"목표 문서 수: {top_n}")
    
    # 상위 N개 문서 선택
    final_docs = retrieved_docs[:top_n]
    
    print(f"\n최종 선정: {len(final_docs)}개 문서")
    
    # 선정된 문서 미리보기
    for i, doc in enumerate(final_docs, 1):
        preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
        print(f"  문서 {i}: {preview}")
    
    print("="*80 + "\n")
    
    return final_docs


# ============================================================================
# Ollama 답변 생성
# ============================================================================

def create_card_recommendation_prompt() -> ChatPromptTemplate:
    """
    카드 추천 전문가 프롬프트 생성
    
    비즈니스 요구사항:
    1. 추천 이유
    2. 상황별 추천
    3. 상품 설명 + 상황별 혜택
    """
    
    system_message = """당신은 전문 카드 추천 상담사입니다.

제공된 카드 정보를 바탕으로 다음 3가지를 반드시 포함하여 답변하세요:

1. **추천 이유**: 
   - 고객의 소비 패턴/상황에 이 카드가 적합한 이유
   - 다른 카드 대비 차별화된 장점

2. **상황별 카드 추천**:
   - 고객의 주요 사용 상황별로 가장 적합한 카드 추천
   - 예: "편의점을 자주 이용하신다면 A카드", "쇼핑몰 결제가 많으시다면 B카드"
   - 각 상황에 맞는 구체적인 카드명과 혜택 제시

3. **상품 설명 + 상황별 혜택**:
   - 추천한 카드의 주요 특징 및 스펙
   - 카테고리별 할인율/적립률 (편의점, 카페, 대중교통 등)
   - 연회비 및 혜택 조건
   - 실제 사용 시나리오별 예상 혜택

제약:
- 컨텍스트에 없는 정보는 추측하지 마세요
- 구체적인 수치와 카드명을 포함하세요
- 전문적이면서도 친근한 어조로 작성하세요

컨텍스트:
{context}
"""
    
    human_message = """고객 질문: {question}

위 질문에 대해 3가지 항목(추천 이유, 상황별 카드 추천, 상품 설명+혜택)을 포함하여 답변해주세요."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])
    
    return prompt


def generate_final_answer(query: str, context_docs: List[Document]) -> str:
    """
    최종 컨텍스트와 쿼리를 Ollama에 전달하여 답변 생성
    
    고성능 LLM(llama3.1:70b 또는 llama3:70b-instruct) 사용
    """
    
    print("\n" + "="*80)
    print("최종 답변 생성")
    print("="*80)
    print(f"쿼리: {query}")
    print(f"컨텍스트 문서 수: {len(context_docs)}")
    
    # 고성능 생성 LLM
    generation_llm = ChatOllama(
        model=settings.OLLAMA_MODEL_NAME,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.7,
        top_p=0.9
    )
    
    # 컨텍스트 문서를 하나의 문자열로 결합
    context_text = "\n\n".join([
        f"[문서 {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(context_docs)
    ])
    
    # 프롬프트 생성
    prompt = create_card_recommendation_prompt()
    
    # RAG 체인 구성 (LCEL)
    rag_chain = prompt | generation_llm | StrOutputParser()
    
    # 답변 생성
    print("\nOllama 답변 생성 중...")
    final_answer = rag_chain.invoke({
        "context": context_text,
        "question": query
    })
    
    print("답변 생성 완료")
    print("="*80 + "\n")
    
    return final_answer


# ============================================================================
# 통합 파이프라인
# ============================================================================

def card_recommendation_rag_pipeline(
    query: str,
    retrieve_k: int = 20,
    final_k: int = 5
) -> Dict[str, Any]:
    """
    카드 추천 RAG 파이프라인 전체 실행
    
    LLM 서버에서 쿼리 라우팅을 통해 호출되며,
    전처리된 쿼리를 입력받아 최종 답변을 생성합니다.
    
    파이프라인 흐름:
    1. 쿼리 변환 (1개 → 4개)
    2. 벡터 DB 검색 (임베딩 모델 사용)
    3. 결과 후처리 및 컨텍스트 선정
    4. Ollama를 통한 최종 답변 생성
    
    Args:
        query (str): 전처리된 사용자 쿼리
        retrieve_k (int): 초기 검색 문서 수
        final_k (int): 최종 선정 문서 수
        
    Returns:
        Dict[str, Any]: 다음 정보를 포함하는 딕셔너리
            - query: 입력 쿼리
            - retrieved_count: 검색된 문서 수
            - final_count: 최종 선정 문서 수
            - answer: 생성된 최종 답변
            - context_docs: 사용된 컨텍스트 문서 리스트
    """
    
    print("\n" + "="*80)
    print("카드 추천 RAG 파이프라인 시작")
    print("="*80)
    print(f"입력 쿼리: {query}")
    print(f"초기 검색 문서 수: {retrieve_k}")
    print(f"최종 선정 문서 수: {final_k}")
    print("="*80)
    
    try:
        # 쿼리 변환 및 멀티 쿼리 검색
        retrieved_docs = execute_multi_query_search(query, k=retrieve_k)
        
        # 결과 후처리 및 최종 컨텍스트 선정
        final_docs = postprocess_and_select_documents(query, retrieved_docs, top_n=final_k)
        
        # 최종 답변 생성
        final_answer = generate_final_answer(query, final_docs)
        
        # 결과 반환
        result = {
            "query": query,
            "retrieved_count": len(retrieved_docs),
            "final_count": len(final_docs),
            "answer": final_answer,
            "context_docs": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in final_docs
            ]
        }
        
        print("\n" + "="*80)
        print("카드 추천 RAG 파이프라인 완료")
        print("="*80 + "\n")
        
        return result
        
    except Exception as e:
        print(f"\n[오류] 파이프라인 실행 중 오류 발생: {str(e)}")
        raise