import os
from typing import List
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever

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
        model="llama3:8b",
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.3
    )
    
    # Multi-Query Retriever 생성
    print("\n쿼리 변환 중...")
    multi_query_retriever = create_multi_query_retriever(base_retriever, query_transform_llm)
    
    # 검색 실행 (내부적으로 쿼리 변환 → 임베딩 → 검색 자동 수행)
    print("멀티 쿼리 검색 실행 중...")
    retrieved_docs = multi_query_retriever.invoke(query)
    
    print(f"검색 완료: {len(retrieved_docs)}개 문서 (중복 제거 후)")
    print("="*80 + "\n")
    
    return retrieved_docs


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