from FlagEmbedding import BGEM3FlagModel # 임베딩 모델 
from sentence_transformers import CrossEncoder # Reranker 모델
from qdrant_client import QdrantClient # Qdrant 클라이언트
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny,
    SparseVector, NamedSparseVector
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

import json
from typing import Any, Tuple, List, Callable

from core.qdrant_upsert import embedding_model, qdrant_connection, COL
from core.qdrant_upsert_utils import _to_list, _extract_sparse
from core.config import settings

# 카드 별칭 정의 맵 로드
with open("./data/alias_map.json", "r", encoding="utf-8") as f:
    CARD_ALIAS_TO_CANONICAL = json.load(f)

canonical_names = list(set(CARD_ALIAS_TO_CANONICAL.values()))
canonical_vectors = embedding_model.encode(
    canonical_names,
    return_dense=True,
    return_sparse=False
)['dense_vecs']
canonical_card_vectors = {
    name: vec for name, vec in zip(canonical_names, canonical_vectors)
}

# Reranker 모델 로드
reranker = CrossEncoder(
    model_name="BAAI/bge-reranker-base",
    max_length=512,
    device=settings.DEVICE
)

def encode_query(embed_model: BGEM3FlagModel, text: str) -> Tuple[List[float], SparseVector]:
    """
    사용자 쿼리 텍스트를 임베딩 모델을 이용하여 밀집 벡터와 희소 벡터로 변환합니다.
    Args:
        - embed_model: 임베딩 모델 인스턴스
        - text: 임베딩할 쿼리 문자열
    Returns:
        - dense: 밀집 벡터 리스트
        - sparse: 희소 벡터 SparseVector 인스턴스
    """
    out = embed_model.encode_queries([text], return_dense=True, return_sparse=True)
    dense = _to_list(out["dense_vecs"][0])
    sparse = _extract_sparse(out)
    return dense, sparse

def smart_filter_router(query: str, query_dense_vec: List, threshold=0.5) -> Filter | None:
    """
    스마트 필터 라우터 (2단계로 구성된 로직)
        1. 키워드 매칭(MatchAny)
        2. 실패 시 유사도 매칭(MatchValue)
    Args:
        - query: 사용자 쿼리 문자열
        - query_dense_vec: 쿼리의 밀집 벡터 리스트
        - threshold: 유사도 임계값 (기본값: 0.5)
    Returns:
        - Filter 인스턴스 또는 None
    """

    # --- 1단계: 빠른 키워드/별칭 검색 (MatchAny) ---
    found_canonical_names = set()
    sorted_aliases = sorted(CARD_ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)
    temp_query = query

    for alias in sorted_aliases:
        # 별칭이 쿼리에 포함되어 있다면 올바른 명칭으로 대체
        if alias in temp_query:
            canonical_name = CARD_ALIAS_TO_CANONICAL[alias]
            found_canonical_names.add(canonical_name)
            temp_query = temp_query.replace(alias, "")

    if found_canonical_names:
        names_list = list(found_canonical_names)
        print(f"[Debug] (Step 1: Keyword) {len(names_list)}개 카드명 감지. 'card_id' IN {names_list} 필터 적용.")
        return Filter(must=[FieldCondition(key="card_id", match=MatchAny(any=names_list))])

    # --- 2단계: 유사도 기반 재시도 (1단계 실패 시) ---
    print("[Debug] (Step 1: Keyword) 감지 실패. (Step 2: Similarity) 실행...")
    best_match_name = None
    best_match_score = -1.0

    q_vec = np.array(query_dense_vec).reshape(1, -1)

    for name, vec in canonical_card_vectors.items():
        score = cosine_similarity(q_vec, vec.reshape(1, -1))[0][0]
        if score > best_match_score:
            best_match_score = score
            best_match_name = name

    if best_match_score > threshold:
        print(f"[Debug] (Step 2: Similarity) '{best_match_name}'이 임계값({threshold}) 이상 ({best_match_score:.2f}) 감지. 'card_id' == '{best_match_name}' 필터 적용.")
        return Filter(must=[FieldCondition(key="card_id", match=MatchValue(value=best_match_name))])

    print(f"[Debug] (Step 2: Similarity) 유사도 높은 카드 없음 (Best: {best_match_name}, Score: {best_match_score:.2f}). 필터 미적용.")
    return None

def hybrid_search(query: str, topk: int=10,) -> List[Tuple[str, float, Any]]:
    """
    하이브리드 검색 함수 (스마트 필터 + Reranker 사용)
    Args:
        - client: QdrantClient 인스턴스
        - target_collection: 검색 대상 Qdrant 컬렉션 이름
        - reranker: 문서 재정렬용 CrossEncoder 인스턴스
        - query: 사용자 쿼리 문자열
        - topk: 최종 반환할 상위 문서 수
    Returns:
        - List of (doc_id, rerank_score, point) 튜플 리스트
    """
    global qdrant_connection, COL, embedding_model, reranker
    # 1. 쿼리 인코딩
    q_dense, q_sparse = encode_query(embedding_model, query)
    print(f"\n[Debug] 쿼리 임베딩 완료. (Dense vector length: {len(q_dense)}, Sparse vector non-zero count: {len(q_sparse.indices)})")

    # 2. 스마트 필터 실행
    query_filter = smart_filter_router(query, q_dense)
    print(f"[Debug] 스마트 필터 생성 완료: {query_filter}")

    # 3. 1차 검색 (후보군 수집) - 필터 적용
    dres = qdrant_connection.search(
        collection_name=COL,
        query_vector=q_dense,
        query_filter=query_filter,
        limit=topk*3
    )
    sres = qdrant_connection.search(
        collection_name=COL,
        query_vector=NamedSparseVector(
            name="sparse",
            vector=SparseVector(
                indices=q_sparse.indices,
                values=q_sparse.values
            )
        ),
        query_filter=query_filter,
        limit=topk*3
    )
    print(f"[Debug] 1차 검색 완료. (Dense hits: {len(dres)}, Sparse hits: {len(sres)})")

    # 4. 후보군 ID 취합
    candidate_ids = set()
    for r in dres: candidate_ids.add(r.id)
    for r in sres: candidate_ids.add(r.id)
    if not candidate_ids: return []

    # 5. 후보군 원본 텍스트 Retrieve
    candidate_points = qdrant_connection.retrieve(
        collection_name=COL, ids=list(candidate_ids),
        with_payload=True
    )
    print(f"[Debug] 후보군 원본 텍스트 조회 완료. (총 {len(candidate_points)}개 문서)")

    # 6. Reranker 입력 생성
    rerank_pairs = []
    print(f"\n[Debug] Reranking {len(candidate_points)} candidates for query: '{query}'")
    for pt in candidate_points:
        # [중요] payload의 'full_text'를 사용 (적재 스크립트에서 저장 필수)
        doc_text = pt.payload.get('full_text', '')
        if not doc_text:
             print(f"[Warning] Doc ID {pt.id} is missing 'full_text'. Falling back to 'preview'.")
             doc_text = pt.payload.get('preview', '') # ⬅️ 비상시 preview 사용
        rerank_pairs.append( (query, doc_text) )
    print(f"[Debug] Reranker 입력 쌍 생성 완료.")

    # 7. Reranking 실행
    rerank_scores = reranker.predict(rerank_pairs)
    print(f"[Debug] Reranking 완료.")

    # 8. 최종 결과 정렬
    reranked_results = []
    for score, pt in zip(rerank_scores, candidate_points):
        reranked_results.append( (pt.id, score, pt) )
    print(f"[Debug] 최종 결과 정렬 완료. Top-{topk} 문서 반환 준비.")

    reranked_results.sort(key=lambda x: x[1], reverse=True) # ⬅️ Reranker 점수로 정렬
    return reranked_results[:topk]

def print_rag_documents(docs: List[Document]) -> None:
    """ RAG 디버깅 용 출력 함수 """
    print(f"\n--- [Debug] {len(docs)}개의 Rerank된 문서를 RAG Chain (LLM)에 전달 ---")
    if not docs:
        print("전달할 문서가 없습니다.")
        return

    for rank, doc in enumerate(docs, start=1):
        meta = doc.metadata
        path = f"{meta.get('tag_major')} > {meta.get('tag_middle')} > {meta.get('tag_minor')}"
        # ⬇️ Reranker score는 .2f (소수점 2자리)로 표시
        print(f"{rank}. score={meta.get('score', 0.0):.2f} | {meta.get('card_id')} | {meta.get('section')} | {meta.get('granularity')}")
        print(f"   doc_id: {meta.get('doc_id')} (UUID: {meta.get('pid')})")
        print(f"   path  : {path}")
        print(f"   prev  : {meta.get('preview','')}")
        print()

class CustomHybridRetriever(BaseRetriever):
    search_func: Callable
    top_k: int = 6

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # 1. Reranker + Filter가 적용된 하이브리드 검색 실행
        results = self.search_func(query, topk=self.top_k) # ⬅️ alpha 없음

        # 2. 랭체인 'Document' 객체로 변환
        docs = []
        for (pid, sc, pt) in results:
            if pt:
                pl = pt.payload
                # ⬇️ [수정됨] LLM이 Reranker와 동일한 full_text를 보도록 수정
                content = pl.get("full_text", pl.get("preview", ""))

                metadata = {
                    "score": sc, # ⬅️ Reranker 점수
                    "pid": pid,
                    "card_id": pl.get("card_id"),
                    "doc_id": pl.get("doc_id"),
                    "section": pl.get("section_canonical"),
                    "granularity": pl.get("granularity"),
                    "tag_major": pl.get("tag_major"),
                    "tag_middle": pl.get("tag_middle"),
                    "tag_minor": pl.get("tag_minor"),
                    "preview": pl.get("preview","")
                }
                docs.append(Document(page_content=content, metadata=metadata))

        # ⬇️ RAG용 디버깅 함수 호출
        print_rag_documents(docs)
        return docs

def format_docs(docs: List[Document]) -> str:
    """ RAG 프롬프트 내에 문서들을 포맷팅하여 삽입하는 함수 """
    return "\n\n".join(
        f"--- (출처: {doc.metadata['card_id']} / {doc.metadata['section']}) ---\n{doc.page_content}"
        for doc in docs
    )


def get_card_description(query: str, top_k: int = 6) -> dict:
    """
    특정 카드의 혜택, 연회비, 설명 등을 제공하는 함수
    
    사용자 쿼리에서 카드명을 추출하고, 해당 카드의 문서만 검색하여
    카드 설명을 생성합니다. 기존 hybrid_search와 CustomHybridRetriever를 활용합니다.
    
    Args:
        query: 사용자 쿼리 (예: "우리카드 7CORE 연회비 얼마야?")
        top_k: 검색할 문서 수 (기본값: 6)
        
    Returns:
        dict: {
            'answer': str - 생성된 답변,
            'card_id': str - 인식된 카드 ID (없으면 None)
        }
    """
    print(f"\n[카드 설명] 쿼리: '{query}'")
    
    # 1. 쿼리 임베딩 생성
    q_dense, _ = encode_query(embedding_model, query)
    
    # 2. 카드명 추출 (스마트 필터 라우터 사용)
    query_filter = smart_filter_router(query, q_dense, threshold=0.5)
    
    # 카드명을 찾지 못한 경우
    if query_filter is None:
        print("[카드 설명] 카드명을 인식할 수 없습니다.")
        return {
            'answer': "죄송합니다. 질문에서 카드명을 인식할 수 없습니다. 카드 이름을 명확히 말씀해 주세요. (예: '우리카드 7CORE', '카드의정석 every point' 등)",
            'card_id': None
        }
    
    # 3. 인식된 카드 ID 추출
    recognized_card_id = None
    if query_filter.must:
        condition = query_filter.must[0]
        if hasattr(condition.match, 'value'):
            recognized_card_id = condition.match.value
        elif hasattr(condition.match, 'any'):
            recognized_card_id = condition.match.any[0] if condition.match.any else None
    
    print(f"[카드 설명] 인식된 카드: {recognized_card_id}")
    
    # 4. 기존 hybrid_search 함수 사용 (이미 필터 적용됨)
    results = hybrid_search(query, topk=top_k)
    
    if not results:
        return {
            'answer': f"'{recognized_card_id}' 카드의 정보를 찾을 수 없습니다.",
            'card_id': recognized_card_id
        }
    
    # 5. 기존 CustomHybridRetriever의 로직 활용 - Document 객체로 변환
    docs = []
    for (pid, sc, pt) in results:
        if pt:
            pl = pt.payload
            content = pl.get("full_text", pl.get("preview", ""))
            metadata = {
                "score": sc,
                "pid": pid,
                "card_id": pl.get("card_id"),
                "doc_id": pl.get("doc_id"),
                "section": pl.get("section_canonical"),
                "granularity": pl.get("granularity"),
                "tag_major": pl.get("tag_major"),
                "tag_middle": pl.get("tag_middle"),
                "tag_minor": pl.get("tag_minor"),
                "preview": pl.get("preview","")
            }
            docs.append(Document(page_content=content, metadata=metadata))
    
    print(f"[카드 설명] {len(docs)}개 문서 검색 완료")
    
    # 6. LLM 인스턴스 생성
    generation_llm = ChatOllama(
        model=settings.OLLAMA_MODEL_NAME,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0
    )
    
    # 7. 챗봇 UI 최적화 프롬프트
    template = """당신은 우리카드 AI 상담사입니다. 사용자의 질문에 친절하고 정확하게 답변하세요.

# 답변 형식 (질문 유형별)

## 1️⃣ 간단한 정보 질문 (연회비, 혜택률 등)
→ 간결하게 1-2문장으로 핵심만 전달
예: "우리카드 7CORE의 연회비는 50,000원입니다."

## 2️⃣ 카드 요약 요청
→ 구조화된 요약 제공
```
💳 [카드명]

📌 핵심 특징
• 주요 혜택 1
• 주요 혜택 2
• 주요 혜택 3

💰 비용
• 연회비: XX원
• 전월실적: XX원 이상

⚠️ 주의사항
• 유의사항 1
• 유의사항 2
```

## 3️⃣ 여러 카드 비교 요청
→ 비교표 형식으로 제공
```
📊 카드 비교

🔵 [카드1]
• 연회비: XX원
• 핵심혜택: ...
• 적합대상: ...

🟢 [카드2]
• 연회비: XX원
• 핵심혜택: ...
• 적합대상: ...

💡 추천
- [카드1]은 ~한 분께 추천
- [카드2]는 ~한 분께 추천
```

## 4️⃣ 상세 설명 요청
→ 섹션별로 구분하여 상세 설명
```
📋 [카드명] 상세 정보

🎁 혜택
• 혜택 1: 상세 설명
• 혜택 2: 상세 설명

💵 연회비 및 실적
• 연회비: ...
• 전월실적 조건: ...

📝 이용 안내
• 주의사항 1
• 주의사항 2
```

# 답변 규칙
✅ Context 정보만 사용 (없으면 "제공되지 않음" 표기)
✅ 구체적인 수치 명시 (연회비, 할인율, 한도 등)
✅ 챗봇 UI에 보기 좋게 줄바꿈과 이모지 활용
✅ 카드명은 정확히 표기
✅ 전문 용어는 쉽게 풀어서 설명

# 카드 정보 (Context)
{context}

# 사용자 질문
{question}

# 답변
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 8. 기존 format_docs 함수 활용
    context = format_docs(docs)
    
    # 9. 답변 생성 (기존 LCEL 패턴 활용)
    chain = prompt | generation_llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    
    print(f"[카드 설명] 답변 생성 완료\n")
    
    return {
        'answer': answer,
        'card_id': recognized_card_id
    }
