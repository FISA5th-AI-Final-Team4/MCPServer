from FlagEmbedding import BGEM3FlagModel # 임베딩 모델 
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
# 새로운 구조: {"카드ID": ["별칭1", "별칭2", ...]}
with open("./data/alias_map.json", "r", encoding="utf-8") as f:
    CARD_ID_TO_ALIASES = json.load(f)

# 역방향 매핑 생성: {"별칭": "카드ID"}
ALIAS_TO_CARD_ID = {}
for card_id, aliases in CARD_ID_TO_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CARD_ID[alias] = card_id

# 카드 ID 리스트 및 임베딩
canonical_names = list(CARD_ID_TO_ALIASES.keys())
canonical_vectors = embedding_model.encode(
    canonical_names,
    return_dense=True,
    return_sparse=False
)['dense_vecs']
canonical_card_vectors = {
    name: vec for name, vec in zip(canonical_names, canonical_vectors)
}

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
    found_card_ids = set()
    sorted_aliases = sorted(ALIAS_TO_CARD_ID.keys(), key=len, reverse=True)
    temp_query = query

    for alias in sorted_aliases:
        # 별칭이 쿼리에 포함되어 있다면 올바른 카드 ID로 매핑
        if alias in temp_query:
            card_id = ALIAS_TO_CARD_ID[alias]
            found_card_ids.add(card_id)
            temp_query = temp_query.replace(alias, "")

    if found_card_ids:
        card_ids_list = list(found_card_ids)
        print(f"[Debug] (Step 1: Keyword) {len(card_ids_list)}개 카드명 감지. 'card_id' IN {card_ids_list} 필터 적용.")
        return Filter(must=[FieldCondition(key="card_id", match=MatchAny(any=card_ids_list))])

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

def hybrid_search(query: str) -> List[Tuple[str, float, Any]]:
    """
    하이브리드 검색 함수 (Dense + Sparse 후보군 합산, 리랭킹 제거)
    카드 필터링 후 해당 카드의 모든 문서를 검색합니다.
    Args:
        - query: 사용자 쿼리 문자열
    Returns:
        - List of (doc_id, score, point) 튜플 리스트 (모든 후보군 반환)
    """
    global qdrant_connection, COL, embedding_model
    
    # 1. 쿼리 인코딩
    q_dense, q_sparse = encode_query(embedding_model, query)
    print(f"\n[Debug] 쿼리 임베딩 완료. (Dense vector length: {len(q_dense)}, Sparse vector non-zero count: {len(q_sparse.indices)})")

    # 2. 스마트 필터 실행
    query_filter = smart_filter_router(query, q_dense)
    print(f"[Debug] 스마트 필터 생성 완료: {query_filter}")

    # 3. Dense + Sparse 검색 (필터 적용, limit 크게 설정하여 모든 문서 가져오기)
    dres = qdrant_connection.search(
        collection_name=COL,
        query_vector=q_dense,
        query_filter=query_filter,
        limit=100  # 충분히 큰 값으로 설정
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
        limit=100  # 충분히 큰 값으로 설정
    )
    print(f"[Debug] 검색 완료. (Dense hits: {len(dres)}, Sparse hits: {len(sres)})")

    # 4. 후보군 합산 (ID 기준 중복 제거)
    seen_ids = set()
    results = []
    
    # Dense 결과 추가
    for r in dres:
        if r.id not in seen_ids:
            results.append((r.id, r.score, r))
            seen_ids.add(r.id)
    
    # Sparse 결과 추가
    for r in sres:
        if r.id not in seen_ids:
            results.append((r.id, r.score, r))
            seen_ids.add(r.id)
    
    print(f"\n{'='*80}")
    print(f"[후보군 합산 결과] 총 {len(results)}개 문서 (리랭킹 없이 모두 반환)")
    print(f"{'='*80}")
    
    for idx, (pid, score, pt) in enumerate(results, 1):
        if pt:
            pl = pt.payload
            card_id = pl.get('card_id', 'N/A')
            doc_id = pl.get('doc_id', 'N/A')
            section = pl.get('section_canonical', 'N/A')
            granularity = pl.get('granularity', 'N/A')
            tag_major = pl.get('tag_major', 'N/A')
            tag_middle = pl.get('tag_middle', 'N/A')
            preview = pl.get('preview', '')[:100]  # 미리보기 100자
            
            print(f"\n[{idx}] score={score:.4f} | UUID: {pid}")
            print(f"    카드: {card_id}")
            print(f"    doc_id: {doc_id}")
            print(f"    구분: {section} ({granularity})")
            print(f"    경로: {tag_major} > {tag_middle}")
            print(f"    내용: {preview}...")
    
    print(f"\n{'='*80}\n")
    
    return results

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

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # 1. 하이브리드 검색 실행 (모든 문서 반환)
        results = self.search_func(query)

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


def get_card_description(query: str) -> dict:
    """
    특정 카드의 혜택, 연회비, 설명 등을 제공하는 함수
    
    사용자 쿼리에서 카드명을 추출하고, 해당 카드의 모든 문서를 검색하여
    카드 설명을 생성합니다. 기존 hybrid_search와 CustomHybridRetriever를 활용합니다.
    
    Args:
        query: 사용자 쿼리 (예: "우리카드 7CORE 연회비 얼마야?")
        
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
            'card_ids': []
        }
    
    # 3. 인식된 카드 ID 추출 (리스트로 변경)
    recognized_card_ids = []
    if query_filter.must:
        condition = query_filter.must[0]
        if hasattr(condition.match, 'value'):
            recognized_card_ids = [condition.match.value]
        elif hasattr(condition.match, 'any'):
            recognized_card_ids = condition.match.any if condition.match.any else []
    
    print(f"[카드 설명] 인식된 카드: {recognized_card_ids}")
    
    # 4. 기존 hybrid_search 함수 사용 (모든 문서 검색)
    results = hybrid_search(query)
    
    if not results:
        card_names_str = "', '".join(recognized_card_ids)
        return {
            'answer': f"'{card_names_str}' 카드의 정보를 찾을 수 없습니다.",
            'card_ids': recognized_card_ids
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
    
    # 6. LLM 인스턴스 생성 (4B 모델)
    generation_llm = ChatOllama(
        model=settings.OLLAMA_MODEL_NAME,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0
    )
    
    # 7. Qwen 4B 모델 프롬프트 (사용자 경험 최적화)
    template = """당신은 친절한 우리카드 상담 AI입니다. 사용자 질문에 대화하듯 자연스럽고 읽기 편하게 답변하세요.

# 핵심 원칙
1. **사용자 질문 의도**에 맞춰 답변 길이와 형식을 조절
2. **핵심 정보 우선** - 가장 중요한 내용을 먼저, 부가 정보는 나중에
3. **자연스러운 대화체** - "~입니다" 대신 "~해요", "~예요" 등 부드러운 어투
4. **시각적 구분** - 이모지는 필수 아닌 선택적 사용, 과도하지 않게
5. **모바일 친화적** - 테이블 대신 리스트, 짧은 문단, 적절한 줄바꿈

# 질문 유형별 답변 전략

## 1. 간단한 사실 확인 ("연회비?", "할인율?", "가족카드?")
→ **즉답 1-2문장** + 필요시 간단한 추가 설명
예: "연회비는 50,000원이에요. 전월실적 조건은 없습니다."

## 2. 카드 전체 소개 ("알려줘", "어떤 카드?", "설명해줘")
→ **3단계 구조**: 한 줄 요약 → 핵심 혜택 → 비용/조건
예시:
**7CORE 카드**는 생활밀착형 7대 영역에서 10% 할인받는 카드예요.

**어디서 할인되나요?**
• 온라인쇼핑 - 쿠팡, SSG, 컬리 등
• 배달앱 - 배민, 쿠팡이츠, 요기요
• 커피 - 스벅, 투썸, 이디야
• 대형마트 - 이마트, 롯데마트
• 교육 - 학원, 서점
• 병원 - 종합병원, 치과, 동물병원
• 주유 - SK, GS, 현대, S-OIL

**월 최대 할인**
소비액에 따라 달라져요:
- 50~100만원: 28,000원
- 100~200만원: 42,000원
- 200만원 이상: 84,000원

**비용**
연회비 50,000원이고, 실적 조건은 없어요.

## 3. 혜택 집중 질문 ("혜택?", "뭐가 좋아?", "할인?")
→ **혜택 중심** + 실사용 팁
예시:
주요 혜택은 **7대 생활영역 10% 할인**이에요.

가장 많이 쓰는 곳:
• 쿠팡, 배민 같은 온라인/배달 (거의 매일 쓰죠)
• 스타벅스 커피 (하루 한잔이면 월 5천원 이상 절약)
• 이마트 장보기 (주말 장보기 할인)

월 최대 84,000원까지 할인되니까, 월 200만원 이상 쓰시면 가장 이득이에요.

## 4. 특정 카테고리 확인 ("스타벅스 돼?", "쿠팡 할인?")
→ **Yes/No 먼저** + 구체적 조건
예: "네, 스타벅스 할인돼요! 커피전문점 10% 할인이 적용되고, 월 소비액에 따라 최대 12,000원까지 받을 수 있어요."

## 5. 비용 관련 ("연회비?", "실적?", "비싸?")
→ **금액 직접 제시** + 면제 조건 (있다면)
예시:
연회비는 **50,000원**이에요.

• 실적 조건 없음 (부담 없이 사용 가능)
• 가족카드는 발급되지 않아요

## 6. 한도/조건 ("얼마까지?", "조건?", "한도?")
→ **구간별 명확하게** + 실제 예시
예시:
월 할인한도는 **소비액에 따라 차등** 적용돼요:

• 50~100만원 쓰시면 → 최대 28,000원
• 100~200만원 쓰시면 → 최대 42,000원
• 200만원 이상 쓰시면 → 최대 84,000원

예를 들어 월 250만원 쓰시면, 7개 영역에서 각 12,000원씩 총 84,000원 할인받을 수 있어요.

## 7. 유의사항 ("조심할 거?", "제외?", "안되는 거?")
→ **부정적 톤 피하기** + "참고하세요" 톤
예시:
몇 가지 참고하실 점:

• 현금서비스/카드대출은 실적에 안 들어가요
• 일부 백화점 내 입점 매장은 제외될 수 있어요
• 연회비 환불은 최대 3개월 소요돼요

큰 제약은 없고, 일반적인 생활 소비는 대부분 할인 적용돼요!

## 8. 비교 질문 ("A vs B", "뭐가 나아?")
→ **테이블 대신 리스트** + 한 줄 결론
예시:
두 카드 비교해드릴게요.

**7CORE 카드**
• 연회비: 50,000원
• 혜택: 7대 영역 10% 할인
• 한도: 최대 84,000원
• 특징: 실적 조건 없음

**EVERY POINT 카드**
• 연회비: 20,000원
• 혜택: 전가맹점 0.7% 적립
• 한도: 최대 30,000원
• 특징: 전월 30만원 이상 필요

▶︎ **온라인/배달 많이 쓰시면** 7CORE가 유리하고, **다양한 곳에서 조금씩** 쓰시면 EVERY POINT가 나아요.

## 9. 추천/적합성 ("나한테 맞아?", "추천?", "누구한테 좋아?")
→ **사용 패턴 기반** + 공감 표현
예시:
이런 분들께 잘 맞아요:

✓ 쿠팡, 배민 자주 시키시는 분
✓ 스타벅스 출근길 커피 루틴 있으신 분
✓ 월 200만원 이상 카드 쓰시는 분
✓ 학원비, 병원비 고정 지출 있으신 분

특히 **생활비 대부분을 카드로** 쓰시면 월 8만원 넘게 할인받을 수 있어서 연회비 충분히 뽑아요.

## 10. Yes/No 질문 ("돼?", "가능?", "지원?")
→ **명확한 답 + 조건 1줄**
예: "네, 가능해요! 온라인쇼핑 10% 할인이 적용되고, 월 소비액에 따라 최대 12,000원까지 할인받을 수 있어요."

---

# 카드 정보
{context}

# 질문
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
        'card_ids': recognized_card_ids
    }
