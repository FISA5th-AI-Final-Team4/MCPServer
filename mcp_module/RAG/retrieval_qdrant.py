from FlagEmbedding import BGEM3FlagModel, FlagReranker # 임베딩 모델 및 리랭커
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

# 리랭커 초기화 (BGE-M3와 동일 모델 사용)
try:
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    print("[Init] 리랭커 로드 완료: bge-reranker-v2-m3")
except Exception as e:
    print(f"[Warning] 리랭커 로드 실패: {e}. 리랭킹 없이 진행합니다.")
    reranker = None

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

def smart_filter_router(query: str, query_dense_vec: List, threshold=0.75) -> Filter | None:
    """
    스마트 필터 라우터 (2단계로 구성된 로직)
        1. 키워드 매칭(MatchAny)
        2. 실패 시 유사도 매칭(MatchValue)
    Args:
        - query: 사용자 쿼리 문자열
        - query_dense_vec: 쿼리의 밀집 벡터 리스트
        - threshold: 유사도 임계값 (기본값: 0.75)
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
    all_results = []
    
    # Dense 결과 추가
    for r in dres:
        if r.id not in seen_ids:
            all_results.append((r.id, r.score, r))
            seen_ids.add(r.id)
    
    # Sparse 결과 추가
    for r in sres:
        if r.id not in seen_ids:
            all_results.append((r.id, r.score, r))
            seen_ids.add(r.id)
    
    # 5. 카드 개수 확인
    card_ids = set()
    for pid, score, pt in all_results:
        if pt:
            card_ids.add(pt.payload.get('card_id', 'Unknown'))
    
    is_comparison = len(card_ids) >= 2  # 2개 이상 카드 = 비교 쿼리
    
    # 6. 상위 N개 선택 전략
    if is_comparison:
        # 비교 쿼리: 섹션별로 균형있게 선택 (각 카드에서 동일 섹션 포함 보장)
        # 섹션별 상위 문서 선택 후 카드별로 분배
        section_priority = ['FEES', 'BENEFITS', 'USAGE_GUIDE', 'RESTRICTIONS', 'CONDITIONS', 'OTHER']
        
        # 카드별, 섹션별로 문서 그룹화
        card_section_docs = {}
        for pid, score, pt in all_results:
            if pt:
                card_id = pt.payload.get('card_id', 'Unknown')
                section = pt.payload.get('section_canonical', 'OTHER')
                
                if card_id not in card_section_docs:
                    card_section_docs[card_id] = {}
                if section not in card_section_docs[card_id]:
                    card_section_docs[card_id][section] = []
                
                card_section_docs[card_id][section].append((pid, score, pt))
        
        # 각 카드에서 상위 8개씩 선택 (섹션 균형 고려)
        results = []
        for card_id in card_section_docs:
            card_docs = []
            # 섹션별로 최소 1개씩은 포함하도록
            for section in section_priority:
                if section in card_section_docs[card_id]:
                    section_docs = sorted(card_section_docs[card_id][section], 
                                        key=lambda x: x[1], reverse=True)[:2]
                    card_docs.extend(section_docs)
            # 상위 8개만 선택
            card_docs = sorted(card_docs, key=lambda x: x[1], reverse=True)[:8]
            results.extend(card_docs)
        
        print(f"\n{'='*80}")
        print(f"[비교 쿼리 감지] {len(card_ids)}개 카드 비교: {', '.join(card_ids)}")
        print(f"[카드별 균형 선택] 총 {len(all_results)}개 → {len(results)}개 문서 (카드당 8개씩)")
        print(f"{'='*80}")
    else:
        # 단일 카드 쿼리: 단순히 상위 15개 선택
        results = sorted(all_results, key=lambda x: x[1], reverse=True)[:15]
        
        print(f"\n{'='*80}")
        print(f"[단일 카드 쿼리] 총 {len(all_results)}개 → 상위 {len(results)}개 문서 선택")
        print(f"{'='*80}")
    
    # 7. 리랭킹 적용 (옵션)
    if reranker and results:
        print(f"\n[리랭킹 시작] {len(results)}개 문서 재평가...")
        
        # 리랭커 입력 형식: [(query, passage), ...]
        pairs = []
        for pid, score, pt in results:
            if pt:
                # preview 또는 full_text 사용
                text = pt.payload.get('preview', pt.payload.get('full_text', ''))
                pairs.append([query, text])
        
        # 리랭킹 실행
        try:
            rerank_scores = reranker.compute_score(pairs)
            
            # 기존 결과에 리랭크 점수 추가
            reranked_results = []
            for i, (pid, old_score, pt) in enumerate(results):
                new_score = rerank_scores[i] if isinstance(rerank_scores, list) else rerank_scores
                reranked_results.append((pid, new_score, pt))
            
            # 리랭크 점수로 재정렬
            results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
            
            print(f"[리랭킹 완료] 상위 5개 점수: {[f'{r[1]:.4f}' for r in results[:5]]}")
        except Exception as e:
            print(f"[리랭킹 실패] {e}. 원본 순서 유지.")
    
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
    query_filter = smart_filter_router(query, q_dense, threshold=0.75)
    
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
    # generation_llm = ChatOllama(
    #     model=settings.OLLAMA_MODEL_NAME,
    #     base_url=settings.OLLAMA_BASE_URL,
    #     temperature=0
    # )
    from langchain_openai import ChatOpenAI

    # 2. 인스턴스 생성
    generation_llm = ChatOpenAI(
    model="gpt-5-mini",  # 또는 "gpt-4o-mini", "gpt-3.5-turbo"
    api_key=settings.OPENAI_API_KEY,  # settings에 정의된 키 사용
    temperature=0
)
    
    # 7. GPT-5-mini 최적화 프롬프트 (다양한 질문 유형 대응)
    
    template = """당신은 우리카드 전문 상담 AI입니다.
제공된 카드 정보를 바탕으로 고객 질문에 정확하고 친절하게 답변하세요.

## 📋 응답 원칙

1. **정확성**: 카드 정보에 있는 내용만 사용. 없는 정보는 "해당 정보는 확인되지 않습니다"로 안내
2. **구체성**: 할인율, 적립률, 연회비 등 수치는 정확히 명시
3. **명확성**: 조건이 있는 혜택은 반드시 조건(전월 실적, 월 한도 등)도 함께 안내
4. **가독성**: 핵심 정보는 볼드(**) 처리, 불릿(•)으로 구분
5. **친절함**: 전문적이면서도 친근한 존댓말 사용

---

## 🎯 질문 유형별 응답 가이드

### 1️⃣ 단순 정보 질문
예: "연회비 얼마야?", "전월 실적 조건이 뭐야?", "출시일 언제야?"
→ **1~2문장으로 직접 답변**

### 2️⃣ 카드 소개/설명 요청
예: "카드의정석 TEN 알려줘", "이 카드 뭐야?", "설명해줘"
→ **핵심 요약 3~4줄 + 주요 혜택 나열**

형식:
**[카드명]**
• [카드 특징 한 줄 요약]
• 연회비: [금액]
🎁 **주요 혜택**
• [혜택1]: [할인율/적립률] ([대상 가맹점])
• [혜택2]: [할인율/적립률] ([대상 가맹점])

### 3️⃣ 상세 설명 요청
예: "자세히 알려줘", "상세하게 설명해줘", "혜택 전부 알려줘"
→ **섹션별 체계적 정리**

형식:
**[카드명]**
• [카드 특징 요약]

🎁 **혜택**
• [혜택명]: [할인율/적립률]
  - 대상: [가맹점 나열]
  - 한도: [월/일 최대 금액]
  - 조건: [전월실적 등]

💳 **비용**
• 연회비: [금액] (국내/해외겸용)
• 전월 실적: [조건] (없으면 "없음")

⚠️ **유의사항**
• [유의사항 나열]

### 4️⃣ 특정 혜택 문의
예: "편의점 할인 뭐야?", "스타벅스 적립?", "주유 혜택 있어?"
→ **해당 혜택만 집중 설명**

형식:
🎁 **[카테고리] 혜택**
• [할인율/적립률]: [구체적 수치]
• 대상: [가맹점 나열]
• 한도: [월/일 한도]
• 조건: [필요 시]

### 5️⃣ 카드 비교 문의
예: "A카드랑 B카드 비교해줘", "어떤 게 나아?", "차이점이 뭐야?"
→ **표 형식 비교 + 추천**

형식:
| 구분 | [카드A] | [카드B] |
|------|---------|---------|
| 연회비 | [금액] | [금액] |
| 핵심혜택 | [요약] | [요약] |
| 전월실적 | [조건] | [조건] |

**→ 추천**: [사용 패턴에 따른 추천]

### 6️⃣ 조건/자격 문의
예: "누가 발급받을 수 있어?", "조건이 뭐야?", "어떻게 신청해?"
→ **자격 조건, 신청 방법 안내**

### 7️⃣ 애매하거나 정보 부족
예: 카드 정보에 없는 내용, 불명확한 질문
→ "해당 정보는 제공된 카드 정보에서 확인되지 않습니다. 우리카드 고객센터(1588-9955) 또는 공식 홈페이지에서 확인해 주세요."

---

## ⚠️ 금지 사항
- 카드 정보에 없는 내용 추측/창작 금지
- 수치 반올림/변형 금지 (정확한 숫자 그대로 사용)
- 과도한 줄바꿈 금지 (가독성 있게 적절히)

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
        'card_list': recognized_card_ids
    }
