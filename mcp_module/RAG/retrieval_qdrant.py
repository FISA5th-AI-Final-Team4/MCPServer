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
        # 비교 쿼리: 각 카드 최소 10개 보장, 동일 섹션만 사용
        section_priority = ['BENEFITS', 'FEES', 'USAGE_GUIDE', 'RESTRICTIONS', 'CONDITIONS', 'OTHER']
        
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
        
        # 공통 섹션 찾기
        all_card_ids = list(card_section_docs.keys())
        if len(all_card_ids) >= 2:
            common_sections = set(card_section_docs[all_card_ids[0]].keys())
            for cid in all_card_ids[1:]:
                common_sections &= set(card_section_docs[cid].keys())
            common_sections = [s for s in section_priority if s in common_sections]
        else:
            common_sections = section_priority
        
        # 사용할 섹션 목록 (처음엔 공통 섹션만)
        sections_to_use = list(common_sections)
        
        # 각 카드 최소 10개 보장 - 부족하면 섹션 추가 (양쪽 모두에)
        min_per_card = 10
        
        # 모든 카드가 가진 문서 수 계산 (현재 섹션 기준)
        def count_docs_for_sections(card_id, sections):
            count = 0
            for section in sections:
                if section in card_section_docs.get(card_id, {}):
                    count += len(card_section_docs[card_id][section])
            return count
        
        # 가장 문서가 적은 카드가 10개 이상 될 때까지 섹션 추가
        while True:
            min_docs = min(count_docs_for_sections(cid, sections_to_use) for cid in all_card_ids)
            if min_docs >= min_per_card:
                break
            
            # 추가할 섹션 찾기 (양쪽 카드 모두 가진 섹션 중 아직 안 쓴 것)
            added = False
            for section in section_priority:
                if section not in sections_to_use:
                    # 모든 카드가 이 섹션을 가지고 있는지 확인
                    all_have = all(section in card_section_docs.get(cid, {}) for cid in all_card_ids)
                    if all_have:
                        sections_to_use.append(section)
                        added = True
                        break
            
            if not added:
                # 더 이상 공통으로 추가할 섹션이 없음
                break
        
        print(f"[사용 섹션] {sections_to_use}")
        
        results = []
        card_doc_counts = {}
        card_section_detail = {}
        
        for card_id in all_card_ids:
            card_docs = []
            card_section_detail[card_id] = {}
            used_pids = set()
            
            # 결정된 섹션에서만 문서 가져오기
            for section in sections_to_use:
                if section in card_section_docs[card_id]:
                    section_docs = sorted(card_section_docs[card_id][section], 
                                        key=lambda x: x[1], reverse=True)
                    for doc in section_docs:
                        if doc[0] not in used_pids:
                            card_docs.append(doc)
                            used_pids.add(doc[0])
                            card_section_detail[card_id][section] = card_section_detail[card_id].get(section, 0) + 1
            
            card_doc_counts[card_id] = len(card_docs)
            results.extend(card_docs)
        
        print(f"\n{'='*80}")
        print(f"[비교 쿼리 감지] {len(card_ids)}개 카드 비교: {', '.join(card_ids)}")
        count_info = ", ".join([f"{k}: {v}개" for k, v in card_doc_counts.items()])
        print(f"[카드별 선택] 총 {len(all_results)}개 → {len(results)}개 문서 ({count_info})")
        for cid, sections in card_section_detail.items():
            section_str = ", ".join([f"{s}: {c}" for s, c in sections.items()])
            print(f"  └ {cid}: {{{section_str}}}")
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
    
    # 7. GPT-5-mini 프롬프트 
    
    template = """당신은 '우리카드' 상품 전문 AI 상담사입니다.
오직 주어진 'Context' 정보만을 바탕으로 사용자의 'Question'에 대해 답변해야 합니다.
Context에 없는 내용은 절대 지어내지 말고, **오직 Context에서 알 수 있는 내용으로만** 포함하여 답변하세요.

## 질문 유형별 응답 형식 가이드라인

### 1번 유형: [카드명]에 대한 단순 상세 정보 **하나**에 대한 질문 ( example : [카드명]의 연회비 알려줘, [카드명]의 실적조건이 어떻게 돼?)

답변 방식 : 1~2문장으로 바로 답변

<1번 유형 출력 예시> "연회비는 국내전용 **15,000원**, 해외겸용 **18,000원**입니다." </1번 유형 출력 예시>
<1번 유형 출력 예시> "전월 실적 **30만원 이상** 시 혜택이 적용됩니다." </1번 유형 출력 예시>

---

### 2번 유형: [카드명]에 대한 광범위한 상세 정보 설명을 요청하는 경우 ( example : [카드명]의 혜택에 대해 알려줘, [카드명]의 이용조건에 대해 설명해줘 )
답변 방식 : 광범위한 상세 정보에 대한 핵심 요약 + 영역별 조건 명시 + 추가 질문 유도

<2번 유형 출력 예시>
**[카드명]**
[한 줄 특징]

🎁 **주요 혜택**
• [혜택1]: [혜택1 관련 수치] ([대상 가맹점/조건 간단히])
• [혜택2]: [혜택2 관련 수치] ([대상 가맹점/조건 간단히])
• [혜택3]: [혜택3 관련 수치] ([대상 가맹점/조건 간단히])

⚠️ [핵심 사용 조건 1줄 제시 - 전월실적/한도 등 중요한 것만]

💡 "[카드명] 자세히 알려줘"라고 요청하시면 전체 혜택과 조건을 안내해 드릴게요!

</2번 유형 출력 예시>
---

### 3번 유형: [카드명]에 대한 **모든 상세 정보 설명을 요청**하는 경우 ( example : [카드명]에 대해 자세히 알려줘 )
답변 방식 : **'Context' 내 모든 정보를 포함**하여 체계적으로 정리 후 제시

<3번 유형 출력 예시>
**[카드명]**
[카드 특징 요약]

💳 **기본 정보**
• 연회비: [금액] (정보 있을 때만)
• 전월 실적: [조건] (정보 있을 때만)

🎁 **혜택 상세**
• **[혜택 카테고리1]**: [할인율/적립률]
  - 대상: [가맹점]
  - 한도: [월/건 한도]
• **[혜택 카테고리2]**: [할인율/적립률]
  - 대상: [가맹점]
  - 한도: [월/건 한도]


⚠️ **유의사항**
• [조건/제외 대상 등 기본 정보와 혜택 상세 외 문서 내 모든 정보들]

</3번 유형 출력 예시>
---

### 4번 유형: [카드명1]과 [카드명2]에 대한 **상품 비교를 요청**하는 경우 ( example : [카드명1]과 [카드명2]를 비교해줘, [카드명1]과 [카드명2]는 뭐가 다른거야? )
답변 방식 : 테이블 형식으로 비교

<4번 유형 출력 예시>
| 구분 | [카드A] | [카드B] |
|------|---------|---------|
| 연회비 | [금액] | [금액] |
| 전월실적 | [조건] | [조건] |
| 주요혜택 | [핵심 2~3개] | [핵심 2~3개] |
| 특화분야 | [ex: 해외/쇼핑] | [ex: 주유/통신] |


---

### 5번 유형: 질문한 [카드명]이 문서 내에 존재하지 않는 경우 

답변 방식 : "해당 카드는 현재 존재하지 않습니다. 찾으시는 상품명을 명확하게 입력해주세요(ex-카드의 정석2 혜택에 대해 자세히 설명해줘)"

---

## 핵심 원칙
1. 답변 생성 시 **오직 Context에서 알 수 있는 내용으로만** 포함하여 답변하세요.(추론하여 답변을 제시하지 말것)
2. 각 유형 별 출력 예시에 포함된 관용어 혹은 꾸밈말 외에는 **오직 상품 정보를 제공하는 답변만**으로 구성하세요.

---
# 카드 정보
{context}

# 질문
{question}

# 답변"""
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
