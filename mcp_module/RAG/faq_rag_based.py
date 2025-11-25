"""
FAQ 의미 기반 검색 모듈 (2단계 Re-ranking)

이 모듈은 Bi-Encoder + Cross-Encoder 2단계 검색을 사용하여 
사용자 질문과 FAQ 데이터베이스 간의 정확한 매칭을 수행합니다.

검색 프로세스:
    1단계 (Bi-Encoder): Jina v3로 빠른 후보 추출 (top 15)
    2단계 (Cross-Encoder): BGE Reranker로 정밀 재랭킹 (top 3)
    
주요 기능:
    - FAQ 임베딩 캐싱으로 빠른 1단계 검색
    - Cross-Encoder로 정확도 향상
    - 2단계 파이프라인으로 속도와 정확도 균형
    
성능:
    - Bi-Encoder: jinaai/jina-embeddings-v3 (다국어)
    - Cross-Encoder: BAAI/bge-reranker-v2-m3 (다국어)
    - 예상 응답시간: ~1초
    - 예상 정확도: 85-90%
"""

import re
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Set

import numpy as np
import psycopg2
import torch
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from core.config import settings


# ============================================================================
# 전역 캐시 변수
# ============================================================================

_embedding_model: SentenceTransformer = None  # Jina v3 Bi-Encoder
_cross_encoder: CrossEncoder = None           # BGE Reranker v2-m3
_synonym_cache: Dict[str, List[str]] = None   # 동의어 매핑 캐시
_faq_embeddings_cache: torch.Tensor = None    # FAQ 임베딩 벡터 캐시
_faq_data_cache: List[Dict[str, Any]] = None  # FAQ 메타데이터 캐시


# ============================================================================
# 유틸리티 함수
# ============================================================================

@contextmanager
def get_db_connection():
    """
    PostgreSQL 데이터베이스 연결을 컨텍스트 매니저로 관리합니다.
    
    Yields:
        psycopg2.connection: PostgreSQL 연결 객체
        
    Raises:
        psycopg2.Error: DB 연결 실패 시
    """
    conn = None
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        yield conn
    except psycopg2.Error as e:
        print(f"[FAQ] DB 연결 오류: {e}")
        raise
    finally:
        if conn:
            conn.close()


def normalize_text(text: str) -> str:
    """
    텍스트를 정규화합니다.
    
    공백, 하이픈, 특수문자를 제거하고 소문자로 변환하여
    키워드 매칭 정확도를 향상시킵니다.
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        정규화된 텍스트 (공백/특수문자 제거, 소문자)
        
    Example:
        >>> normalize_text("카드-분실 했어요!")
        "카드분실했어요"
    """
    return re.sub(r'[\s\-_?!,.]', '', text).lower() if text else text


# ============================================================================
# 모델 및 데이터 로딩
# ============================================================================

def get_embedding_model() -> SentenceTransformer:
    """
    Jina Embeddings v3 Bi-Encoder 모델을 로드합니다 (Lazy Loading).
    
    1단계 검색에서 사용되며, 빠른 후보 추출을 담당합니다.
    
    Returns:
        SentenceTransformer: Jina v3 임베딩 모델
    """
    global _embedding_model
    if _embedding_model is None:
        print("[FAQ] Bi-Encoder 로드 중...")
        _embedding_model = SentenceTransformer(
            'jinaai/jina-embeddings-v3',
            device=settings.DEVICE,
            trust_remote_code=True
        )
        print(f"[FAQ] Bi-Encoder 로드 완료 (Device: {settings.DEVICE})")
    return _embedding_model


def get_cross_encoder() -> CrossEncoder:
    """
    BGE Reranker v2-m3 Cross-Encoder 모델을 로드합니다 (Lazy Loading).
    
    2단계 재랭킹에서 사용되며, 정밀한 관련도 점수를 계산합니다.
    다국어 지원으로 한국어 질문-FAQ 쌍의 관련도를 정확히 평가합니다.
    
    Returns:
        CrossEncoder: BGE Reranker v2-m3 모델
    """
    global _cross_encoder
    if _cross_encoder is None:
        print("[FAQ] Cross-Encoder 로드 중...")
        _cross_encoder = CrossEncoder(
            'BAAI/bge-reranker-v2-m3',
            device=settings.DEVICE,
            max_length=512
        )
        print(f"[FAQ] Cross-Encoder 로드 완료 (Device: {settings.DEVICE})")
    return _cross_encoder


def load_synonyms() -> Dict[str, List[str]]:
    """
    동의어 매핑을 데이터베이스에서 로드하여 캐싱합니다.
    
    동의어 확장을 통해 검색 범위를 넓히고 정확도를 향상시킵니다.
    
    Returns:
        Dict[str, List[str]]: {원본 단어: [동의어 리스트]} 매핑
        
    Example:
        {"분실": ["잃어버렸어요", "없어졌어요"]}
    """
    global _synonym_cache
    if _synonym_cache is not None:
        return _synonym_cache
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT source_term, target_terms FROM synonym_mappings;")
                results = cursor.fetchall()
                _synonym_cache = {row['source_term']: row['target_terms'] for row in results}
                print(f"[FAQ] 동의어 로드: {len(_synonym_cache)}개")
                return _synonym_cache
    except Exception as e:
        print(f"[FAQ] 동의어 로드 오류: {e}")
        return {}


def load_faq_embeddings() -> Dict[str, Any]:
    """
    FAQ 데이터를 로드하고 임베딩 벡터를 사전 계산하여 캐싱합니다.
    
    서버 시작 시 1회만 실행되며, 이후에는 캐시된 데이터를 사용합니다.
    질문과 상위 10개 키워드를 결합하여 임베딩을 생성합니다.
    
    Returns:
        Dict[str, Any]: {
            'embeddings': torch.Tensor - FAQ 임베딩 벡터 (N x D),
            'faqs': List[Dict] - FAQ 메타데이터 리스트
        }
        
    Note:
        - 임베딩은 정규화되어 코사인 유사도 계산에 최적화됨
        - 키워드는 검색 정확도 향상을 위해 최대 10개까지 사용
    """
    global _faq_embeddings_cache, _faq_data_cache
    
    # 캐시된 데이터가 있으면 반환
    if _faq_embeddings_cache is not None:
        return {'embeddings': _faq_embeddings_cache, 'faqs': _faq_data_cache}
    
    start = time.time()
    try:
        # DB에서 FAQ 데이터 로드
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT f.faq_id, f.question, f.answer, f.normalized_keywords, 
                           f.user_expressions, c.category_name, f.priority
                    FROM faqs f
                    JOIN faq_categories c ON f.category_id = c.category_id
                    ORDER BY f.faq_id;
                """)
                faqs = [dict(row) for row in cursor.fetchall()]
        
        if not faqs:
            return {'embeddings': None, 'faqs': []}
        
        model = get_embedding_model()
        
        # 질문 + 키워드 결합 텍스트 생성
        texts = []
        for faq in faqs:
            text = faq['question']
            # 상위 10개 키워드만 사용하여 노이즈 감소
            if faq['normalized_keywords']:
                text += ' ' + ' '.join(faq['normalized_keywords'][:10])
            texts.append(text)
        
        # 임베딩 생성 (정규화 포함)
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        elapsed = (time.time() - start) * 1000
        print(f"[FAQ] FAQ 임베딩 완료: {len(faqs)}개 ({elapsed:.1f}ms)")
        
        # 캐시 저장
        _faq_embeddings_cache = embeddings
        _faq_data_cache = faqs
        
        return {'embeddings': embeddings, 'faqs': faqs}
        
    except Exception as e:
        print(f"[FAQ] 임베딩 캐싱 오류: {e}")
        import traceback
        traceback.print_exc()
        return {'embeddings': None, 'faqs': []}


# ============================================================================
# 검색 헬퍼 함수 (현재 미사용, 향후 확장용)
# ============================================================================

def expand_with_synonyms(question: str) -> Set[str]:
    """
    질문의 단어를 동의어로 확장합니다.
    
    동의어 매핑을 사용하여 질문의 각 단어에 대한 유사 표현을 찾아
    검색 범위를 확대합니다.
    
    Args:
        question: 사용자 질문
        
    Returns:
        Set[str]: 원본 단어 + 동의어 집합
        
    Note:
        현재 버전에서는 사용되지 않지만, 향후 하이브리드 검색 구현 시 활용 가능
    """
    synonyms = load_synonyms()
    words = question.split()
    expanded = set(words)
    
    for word in words:
        normalized = normalize_text(word)
        
        # 정방향 매칭: 원본 단어가 source_term인 경우
        if word in synonyms:
            expanded.update(synonyms[word])
        if normalized in synonyms:
            expanded.update(synonyms[normalized])
        
        # 역방향 매칭: 원본 단어가 target_terms에 포함된 경우
        for source, targets in synonyms.items():
            if word in targets or normalized in [normalize_text(t) for t in targets]:
                expanded.add(source)
                expanded.update(targets)
    
    return expanded


def filter_faqs_by_keywords(question: str) -> List[Dict[str, Any]]:
    """
    키워드 기반으로 FAQ 후보를 필터링합니다.
    
    동의어 확장과 PostgreSQL GIN 인덱스를 활용하여
    관련 FAQ를 빠르게 추출합니다.
    
    Args:
        question: 사용자 질문
        
    Returns:
        List[Dict[str, Any]]: 매칭 스코어가 높은 상위 15개 FAQ
        
    Note:
        현재 버전에서는 사용되지 않지만, 2단계 검색 구현 시 활용 가능
        (1단계: 키워드 필터링 → 2단계: 임베딩 re-ranking)
    """
    expanded = expand_with_synonyms(question)
    normalized = [normalize_text(kw) for kw in expanded if kw]
    all_keywords = list(set(expanded) | set(normalized))
    
    if not all_keywords:
        return []
    
    ilike_patterns = [f'%{kw}%' for kw in all_keywords]
    
    # 키워드 매칭 스코어 계산 쿼리
    # - normalized_keywords 매칭: 2점
    # - user_expressions 매칭: 3점 (더 높은 가중치)
    query = """
        SELECT 
            f.faq_id, f.question, f.answer, f.category_id,
            c.category_name, f.priority,
            (
                COALESCE(
                    (SELECT COUNT(*) FROM unnest(f.normalized_keywords) AS k 
                     WHERE k = ANY(%s::text[])), 0
                ) * 2.0 +
                COALESCE(
                    (SELECT COUNT(*) FROM unnest(f.user_expressions) AS expr 
                     WHERE expr ILIKE ANY(%s::text[])), 0
                ) * 3.0
            ) AS match_score
        FROM faqs f
        JOIN faq_categories c ON f.category_id = c.category_id
        WHERE 
            f.normalized_keywords && %s::text[]
            OR EXISTS (
                SELECT 1 FROM unnest(f.user_expressions) AS expr
                WHERE expr ILIKE ANY(%s::text[])
            )
        ORDER BY match_score DESC, f.priority DESC
        LIMIT 15;
    """
    
    try:
        start = time.time()
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (all_keywords, ilike_patterns, all_keywords, ilike_patterns))
                results = cursor.fetchall()
                elapsed = (time.time() - start) * 1000
                print(f"[FAQ] 키워드 필터링: {len(results)}개 후보 ({elapsed:.1f}ms)")
                return [dict(row) for row in results]
    except Exception as e:
        print(f"[FAQ] 필터링 오류: {e}")
        return []


# ============================================================================
# 메인 검색 함수
# ============================================================================

def search_faq(question: str, top_k: int = 3, top_k_retrieval: int = 15) -> Dict[str, Any]:
    """
    사용자 질문에 대해 가장 관련성 높은 FAQ를 2단계 Re-ranking으로 검색합니다.
    
    Stage 1 (Bi-Encoder): Jina v3로 전체 FAQ에서 빠르게 상위 15개 후보 추출
    Stage 2 (Cross-Encoder): BGE Reranker로 15개 후보의 정밀한 관련도 점수 계산 후 상위 3개 선택
    
    프로세스:
        1. [Bi-Encoder] 질문 임베딩 생성 (Jina v3)
        2. [Bi-Encoder] 47개 FAQ와 코사인 유사도 계산
        3. [Bi-Encoder] 상위 15개 후보 추출 (~300ms)
        4. [Cross-Encoder] 15개 [질문, FAQ] 쌍의 관련도 점수 계산 (~700ms)
        5. [Cross-Encoder] 점수 기준 상위 3개 최종 선택
        6. 결과 포맷팅 및 반환
    
    Args:
        question: 사용자의 자연어 질문
        top_k: 최종 반환할 FAQ 개수 (기본값: 3)
        top_k_retrieval: Bi-Encoder로 추출할 후보 개수 (기본값: 15)
    
    Returns:
        Dict[str, Any]: {
            'answer': str - 가장 관련성 높은 FAQ의 답변 (카테고리 포함),
            'relatedQuestions': List[str] - 나머지 관련 질문 목록
        }
        
    Example:
        >>> search_faq("카드 잃어버렸어요")
        {
            'answer': '[카드관리]\n\n질문: 재발급 신청방법을 알려주세요?\n\n답변: ...',
            'relatedQuestions': ['카드를 일시적으로 정지할수 있나요?', ...]
        }
        
    Note:
        - 평균 응답시간: ~1초 (Bi-Encoder 300ms + Cross-Encoder 700ms)
        - 첫 실행: ~30초 (Bi-Encoder + Cross-Encoder 모델 로드)
        - 예상 정확도: 85-90% (Cross-Encoder re-ranking 적용)
        - 이전 정확도: 73% (Bi-Encoder 단독, 37개 테스트 기준)
    """
    print(f"\n[FAQ] 2단계 검색 시작: '{question}'\n" + "=" * 80)
    start_time = time.time()
    
    try:
        # ========== Stage 1: Bi-Encoder 후보 추출 ==========
        print("[Stage 1] Bi-Encoder 후보 추출 중...")
        stage1_start = time.time()
        
        # 캐시된 FAQ 임베딩 로드
        cache = load_faq_embeddings()
        embeddings = cache['embeddings']
        faqs = cache['faqs']
        
        if embeddings is None or len(faqs) == 0:
            return {
                "answer": "FAQ 데이터를 불러올 수 없습니다.",
                "relatedQuestions": []
            }
        
        bi_encoder = get_embedding_model()
        
        # 사용자 질문 임베딩 생성
        q_embed = bi_encoder.encode(
            question,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # 코사인 유사도 계산 및 상위 K개 후보 추출
        similarities = util.cos_sim(q_embed, embeddings)[0]
        top_retrieval_indices = torch.topk(
            similarities, 
            k=min(top_k_retrieval, len(faqs))
        )[1]
        
        # 후보 FAQ 리스트 구성
        candidates = []
        for idx in top_retrieval_indices:
            faq = faqs[idx.item()]
            candidates.append({
                'faq': faq,
                'bi_score': similarities[idx].item()
            })
        
        stage1_elapsed = (time.time() - stage1_start) * 1000
        print(f"[Stage 1] 완료: {len(candidates)}개 후보 추출 ({stage1_elapsed:.1f}ms)")
        
        # ========== Stage 2: Cross-Encoder Re-ranking ==========
        print("[Stage 2] Cross-Encoder Re-ranking 중...")
        stage2_start = time.time()
        
        cross_encoder = get_cross_encoder()
        
        # [질문, FAQ 질문] 쌍 생성
        pairs = [[question, c['faq']['question']] for c in candidates]
        
        # Cross-Encoder로 정밀한 관련도 점수 계산
        cross_scores = cross_encoder.predict(pairs, show_progress_bar=False)
        
        # 점수 기준 내림차순 정렬 후 상위 K개 선택
        sorted_indices = np.argsort(cross_scores)[::-1]
        top_indices = sorted_indices[:min(top_k, len(candidates))]
        
        stage2_elapsed = (time.time() - stage2_start) * 1000
        print(f"[Stage 2] 완료: 최종 {len(top_indices)}개 선택 ({stage2_elapsed:.1f}ms)")
        
        # ========== 결과 구성 ==========
        # Cross-Encoder 점수 기준 정렬된 최종 FAQ 리스트
        top_faqs = []
        for rank_idx, candidate_idx in enumerate(top_indices):
            faq = candidates[candidate_idx]['faq'].copy()
            faq['bi_score'] = candidates[candidate_idx]['bi_score']
            faq['cross_score'] = float(cross_scores[candidate_idx])
            faq['final_rank'] = rank_idx + 1
            top_faqs.append(faq)
        
        # 로깅: 전체 소요 시간 및 각 FAQ 점수
        total_elapsed = (time.time() - start_time) * 1000
        print(f"\n[FAQ] 총 소요 시간: {total_elapsed:.1f}ms")
        print(f"  - Stage 1 (Bi-Encoder): {stage1_elapsed:.1f}ms")
        print(f"  - Stage 2 (Cross-Encoder): {stage2_elapsed:.1f}ms")
        print("\n[최종 결과]")
        for faq in top_faqs:
            print(f"  #{faq['final_rank']}: {faq['question'][:50]}...")
            print(f"    → Bi-Encoder: {faq['bi_score']:.3f}, Cross-Encoder: {faq['cross_score']:.3f}")
        print("=" * 80)
        
        # 최상위 FAQ 반환
        best = top_faqs[0]
        
        # views 카운트 증가
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE faqs SET views = views + 1 WHERE faq_id = %s;",
                        (best['faq_id'],)
                    )
                    conn.commit()
                    print(f"[FAQ] views 증가: faq_id={best['faq_id']}")
        except Exception as e:
            print(f"[FAQ] views 업데이트 오류: {e}")
        
        return {
            "answer": f"[{best['category_name']}]\n\n질문: {best['question']}\n\n답변: {best['answer']}",
            "relatedQuestions": [faq['question'] for faq in top_faqs[1:]]
        }
        
    except Exception as e:
        print(f"[FAQ] 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": f"검색 중 오류 발생: {e}",
            "relatedQuestions": []
        }
