"""
FAQ 의미 기반 검색 모듈

[검색 프로세스]
1. FAQ 임베딩 캐싱 (서버 시작 시 1회, ~5초)
2. 동의어 확장 + 키워드 필터링 (24개 → 5~15개)
3. 의미 유사도 계산 및 정렬

[성능]
- 평균 검색 시간: ~112ms
- 캐시 메모리: ~37KB (24개 FAQ)
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, List, Set
from contextlib import contextmanager
import re
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings

# ============================================================================
# 전역 캐시
# ============================================================================
_embedding_model = None           # jhgan/ko-sbert-multitask (384차원)
_synonym_cache = None             # 동의어 매핑 (61개)
_faq_embeddings_cache = None      # FAQ 임베딩 벡터


def get_embedding_model() -> SentenceTransformer:
    """임베딩 모델 로드 (Lazy Loading)"""
    global _embedding_model
    if _embedding_model is None:
        print("[FAQ] 임베딩 모델 로드 중...")
        _embedding_model = SentenceTransformer('jhgan/ko-sbert-multitask', device=settings.DEVICE)
        print(f"[FAQ] 임베딩 모델 로드 완료 (Device: {settings.DEVICE})")
    return _embedding_model


@contextmanager
def get_db_connection():
    """PostgreSQL 연결 관리"""
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
    """텍스트 정규화 (공백/특수문자 제거, 소문자 변환)"""
    return re.sub(r'[\s\-_?!,.]', '', text).lower() if text else text


def load_synonyms() -> Dict[str, List[str]]:
    """동의어 매핑 로드 및 캐싱"""
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
    """FAQ 임베딩 사전 계산 및 캐싱"""
    global _faq_embeddings_cache
    if _faq_embeddings_cache is not None:
        return _faq_embeddings_cache
    
    start = time.time()
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT f.faq_id, f.question, f.answer, f.search_text, c.category_name, f.priority
                    FROM faqs f
                    JOIN faq_categories c ON f.category_id = c.category_id
                    ORDER BY f.faq_id;
                """)
                faqs = [dict(row) for row in cursor.fetchall()]
        
        if not faqs:
            return {'faqs': [], 'embeddings': None, 'faq_map': {}}
        
        model = get_embedding_model()
        embeddings = model.encode(
            [faq['search_text'] for faq in faqs],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        print(f"[FAQ] FAQ 임베딩 완료: {len(faqs)}개 ({(time.time() - start) * 1000:.1f}ms)")
        
        _faq_embeddings_cache = {
            'faqs': faqs,
            'embeddings': embeddings,
            'faq_map': {faq['faq_id']: idx for idx, faq in enumerate(faqs)}
        }
        return _faq_embeddings_cache
        
    except Exception as e:
        print(f"[FAQ] 임베딩 캐싱 오류: {e}")
        return {'faqs': [], 'embeddings': None, 'faq_map': {}}


def expand_with_synonyms(question: str) -> Set[str]:
    """질문을 동의어로 확장하여 검색 범위 증가"""
    synonyms = load_synonyms()
    words = question.split()
    expanded = set(words)
    
    for word in words:
        normalized = normalize_text(word)
        
        # 정확 매칭
        if word in synonyms:
            expanded.update(synonyms[word])
        if normalized in synonyms:
            expanded.update(synonyms[normalized])
        
        # 역방향 매칭
        for source, targets in synonyms.items():
            if word in targets or normalized in [normalize_text(t) for t in targets]:
                expanded.add(source)
                expanded.update(targets)
    
    return expanded


def filter_faqs_by_keywords(question: str) -> List[Dict[str, Any]]:
    """동의어 확장 + GIN 인덱스로 FAQ 후보 필터링"""
    expanded = expand_with_synonyms(question)
    normalized = [normalize_text(kw) for kw in expanded if kw]
    all_keywords = list(set(expanded) | set(normalized))
    
    if not all_keywords:
        return []
    
    ilike_patterns = [f'%{kw}%' for kw in all_keywords]
    
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
                print(f"[FAQ] 키워드 필터링: {len(results)}개 후보 ({(time.time() - start) * 1000:.1f}ms)")
                return [dict(row) for row in results]
    except Exception as e:
        print(f"[FAQ] 필터링 오류: {e}")
        return []


def search_faq(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    의미 기반 FAQ 검색
    
    Args:
        question: 사용자 질문
        top_k: 반환할 FAQ 개수
    
    Returns:
        {"answer": 답변, "relatedQuestions": 관련 질문 리스트}
    """
    print(f"\n[FAQ] 검색: '{question}'\n" + "=" * 80)
    start_time = time.time()
    
    try:
        cache = load_faq_embeddings()
        all_faqs = cache['faqs']
        all_embeddings = cache['embeddings']
        
        if not all_faqs or all_embeddings is None:
            return {"answer": "FAQ 데이터를 불러올 수 없습니다.", "relatedQuestions": []}
        
        candidates = filter_faqs_by_keywords(question)
        
        if candidates:
            indices = [cache['faq_map'][faq['faq_id']] for faq in candidates if faq['faq_id'] in cache['faq_map']]
            candidates = [all_faqs[i] for i in indices] if indices else all_faqs
            candidate_embeds = all_embeddings[indices] if indices else all_embeddings
        else:
            candidates, candidate_embeds = all_faqs, all_embeddings
        
        embed_start = time.time()
        q_embed = get_embedding_model().encode(question, convert_to_numpy=True, normalize_embeddings=True)
        similarities = cosine_similarity(q_embed.reshape(1, -1), candidate_embeds)[0]
        
        for i, faq in enumerate(candidates):
            faq['final_score'] = float(similarities[i])
        
        print(f"[FAQ] 임베딩+유사도: {(time.time() - embed_start) * 1000:.1f}ms (후보: {len(candidates)}개)")
        
        top_faqs = sorted(candidates, key=lambda x: x['final_score'], reverse=True)[:top_k]
        
        print(f"[FAQ] 총 시간: {(time.time() - start_time) * 1000:.1f}ms")
        for i, faq in enumerate(top_faqs, 1):
            print(f"[FAQ] #{i}: {faq['question'][:45]}... ({faq['final_score']:.3f})")
        print("=" * 80)
        
        best = top_faqs[0]
        return {
            "answer": f"[{best['category_name']}]\n\n질문: {best['question']}\n\n답변: {best['answer']}",
            "relatedQuestions": [f['question'] for f in top_faqs[1:]]
        }
        
    except Exception as e:
        print(f"[FAQ] 오류: {e}")
        return {"answer": f"검색 중 오류 발생: {e}", "relatedQuestions": []}
