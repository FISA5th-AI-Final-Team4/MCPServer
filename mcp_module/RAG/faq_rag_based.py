"""
FAQ Optimized Hybrid Search 모듈

3단계 최적화 검색:
1. 동의어 확장 (synonym_mappings 활용 - 61개 매핑)
2. 키워드 필터링 (GIN 인덱스: normalized_keywords, user_expressions)
3. 의미 검색 (필터링된 후보군만 임베딩)

성능 개선:
- DB 인덱스 활용 (normalized_keywords, user_expressions GIN 인덱스)
- 동의어 자동 확장으로 검색 범위 확대
- 임베딩 대상 축소 (24개 → 5~15개 후보로 좁혀서 처리)

필드 사용:
- normalized_keywords: 핵심 키워드 (정규화됨)
- user_expressions: 사용자 표현 패턴 (높은 가중치)
- keywords: 사용 중단 (normalized_keywords로 통합)
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
# 전역 변수
# ============================================================================
_embedding_model = None
_synonym_cache = None  # 동의어 캐싱


def get_embedding_model() -> SentenceTransformer:
    """임베딩 모델 로드"""
    global _embedding_model
    if _embedding_model is None:
        print("[FAQ] 임베딩 모델 로드 중...")
        _embedding_model = SentenceTransformer(
            'jhgan/ko-sbert-multitask',
            device=settings.DEVICE
        )
        print(f"[FAQ] 임베딩 모델 로드 완료 (Device: {settings.DEVICE})")
    return _embedding_model


# ============================================================================
# 유틸리티 함수
# ============================================================================

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
    """텍스트 정규화"""
    if not text:
        return text
    return re.sub(r'[\s\-_?!,.]', '', text).lower()


def load_synonyms() -> Dict[str, List[str]]:
    """동의어 매핑 로드 및 캐싱 (앱 시작 시 1회)"""
    global _synonym_cache
    if _synonym_cache is not None:
        return _synonym_cache
    
    query = "SELECT source_term, target_terms FROM synonym_mappings;"
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                _synonym_cache = {row['source_term']: row['target_terms'] for row in results}
                print(f"[FAQ] 동의어 로드: {len(_synonym_cache)}개")
                return _synonym_cache
    except Exception as e:
        print(f"[FAQ] 동의어 로드 오류: {e}")
        return {}


def expand_with_synonyms(question: str) -> Set[str]:
    """
    동의어 확장 (핵심 최적화)
    
    예: "리밋 올리기" → ["리밋", "한도", "이용한도", "올리기"]
    """
    synonyms = load_synonyms()  # 동의어 사전 불러옴
    words = question.split()    # 질문을 단어별로 분리
    expanded = set(words)       # 중복 방지위해 집합(Set)으로 초기화
    
    for word in words:
        normalized_word = normalize_text(word)
        # source_term 매칭/ 단어가 사전에 Key(기준어)로 있는지 확인 # 있으면 그 짝(Value)들 다 넣음   
        if word in synonyms:                
            expanded.update(synonyms[word])     
        if normalized_word in synonyms:         
            expanded.update(synonyms[normalized_word])      
            
        # target_terms 역방향 매칭/ 단어가 사전에 Value(동의어 목록)속에 있는지 확인
        for source, targets in synonyms.items():
            
            # 내 단어(word)가 어떤 기준어(source)의 동의어 리스트(targets)에 포함이 되는가 ? 
            if word in targets or normalized_word in [normalize_text(t) for t in targets]:
                expanded.add(source)
                expanded.update(targets)
    
    return expanded


def filter_faqs_by_keywords(question: str) -> List[Dict[str, Any]]:
    """
    키워드 + 동의어 기반 필터링 (GIN 인덱스 활용)
    
    활용 인덱스:
    - idx_faq_normalized_keywords (GIN) - 정규화된 키워드
    - idx_faq_user_expressions (GIN) - 사용자 표현 패턴
    
    성능: 24개 전체 → 5~15개 후보군
    """
    # 1. 동의어 확장
    expanded_keywords = expand_with_synonyms(question)
    expanded_list = list(expanded_keywords)
    
    # 2. 정규화 (중복 제거)
    normalized = [normalize_text(kw) for kw in expanded_list if kw]
    all_keywords = list(set(expanded_list + normalized))
    
    if not all_keywords:
        return []
    
    # 3. ILIKE 패턴 생성 (user_expressions 부분 매칭용)
    ilike_patterns = [f'%{kw}%' for kw in all_keywords]
    
    # 4. 필터링 쿼리 (keywords 필드 제거, normalized_keywords만 사용)
    query = """
        SELECT 
            f.faq_id, f.question, f.answer, f.category_id,
            c.category_name, f.priority,
            -- 매칭 점수 계산 (가중치: normalized_keywords=2.0, user_expressions=3.0)
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
            f.normalized_keywords && %s::text[]  -- GIN 인덱스 활용
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
                cursor.execute(query, (
                    all_keywords, ilike_patterns,
                    all_keywords, ilike_patterns
                ))
                results = cursor.fetchall()
                elapsed = (time.time() - start) * 1000
                print(f"[FAQ] 키워드 필터링: {len(results)}개 후보 ({elapsed:.1f}ms)")
                print(f"[FAQ] 확장 키워드 샘플: {all_keywords[:5]}...")
                return [dict(row) for row in results]
    except Exception as e:
        print(f"[FAQ] 필터링 오류: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# 메인 검색 함수
# ============================================================================

def search_faq(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    3단계 최적화 검색
    
    1. 동의어 확장 (synonym_mappings: 61개 매핑)
    2. 키워드 필터링 (GIN 인덱스, 24개 → 5~15개)
    3. 의미 검색 (필터링된 후보군만 임베딩)
    
    """
    print(f"\n[FAQ] 검색: '{question}'")
    print("=" * 80)
    start_time = time.time()
    
    try:
        # 1단계: 동의어 확장 + 키워드 필터링 (GIN 인덱스)
        candidates = filter_faqs_by_keywords(question)
        
        if not candidates:
            print("[FAQ] 필터링 결과 없음 → 전체 검색 폴백")
            # 폴백: 우선순위 높은 FAQ
            query = """
                SELECT f.faq_id, f.question, f.answer, c.category_name, f.priority
                FROM faqs f
                JOIN faq_categories c ON f.category_id = c.category_id
                ORDER BY f.priority DESC, f.views DESC
                LIMIT 10;
            """
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query)
                    candidates = [dict(row) for row in cursor.fetchall()]
        
        # 2단계: 의미 검색 (필터링된 후보군만)
        embed_start = time.time()
        model = get_embedding_model()
        
        q_embed = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
        
        candidate_questions = [faq['question'] for faq in candidates]
        candidate_embeds = model.encode(
            candidate_questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        similarities = cosine_similarity(q_embed.reshape(1, -1), candidate_embeds)[0]
        
        # 3단계: 최종 점수 계산 (키워드 + 의미 하이브리드)
        for i, faq in enumerate(candidates):
            semantic_score = float(similarities[i])
            keyword_score = float(faq.get('match_score', 0))  # Decimal → float
            
            faq['semantic_score'] = semantic_score
            faq['keyword_score'] = keyword_score
            faq['final_score'] = (
                keyword_score * 0.2 +   # 키워드 매칭 20%
                semantic_score * 0.8    # 의미 유사도 80%
            )
        
        embed_time = (time.time() - embed_start) * 1000
        print(f"[FAQ] 임베딩: {embed_time:.1f}ms ({len(candidates)}개)")
        
        # 정렬 및 선택
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        top_faqs = candidates[:top_k]
        
        total_time = (time.time() - start_time) * 1000
        print(f"[FAQ] 총 시간: {total_time:.1f}ms")
        
        # Top-3 결과 로그
        for idx, faq in enumerate(top_faqs, 1):
            print(f"[FAQ] #{idx}: {faq['question'][:45]}... "
                  f"(점수: {faq['final_score']:.3f} = 의미 {faq['semantic_score']:.3f} "
                  f"+ 키워드 {faq['keyword_score']:.1f})")
        
        # 답변 구성
        best = top_faqs[0]
        answer = f"[{best['category_name']}]\n\n"
        answer += f"질문: {best['question']}\n\n"
        answer += f"답변: {best['answer']}"
        
        related_questions = [faq['question'] for faq in top_faqs[1:]]
        
        print("=" * 80)
        return {
            "answer": answer,
            "relatedQuestions": related_questions
        }
        
    except Exception as e:
        print(f"[FAQ] 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": f"검색 중 오류가 발생했습니다: {str(e)}",
            "relatedQuestions": []
        }
