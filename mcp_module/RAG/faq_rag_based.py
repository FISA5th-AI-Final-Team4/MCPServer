"""
FAQ Hybrid Search 모듈

한국어 특화 4단계 검색 파이프라인:
1. Keyword Search - PostgreSQL GIN 인덱스 활용 (후보군 10개)
2. Semantic Search - 임베딩 기반 의미 유사도 계산
3. Hybrid Score - 키워드(30%) + 의미(70%) 결합
4. Reranker - Cross-Encoder로 최종 정확도 향상 (top-3)
"""

import re
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
from contextlib import contextmanager

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings

# ============================================================================
# 전역 모델 변수 (Lazy Loading)
# ============================================================================
_embedding_model = None  # jhgan/ko-sbert-multitask (110MB)
_reranker_model = None   # jhgan/ko-sroberta-multitask (110MB)


def get_embedding_model() -> SentenceTransformer:
    """임베딩 모델 로드 (한국어 특화 BERT, 384차원)"""
    global _embedding_model
    if _embedding_model is None:
        print("[FAQ] 임베딩 모델 로드 중...")
        _embedding_model = SentenceTransformer(
            'jhgan/ko-sbert-multitask',
            device=settings.DEVICE
        )
        print(f"[FAQ] 임베딩 모델 로드 완료 (Device: {settings.DEVICE})")
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    """리랭커 모델 로드 (한국어 특화 RoBERTa Cross-Encoder)"""
    global _reranker_model
    if _reranker_model is None:
        print("[FAQ] Reranker 모델 로드 중...")
        _reranker_model = CrossEncoder(
            'jhgan/ko-sroberta-multitask',
            max_length=512,
            device=settings.DEVICE
        )
        print(f"[FAQ] Reranker 모델 로드 완료 (Device: {settings.DEVICE})")
    return _reranker_model


# ============================================================================
# 유틸리티 함수
# ============================================================================

def normalize_korean(text: str) -> str:
    """한글 텍스트 정규화 (띄어쓰기, 특수문자 제거)"""
    if not text:
        return text
    return re.sub(r'[\s\-_]', '', text).lower()


def extract_keywords_with_ngram(question: str) -> List[str]:
    """
    질문에서 키워드 추출 (3단계 전략)
    1. 원본 단어 (2글자 이상)
    2. 정규화 단어 (띄어쓰기 제거)
    3. N-gram (2~4글자)
    """
    # 1단계: 원본 단어
    words = [w for w in question.replace('?', '').replace(',', '').split() if len(w) > 1]
    
    # 2단계: 정규화 단어
    normalized = [normalize_korean(w) for w in words]
    
    # 3단계: N-gram
    full_normalized = normalize_korean(question)
    ngrams = [
        full_normalized[i:i+length]
        for i in range(len(full_normalized))
        for length in [2, 3, 4]
        if i + length <= len(full_normalized)
    ]
    
    return list(set(words + normalized + ngrams))


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


# ============================================================================
# 1단계: Keyword Search
# ============================================================================

def keyword_search(question: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """PostgreSQL GIN 인덱스 기반 키워드 매칭"""
    keywords = extract_keywords_with_ngram(question)
    
    query = """
        SELECT 
            f.faq_id, f.question, f.answer, f.keywords,
            f.priority, f.views, c.category_name,
            (SELECT COUNT(*) FROM unnest(f.keywords) AS k WHERE k = ANY(%s::text[])) AS match_count
        FROM faqs f
        JOIN faq_categories c ON f.category_id = c.category_id
        WHERE f.keywords && %s::text[]
        ORDER BY match_count DESC, f.priority DESC, f.views DESC
        LIMIT %s;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (keywords, keywords, top_k))
                results = cursor.fetchall()
                
                if not results:
                    return []
                
                # 키워드 점수 정규화 (0~1)
                max_match = max(row['match_count'] for row in results)
                faq_list = []
                for row in results:
                    faq = dict(row)
                    faq['keyword_score'] = faq['match_count'] / max_match if max_match > 0 else 0
                    faq_list.append(faq)
                
                print(f"[FAQ] 키워드 검색: {len(faq_list)}개")
                return faq_list
                
    except Exception as e:
        print(f"[FAQ] 키워드 검색 오류: {e}")
        return []


# ============================================================================
# 2단계: Semantic Search
# ============================================================================

def semantic_search(question: str, faq_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """임베딩 기반 의미 유사도 계산 (코사인 유사도)"""
    if not faq_candidates:
        return []
    
    model = get_embedding_model()
    
    # 질문 임베딩
    q_embed = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    
    # FAQ 임베딩 (배치 처리)
    faq_embeds = model.encode(
        [faq['question'] for faq in faq_candidates],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(q_embed.reshape(1, -1), faq_embeds)[0]
    
    # 점수 추가
    for i, faq in enumerate(faq_candidates):
        faq['semantic_score'] = float(similarities[i])
    
    print(f"[FAQ] 의미 검색: 유사도 [{similarities.min():.3f}~{similarities.max():.3f}]")
    return faq_candidates


# ============================================================================
# 3단계: Hybrid Score
# ============================================================================

def calculate_hybrid_score(
    faq_candidates: List[Dict[str, Any]], 
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """하이브리드 점수 계산 (키워드 30% + 의미 70%)"""
    for faq in faq_candidates:
        faq['hybrid_score'] = (
            faq.get('keyword_score', 0) * keyword_weight +
            faq.get('semantic_score', 0) * semantic_weight
        )
    
    faq_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
    print(f"[FAQ] 하이브리드 점수: 키워드({keyword_weight}) + 의미({semantic_weight})")
    return faq_candidates


# ============================================================================
# 4단계: Reranker
# ============================================================================

def rerank_faqs(question: str, faq_candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Cross-Encoder 기반 최종 정확도 재평가"""
    if not faq_candidates:
        return []
    
    model = get_reranker_model()
    
    # 질문-FAQ 쌍 생성
    pairs = [[question, f"{faq['question']} {faq['answer']}"] for faq in faq_candidates]
    
    # 리랭킹 점수 계산
    scores = model.predict(pairs, show_progress_bar=False)
    
    # 점수 추가 및 정렬
    for i, faq in enumerate(faq_candidates):
        faq['rerank_score'] = float(scores[i])
    
    faq_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    top_faqs = faq_candidates[:top_k]
    
    print(f"[FAQ] Reranking: Top-{top_k}")
    for i, faq in enumerate(top_faqs, 1):
        print(f"  {i}. {faq['question'][:30]}... ({faq['rerank_score']:.3f})")
    
    return top_faqs


# ============================================================================
# 메인 검색 함수
# ============================================================================

def search_faq(question: str, top_k: int = 3) -> str:
    """
    4단계 Hybrid Search 파이프라인
    1. Keyword Search (후보 10개)
    2. Semantic Search (의미 유사도)
    3. Hybrid Score (키워드 30% + 의미 70%)
    4. Reranker (최종 top-3)
    """
    print(f"\n[FAQ] 검색: '{question}'")
    print("=" * 80)
    
    try:
        # 1단계: 키워드 검색
        candidates = keyword_search(question, top_k=max(top_k * 3, 10))
        if not candidates:
            return "관련된 FAQ를 찾을 수 없습니다. 고객센터(1588-0000)로 문의해주세요."
        
        # 2단계: 의미 검색
        candidates = semantic_search(question, candidates)
        
        # 3단계: 하이브리드 점수
        candidates = calculate_hybrid_score(candidates, keyword_weight=0.3, semantic_weight=0.7)
        
        # 4단계: 리랭킹
        final_faqs = rerank_faqs(question, candidates, top_k=top_k)
        if not final_faqs:
            return "적절한 FAQ를 찾을 수 없습니다."
        
        # 답변 포맷팅
        best = final_faqs[0]
        answer = f"[{best['category_name']}]\n\n질문: {best['question']}\n\n답변: {best['answer']}\n"
        
        if len(final_faqs) > 1:
            answer += "\n\n[관련 FAQ]\n"
            for idx, faq in enumerate(final_faqs[1:], 1):
                answer += f"{idx}. {faq['question']} (관련도: {faq['rerank_score']:.2f})\n"
        
        print("=" * 80)
        print(f"[FAQ] 완료: {len(final_faqs)}개 반환\n")
        return answer
        
    except Exception as e:
        print(f"[FAQ] 오류: {e}")
        import traceback
        traceback.print_exc()
        return f"검색 중 오류가 발생했습니다: {str(e)}"
