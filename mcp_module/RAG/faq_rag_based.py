"""
FAQ Semantic Search 모듈

한국어 특화 의미 기반 검색:
- 임베딩 모델: jhgan/ko-sbert-multitask (110MB)
- 코사인 유사도로 FAQ 매칭
- 연관 질문 함께 반환
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any
from contextlib import contextmanager

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings

# ============================================================================
# 전역 모델 변수 (Lazy Loading)
# ============================================================================
_embedding_model = None


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


def get_all_faqs():
    """DB에서 모든 FAQ 가져오기"""
    query = """
        SELECT 
            f.faq_id, f.question, f.answer, c.category_name
        FROM faqs f
        JOIN faq_categories c ON f.category_id = c.category_id
        ORDER BY f.priority DESC, f.views DESC;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                return [dict(row) for row in results]
    except Exception as e:
        print(f"[FAQ] DB 조회 오류: {e}")
        return []


# ============================================================================
# 메인 검색 함수
# ============================================================================

def search_faq(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    순수 의미 검색 (Semantic Search Only)
    
    Args:
        question: 사용자 질문
        top_k: 반환할 FAQ 개수 (기본값: 3)
        
    Returns:
        {
            "answer": "최상위 FAQ 답변",
            "relatedQuestions": ["연관 질문1", "연관 질문2", ...]
        }
    """
    print(f"\n[FAQ] 검색: '{question}'")
    print("=" * 80)
    
    try:
        # 모든 FAQ 가져오기
        all_faqs = get_all_faqs()
        if not all_faqs:
            return {
                "answer": "FAQ 데이터를 불러올 수 없습니다.",
                "relatedQuestions": []
            }
        
        # 임베딩 모델 로드
        model = get_embedding_model()
        
        # 질문 임베딩
        q_embed = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
        
        # 모든 FAQ 질문 임베딩
        faq_questions = [faq['question'] for faq in all_faqs]
        faq_embeds = model.encode(
            faq_questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(q_embed.reshape(1, -1), faq_embeds)[0]
        
        # 유사도 점수 추가 및 정렬
        for i, faq in enumerate(all_faqs):
            faq['similarity'] = float(similarities[i])
        
        all_faqs.sort(key=lambda x: x['similarity'], reverse=True)
        top_faqs = all_faqs[:top_k]
        
        print(f"[FAQ] 검색 완료: Top-{top_k} (유사도: {top_faqs[0]['similarity']:.3f})")
        
        # 최상위 FAQ
        best = top_faqs[0]
        
        # 답변 구성
        answer = f"[{best['category_name']}]\n\n"
        answer += f"질문: {best['question']}\n\n"
        answer += f"답변: {best['answer']}"
        
        # 연관 질문 (2번째부터)
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
