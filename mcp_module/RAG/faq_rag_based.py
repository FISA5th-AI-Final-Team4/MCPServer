"""
FAQ PostgreSQL 기반 검색 모듈

LLM Server의 operation_id: query_faq_database
"""

import re
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any
from contextlib import contextmanager

from core.config import settings


def normalize_korean(text: str) -> str:
    """
    한글 텍스트 정규화
    
    띄어쓰기, 하이픈, 언더스코어 제거하여 검색 유연성 향상
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        정규화된 텍스트 (공백 제거)
        
    Examples:
        >>> normalize_korean("연 회비")
        "연회비"
        >>> normalize_korean("카드-발급")
        "카드발급"
    """
    if not text:
        return text
    # 공백, 하이픈, 언더스코어 제거
    text = re.sub(r'[\s\-_]', '', text)
    return text.lower()


@contextmanager
def get_db_connection():
    """PostgreSQL 데이터베이스 연결 contextmanager"""
    conn = None
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        yield conn
    except psycopg2.Error as e:
        print(f"데이터베이스 연결 오류: {e}")
        raise
    finally:
        if conn:
            conn.close()


def search_faq(question: str, top_k: int = 3) -> str:
    """
    사용자 질문에 대한 FAQ 검색 및 답변 생성
    
    키워드 배열 기반 검색 (pg_trgm은 한글 지원 제한)
    
    Args:
        question: 사용자의 질문
        top_k: 반환할 최대 FAQ 개수
        
    Returns:
        str: FAQ 답변이 포함된 텍스트
    """
    
    # 질문에서 키워드 추출
    question_words = [word for word in question.replace('?', '').replace(',', '').split() if len(word) > 1]
    
    # 정규화된 키워드 추가 (띄어쓰기 제거)
    normalized_words = [normalize_korean(word) for word in question_words]
    
    # 전체 질문을 정규화한 후 다시 분리 (연속된 단어 조합 포착)
    # 예: "카드 발 급" → "카드발급" → n-gram으로 ["카드", "발급", "카드발급"] 등 추출
    normalized_full = normalize_korean(question)
    # 정규화된 전체 문장에서 2-4글자 단어 추출
    additional_words = []
    for i in range(len(normalized_full)):
        for length in [2, 3, 4]:
            if i + length <= len(normalized_full):
                word = normalized_full[i:i+length]
                if len(word) > 1:
                    additional_words.append(word)
    
    # 원본 + 정규화 + 추가 단어 합침 (중복 제거)
    all_keywords = list(set(question_words + normalized_words + additional_words))
    
    query = """
        SELECT 
            f.faq_id,
            f.question,
            f.answer,
            f.keywords,
            f.priority,
            f.views,
            c.category_name,
            (
                SELECT COUNT(*)
                FROM unnest(f.keywords) AS k
                WHERE k = ANY(%s::text[])
            ) AS match_count
        FROM faqs f
        JOIN faq_categories c ON f.category_id = c.category_id
        WHERE f.keywords && %s::text[]
        ORDER BY match_count DESC, f.priority DESC, f.views DESC
        LIMIT %s;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (all_keywords, all_keywords, top_k))
                results = cursor.fetchall()
                
                if not results:
                    return "죄송합니다. 관련된 FAQ를 찾을 수 없습니다. 다른 질문을 해주시거나, 고객센터로 문의해주세요."
                
                faq_list = [dict(row) for row in results]
                best_match = faq_list[0]
                
                answer = f"[{best_match['category_name']}]\n\n"
                answer += f"질문: {best_match['question']}\n\n"
                answer += f"답변: {best_match['answer']}\n"
                
                if len(faq_list) > 1:
                    answer += "\n\n[관련 FAQ]\n"
                    for idx, faq in enumerate(faq_list[1:], 1):
                        answer += f"{idx}. {faq['question']} (매칭 키워드: {faq['match_count']}개)\n"
                
                return answer
                
    except Exception as e:
        print(f"FAQ 검색 오류: {e}")
        return f"검색 중 오류가 발생했습니다: {str(e)}"
