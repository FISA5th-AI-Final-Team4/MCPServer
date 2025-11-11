"""
FAQ PostgreSQL 기반 검색 모듈

LLM Server의 operation_id: query_faq_database
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any
from contextlib import contextmanager

from core.config import settings


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


def search_faq(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    사용자 질문에 대한 FAQ 검색 및 답변 생성
    
    Args:
        question: 사용자의 질문
        top_k: 반환할 최대 FAQ 개수
        
    Returns:
        {
            "success": bool,
            "query": str,
            "results": List[Dict],
            "answer": str,
            "total_found": int
        }
    """
    
    query = """
        SELECT 
            f.faq_id,
            f.question,
            f.answer,
            f.keywords,
            f.priority,
            f.views,
            c.category_name,
            similarity(f.question, %s) AS similarity
        FROM faqs f
        JOIN faq_categories c ON f.category_id = c.category_id
        WHERE similarity(f.question, %s) > 0.1
        ORDER BY similarity DESC, f.priority DESC, f.views DESC
        LIMIT %s;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (question, question, top_k))
                results = cursor.fetchall()
                
                if not results:
                    return {
                        "success": False,
                        "query": question,
                        "results": [],
                        "answer": "죄송합니다. 관련된 FAQ를 찾을 수 없습니다. 다른 질문을 해주시거나, 고객센터로 문의해주세요.",
                        "total_found": 0
                    }
                
                faq_list = [dict(row) for row in results]
                best_match = faq_list[0]
                
                answer = f"[{best_match['category_name']}]\n\n"
                answer += f"질문: {best_match['question']}\n\n"
                answer += f"답변: {best_match['answer']}\n"
                
                if len(faq_list) > 1:
                    answer += "\n\n[관련 FAQ]\n"
                    for idx, faq in enumerate(faq_list[1:], 1):
                        answer += f"{idx}. {faq['question']} (유사도: {faq['similarity']:.2f})\n"
                
                return {
                    "success": True,
                    "query": question,
                    "results": faq_list,
                    "answer": answer,
                    "total_found": len(faq_list),
                    "best_similarity": float(best_match['similarity'])
                }
                
    except Exception as e:
        print(f"FAQ 검색 오류: {e}")
        return {
            "success": False,
            "query": question,
            "results": [],
            "answer": f"검색 중 오류가 발생했습니다: {str(e)}",
            "total_found": 0
        }
