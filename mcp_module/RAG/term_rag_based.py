"""
금융 용어 PostgreSQL 기반 검색 모듈

LLM Server의 operation_id: query_term_database
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


def search_term(term_query: str) -> Dict[str, Any]:
    """
    금융 용어 검색 및 설명 생성
    
    Args:
        term_query: 검색할 용어
        
    Returns:
        {
            "success": bool,
            "query": str,
            "term_info": Dict,
            "answer": str,
            "related_terms": List
        }
    """
    
    query = """
        SELECT 
            t.term_id,
            t.term,
            t.definition,
            t.english,
            t.related_terms,
            t.examples,
            c.category_name,
            similarity(t.term, %s) AS similarity
        FROM terms t
        JOIN term_categories c ON t.category_id = c.category_id
        WHERE similarity(t.term, %s) > 0.2
        ORDER BY similarity DESC
        LIMIT 1;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (term_query, term_query))
                result = cursor.fetchone()
                
                if not result:
                    return {
                        "success": False,
                        "query": term_query,
                        "term_info": None,
                        "answer": f"'{term_query}'에 대한 용어를 찾을 수 없습니다. 다른 키워드로 검색해주세요.",
                        "related_terms": []
                    }
                
                term_info = dict(result)
                
                answer = f"{term_info['term']}"
                if term_info['english']:
                    answer += f" ({term_info['english']})"
                answer += f"\n\n[{term_info['category_name']}]\n\n"
                answer += f"[정의]\n{term_info['definition']}\n"
                
                if term_info['examples']:
                    examples = term_info['examples']
                    if isinstance(examples, dict) and examples:
                        answer += f"\n\n[예시]\n"
                        for key, value in examples.items():
                            answer += f"- {value}\n"
                
                related_terms = []
                if term_info['related_terms']:
                    related_query = """
                        SELECT term, definition
                        FROM terms
                        WHERE term = ANY(%s)
                        LIMIT 5;
                    """
                    cursor.execute(related_query, (term_info['related_terms'],))
                    related_results = cursor.fetchall()
                    related_terms = [dict(row) for row in related_results]
                    
                    if related_terms:
                        answer += f"\n\n[관련 용어]\n"
                        for rt in related_terms:
                            answer += f"- {rt['term']}: {rt['definition'][:50]}...\n"
                
                return {
                    "success": True,
                    "query": term_query,
                    "term_info": term_info,
                    "answer": answer,
                    "related_terms": related_terms,
                    "similarity": float(term_info['similarity'])
                }
                
    except Exception as e:
        print(f"용어 검색 오류: {e}")
        return {
            "success": False,
            "query": term_query,
            "term_info": None,
            "answer": f"검색 중 오류가 발생했습니다: {str(e)}",
            "related_terms": []
        }