"""
금융 용어 PostgreSQL 기반 검색 모듈

LLM Server의 operation_id: query_term_database
"""

from typing import Dict, Any


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
    # TODO: PostgreSQL 연결 및 용어 검색 구현
    pass