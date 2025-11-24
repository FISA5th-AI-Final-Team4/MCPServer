"""
금융 용어 PostgreSQL 기반 검색 모듈 (pg_bigm 유사도 검색)

처리 프로세스:
1. 키워드 추출 (Python): 질문에서 불필요한 어미/조사 제거
2. SQL 실행 (Python → DB): 추출된 키워드로 DB 질의
3. pg_bigm 유사도 계산 (DB): 오타/띄어쓰기 허용하며 유사도 점수 계산
4. 최고점 반환 (DB → Python): 가장 유사한 용어 1개 반환

LLM Server의 operation_id: query_term_database
"""

import re
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any
from contextlib import contextmanager

from core.config import settings


def extract_keyword(text: str) -> str:
    """
    1단계: 키워드 추출 (Python 영역)
    
    자연어 질문에서 핵심 용어만 추출
    - 조사 제거: ~이, ~가, ~을, ~를, ~은, ~는
    - 어미 제거: ~뭐야, ~이야, ~인가요, ~이에요
    - 특수문자 제거: ?, !, .
    
    Args:
        text: 사용자 질문 (예: "주택청약종합저축이 뭐야?")
        
    Returns:
        정제된 키워드 (예: "주택청약종합저축")
        
    Examples:
        >>> extract_keyword("연회비가 뭐야?")
        "연회비"
        >>> extract_keyword("APR이 무엇인가요?")
        "APR"
    """
    if not text:
        return text
    
    # 불필요한 패턴 제거
    patterns = [
        r'[이가]?\s*뭐야\??',          # "뭐야?", "가 뭐야?"
        r'[이가]?\s*무엇인가요\??',    # "무엇인가요?", "이 무엇인가요?"
        r'[이가]?\s*무엇이에요\??',    # "무엇이에요?"
        r'[이가]?\s*무슨\s*뜻',        # "무슨 뜻"
        r'[이가]?\s*뜻[이가]?\s*뭐',   # "뜻이 뭐", "뜻 뭐"
        r'[이가]?\s*의미[가는]?',      # "의미가", "의미는"
        r'[을를]\s*알려[줘주]?',       # "을 알려줘", "를 알려줘"
        r'[에대해서]?\s*설명',         # "에 대해 설명", "설명"
        r'[이란란]?\??',               # "이란?", "란?"
        r'[?!.\s]+$',                  # 끝의 특수문자 및 공백
    ]
    
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    # 앞뒤 공백 제거
    cleaned = cleaned.strip()
    
    return cleaned if cleaned else text


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


def search_term(term_query: str) -> str:
    """
    금융 용어 검색 (pg_bigm 유사도 기반)
    
    전체 처리 프로세스:
    1. 키워드 추출 (Python): 불필요한 어미 제거
    2. SQL 실행 (Python → DB): 키워드로 DB 질의
    3. pg_bigm 유사도 계산 (DB): 오타 허용 유사도 점수 계산
    4. 최고점 반환 (DB → Python): Top-1 용어 반환
    
    Args:
        term_query: 사용자 질문 (예: "주택청약종합저축이 뭐야?")
        
    Returns:
        str: 용어 정의와 관련 정보가 포함된 답변 텍스트
        
    Examples:
        >>> search_term("연회비가 뭐야?")
        "연회비 (Annual Fee)\n\n[카드 기본]\n\n[정의]\n..."
        
        >>> search_term("연회삐")  # 오타 허용
        "연회비 (Annual Fee)\n\n[카드 기본]\n\n[정의]\n..."
    """
    
    print(f"[TERM] 원본 질문: '{term_query}'")
    
    # 1단계: 키워드 추출 (Python 영역)
    keyword = extract_keyword(term_query)
    print(f"[TERM] 추출 키워드: '{keyword}'")
    
    # 2-4단계: pg_bigm 유사도 검색 (DB 영역)
    query = """
        SELECT 
            t.term_id,
            t.term,
            t.definition,
            t.english,
            t.related_terms,
            t.examples,
            c.category_name,
            bigm_similarity(t.term, %s) AS similarity_score
        FROM terms t
        JOIN term_categories c ON t.category_id = c.category_id
        WHERE LENGTH(t.term) > 1                     -- 1글자 용어 제외 (오탐 방지)
          AND bigm_similarity(t.term, %s) > 0.1      -- 유사도 임계값 (10% 이상)
        ORDER BY similarity_score DESC               -- 유사도 높은 순
        LIMIT 1;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # pg_bigm 확장 설치 확인 (최초 1회)
                cursor.execute("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'pg_bigm'
                    ) AS is_installed;
                """)
                is_installed = cursor.fetchone()['is_installed']
                
                if not is_installed:
                    print("[TERM] pg_bigm 확장 설치 중...")
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_bigm;")
                    conn.commit()
                    print("[TERM] pg_bigm 설치 완료")
                
                # 유사도 검색 실행
                cursor.execute(query, (keyword, keyword))
                result = cursor.fetchone()
                
                if not result:
                    return f"'{keyword}'에 대한 용어를 찾을 수 없습니다. 다른 키워드로 검색해주세요."
                
                term_info = dict(result)
                # Decimal 타입을 float로 변환
                similarity = float(term_info['similarity_score'])
                print(f"[TERM] 매칭 결과: '{term_info['term']}' (유사도: {similarity:.2f})")
                
                # 답변 구성
                answer = f"{term_info['term']}"
                if term_info['english']:
                    answer += f" ({term_info['english']})"
                answer += f"\n\n[{term_info['category_name']}]\n\n"
                answer += f"[정의]\n{term_info['definition']}\n"
                
                # 관련 용어 처리
                if term_info['related_terms']:
                    related_query = """
                        SELECT term, definition
                        FROM terms
                        WHERE term = ANY(%s)
                        LIMIT 5;
                    """
                    cursor.execute(related_query, (term_info['related_terms'],))
                    related_results = cursor.fetchall()
                    
                    if related_results:
                        answer += f"\n\n[관련 용어]\n"
                        for row in related_results:
                            answer += f"- {row['term']}: {row['definition'][:50]}...\n"
                
                return answer
                
    except Exception as e:
        print(f"[TERM] 오류: {e}")
        import traceback
        traceback.print_exc()
        return f"검색 중 오류가 발생했습니다: {str(e)}"