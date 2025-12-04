"""
금융 용어 PostgreSQL 기반 검색 모듈 (pg_bigm 유사도 검색)

처리 프로세스:
1. 키워드 추출 (Python): 질문에서 불필요한 어미/조사 제거
2. SQL 실행 (Python → DB): 추출된 키워드로 DB 질의
3. pg_bigm 유사도 계산 (DB): 2-gram 기반 오타/띄어쓰기 허용 유사도 점수 계산
4. 최고점 반환 (DB → Python): 가장 유사한 용어 1개 반환

특징:
- pg_bigm: 한글/CJK 문자 최적화 (2-gram)
- 오타 1~2글자 허용 (예: "연회삐" → "연회비")
- 띄어쓰기 무시 (예: "연 회비" → "연회비")

LLM Server의 operation_id: query_term_database
"""

import re
import psycopg2
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


def search_term(term_query: str, top_k: int = 3) -> dict:
    """
    금융 용어 검색 (pg_bigm 유사도 기반)
    
    Args:
        term_query: 사용자 질문 (예: "연회비가 뭐야?", "연회삐")
        top_k: 반환할 용어 개수 (기본값: 3)
        
    Returns:
        dict: {
            'answer': str - 최상위 용어 정의,
            'relatedQuestions': List[str] - 관련 용어 목록 (최대 2개)
        }
        
    Examples:
        >>> search_term("연회비가 뭐야?")
        {
            'answer': "연회비 (Annual Fee)\n\n[카드 기본]\n\n[정의]\n...",
            'relatedQuestions': ["실적", "한도"]
        }
    """
    print(f"[TERM] 검색: '{term_query}'")
    
    # 키워드 추출
    keyword = extract_keyword(term_query)
    print(f"[TERM] 키워드: '{keyword}'")
    
    # SQL 쿼리 (pg_bigm 유사도 검색) - 상위 3개 반환
    query = """
        SELECT 
            t.term_id, t.term, t.definition, t.english,
            c.category_name,
            bigm_similarity(t.term, '{keyword}') AS similarity_score
        FROM terms t
        JOIN term_categories c ON t.category_id = c.category_id
        WHERE LENGTH(t.term) > 1
          AND bigm_similarity(t.term, '{keyword}') > 0.1
        ORDER BY similarity_score DESC
        LIMIT {top_k};
    """.format(keyword=keyword.replace("'", "''"), top_k=top_k)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # pg_bigm 확장 활성화 확인
                cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_bigm');")
                if not cursor.fetchone()[0]:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_bigm;")
                    conn.commit()
                    print("[TERM] pg_bigm 확장 설치 완료")
                
                # 유사도 검색 (상위 3개)
                cursor.execute(query)
                results = cursor.fetchall()
                
                if not results:
                    return {
                        "answer": f"'{keyword}'에 대한 용어를 찾을 수 없습니다.",
                        "relatedQuestions": []
                    }
                
                # Tuple → Dict 변환
                columns = ['term_id', 'term', 'definition', 'english', 
                          'category_name', 'similarity_score']
                term_list = [dict(zip(columns, row)) for row in results]
                
                # 최상위 용어 (1순위)
                best = term_list[0]
                similarity = float(best['similarity_score'])
                print(f"[TERM] 매칭: '{best['term']}' (유사도: {similarity:.2f})")
                
                # 답변 구성 (1순위 용어)
                answer = best['term']
                if best['english']:
                    answer += f" ({best['english']})"
                answer += f"\n\n[{best['category_name']}]\n\n"
                answer += f"[정의]\n{best['definition']}\n"
                
                # 관련 용어 목록 (2, 3순위 용어명)
                related_questions = [term['term'] for term in term_list[1:]]
                
                # views 카운트 증가 (1순위만)
                cursor.execute(
                    "UPDATE terms SET views = views + 1 WHERE term_id = %s;",
                    (best['term_id'],)
                )
                conn.commit()
                print(f"[TERM] views 증가: term_id={best['term_id']}")
                print(f"[TERM] 관련 용어: {related_questions}")
                
                return {
                    "answer": answer,
                    "relatedQuestions": related_questions
                }
                
    except Exception as e:
        print(f"[TERM] 오류: {e}")
        return {
            "answer": f"검색 중 오류가 발생했습니다: {str(e)}",
            "relatedQuestions": []
        }