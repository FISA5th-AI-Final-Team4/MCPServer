"""
QnA 시스템 종합 테스트

테스트 케이스:
1. FAQ 검색
   - 정확한 키워드 매칭
   - 부분 키워드 매칭
   - 여러 키워드 매칭
   - 매칭 실패 케이스
   
2. Term 검색
   - 정확한 용어 매칭
   - 부분 문자열 매칭
   - 띄어쓰기 포함 검색
   - 매칭 실패 케이스

3. 성능 테스트
   - 응답 시간 측정
"""

import time
from mcp_module.RAG.faq_rag_based import search_faq
from mcp_module.RAG.term_rag_based import search_term


def print_separator(title: str):
    """테스트 구분선 출력"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_faq_search():
    """FAQ 검색 테스트"""
    print_separator("FAQ 검색 테스트")
    
    test_cases = [
        {
            "name": "정확한 키워드 매칭",
            "query": "발급",
            "expected": "FAQ-001"
        },
        {
            "name": "여러 키워드 매칭",
            "query": "카드 발급 방법",
            "expected": "FAQ-001"
        },
        {
            "name": "혜택 관련 질문",
            "query": "편의점 할인",
            "expected": "FAQ-002"
        },
        {
            "name": "결제 방법 질문",
            "query": "할부 개월수",
            "expected": "FAQ-003"
        },
        {
            "name": "부분 키워드 (일시불)",
            "query": "일시불",
            "expected": "FAQ-003"
        },
        {
            "name": "매칭 없음 (존재하지 않는 키워드)",
            "query": "해외여행 보험",
            "expected": None
        },
        {
            "name": "짧은 단어 (필터링 테스트)",
            "query": "카드는 어떻게?",
            "expected": "check"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {test['name']}")
        print(f"쿼리: '{test['query']}'")
        
        start_time = time.time()
        result = search_faq(test['query'], top_k=3)
        elapsed = (time.time() - start_time) * 1000
        
        print(f"응답 시간: {elapsed:.2f}ms")
        print(f"성공 여부: {result['success']}")
        print(f"검색된 FAQ 수: {result['total_found']}")
        
        if result['success']:
            print(f"최고 매칭: {result['results'][0]['faq_id']}")
            print(f"매칭 키워드 수: {result['best_similarity']}")
            print(f"\n답변 미리보기:\n{result['answer'][:200]}...")
        else:
            print(f"답변: {result['answer']}")
        
        # 검증
        if test['expected']:
            if test['expected'] == "check":
                print(f"✓ 실행 완료 (수동 확인 필요)")
            elif result['success']:
                actual_id = result['results'][0]['faq_id']
                if actual_id == test['expected']:
                    print(f"✓ 테스트 통과 (예상: {test['expected']}, 실제: {actual_id})")
                else:
                    print(f"✗ 테스트 실패 (예상: {test['expected']}, 실제: {actual_id})")
            else:
                print(f"✗ 검색 실패 (예상: {test['expected']})")
        else:
            if not result['success']:
                print(f"✓ 테스트 통과 (매칭 없음 예상)")
            else:
                print(f"✗ 테스트 실패 (매칭 없어야 함)")


def test_term_search():
    """용어 검색 테스트"""
    print_separator("용어 검색 테스트")
    
    test_cases = [
        {
            "name": "정확한 용어 매칭",
            "query": "연회비",
            "expected": "TERM-001",
            "expected_sim": 1.0
        },
        {
            "name": "정확한 용어 매칭 (캐시백)",
            "query": "캐시백",
            "expected": "TERM-002",
            "expected_sim": 1.0
        },
        {
            "name": "정확한 용어 매칭 (신용 한도)",
            "query": "신용 한도",
            "expected": "TERM-003",
            "expected_sim": 1.0
        },
        {
            "name": "부분 문자열 매칭",
            "query": "한도",
            "expected": "TERM-003",
            "expected_sim": 0.8
        },
        {
            "name": "띄어쓰기 포함 (연 회비)",
            "query": "연 회비",
            "expected": None  # 현재 구현에서는 실패 예상
        },
        {
            "name": "영문 검색 시도",
            "query": "Annual Fee",
            "expected": None  # 영문 필드는 검색 안함
        },
        {
            "name": "매칭 없음",
            "query": "주식투자",
            "expected": None
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {test['name']}")
        print(f"쿼리: '{test['query']}'")
        
        start_time = time.time()
        result = search_term(test['query'])
        elapsed = (time.time() - start_time) * 1000
        
        print(f"응답 시간: {elapsed:.2f}ms")
        print(f"성공 여부: {result['success']}")
        
        if result['success']:
            term_info = result['term_info']
            print(f"검색된 용어: {term_info['term']}")
            print(f"용어 ID: {term_info['term_id']}")
            print(f"유사도: {result['similarity']}")
            print(f"영문: {term_info.get('english', 'N/A')}")
            print(f"관련 용어: {len(result['related_terms'])}개")
            print(f"\n정의 미리보기:\n{term_info['definition'][:100]}...")
        else:
            print(f"답변: {result['answer']}")
        
        # 검증
        if test['expected']:
            if result['success']:
                actual_id = term_info['term_id']
                actual_sim = result['similarity']
                if actual_id == test['expected']:
                    print(f"✓ 용어 ID 일치 (예상: {test['expected']}, 실제: {actual_id})")
                    if 'expected_sim' in test and actual_sim == test['expected_sim']:
                        print(f"✓ 유사도 일치 (예상: {test['expected_sim']}, 실제: {actual_sim})")
                else:
                    print(f"✗ 테스트 실패 (예상: {test['expected']}, 실제: {actual_id})")
            else:
                print(f"✗ 검색 실패 (예상: {test['expected']})")
        else:
            if not result['success']:
                print(f"✓ 테스트 통과 (매칭 없음 예상)")
            else:
                print(f"✗ 테스트 실패 (매칭 없어야 함)")


def test_performance():
    """성능 테스트 (100회 반복)"""
    print_separator("성능 테스트 (100회 반복)")
    
    # FAQ 성능
    print("\n[FAQ 검색 성능]")
    times = []
    for _ in range(100):
        start = time.time()
        search_faq("카드 발급")
        times.append((time.time() - start) * 1000)
    
    print(f"평균: {sum(times)/len(times):.2f}ms")
    print(f"최소: {min(times):.2f}ms")
    print(f"최대: {max(times):.2f}ms")
    print(f"중앙값: {sorted(times)[len(times)//2]:.2f}ms")
    
    # Term 성능
    print("\n[용어 검색 성능]")
    times = []
    for _ in range(100):
        start = time.time()
        search_term("연회비")
        times.append((time.time() - start) * 1000)
    
    print(f"평균: {sum(times)/len(times):.2f}ms")
    print(f"최소: {min(times):.2f}ms")
    print(f"최대: {max(times):.2f}ms")
    print(f"중앙값: {sorted(times)[len(times)//2]:.2f}ms")


def test_edge_cases():
    """엣지 케이스 테스트"""
    print_separator("엣지 케이스 테스트")
    
    test_cases = [
        ("빈 문자열", ""),
        ("공백만", "   "),
        ("특수문자만", "!@#$%"),
        ("아주 긴 질문", "카드 " * 100),
        ("숫자 포함", "123456"),
        ("한글자", "카"),
    ]
    
    for name, query in test_cases:
        print(f"\n[테스트] {name}")
        print(f"쿼리: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        try:
            result = search_faq(query)
            print(f"FAQ 결과: {result['success']} (검색된 수: {result['total_found']})")
        except Exception as e:
            print(f"FAQ 오류: {e}")
        
        try:
            result = search_term(query)
            print(f"Term 결과: {result['success']}")
        except Exception as e:
            print(f"Term 오류: {e}")


def test_db_data():
    """DB 데이터 확인 테스트"""
    print_separator("데이터베이스 데이터 확인")
    
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from core.config import settings
    
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # FAQ 통계
        cursor.execute("SELECT COUNT(*) as cnt FROM faqs;")
        faq_count = cursor.fetchone()['cnt']
        print(f"\n전체 FAQ 수: {faq_count}개")
        
        cursor.execute("SELECT faq_id, question, array_length(keywords, 1) as keyword_count FROM faqs;")
        faqs = cursor.fetchall()
        print("\nFAQ 목록:")
        for faq in faqs:
            print(f"  - {faq['faq_id']}: {faq['question'][:30]}... (키워드: {faq['keyword_count']}개)")
        
        # Term 통계
        cursor.execute("SELECT COUNT(*) as cnt FROM terms;")
        term_count = cursor.fetchone()['cnt']
        print(f"\n전체 용어 수: {term_count}개")
        
        cursor.execute("SELECT term_id, term, english FROM terms;")
        terms = cursor.fetchall()
        print("\n용어 목록:")
        for term in terms:
            print(f"  - {term['term_id']}: {term['term']} ({term['english']})")
        
        cursor.close()
        conn.close()
        print("\n✓ DB 연결 및 데이터 확인 완료")
        
    except Exception as e:
        print(f"✗ DB 연결 오류: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  QnA 시스템 종합 테스트 시작")
    print("=" * 80)
    
    # 1. DB 데이터 확인
    test_db_data()
    
    # 2. FAQ 검색 테스트
    test_faq_search()
    
    # 3. 용어 검색 테스트
    test_term_search()
    
    # 4. 엣지 케이스
    test_edge_cases()
    
    # 5. 성능 테스트
    test_performance()
    
    print_separator("테스트 완료")
