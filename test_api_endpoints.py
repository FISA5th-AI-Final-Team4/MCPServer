"""
API 엔드포인트 테스트 스크립트
"""

import requests
import json

BASE_URL = "http://localhost:8011"

def test_faq_query():
    """FAQ 검색 테스트"""
    print("=" * 80)
    print("FAQ 검색 테스트")
    print("=" * 80)
    
    test_cases = [
        {"query": "발급", "top_k": 3},
        {"query": "카드 발급 방법", "top_k": 3},
        {"query": "편의점 할인", "top_k": 3},
        {"query": "할부", "top_k": 3},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] 쿼리: {case['query']}")
        try:
            response = requests.post(
                f"{BASE_URL}/tools/faq-query",
                json=case,
                timeout=10
            )
            
            print(f"상태 코드: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"성공: {data['success']}")
                print(f"검색된 FAQ 수: {data['total_found']}")
                if data['success'] and data['results']:
                    print(f"최고 매칭: {data['results'][0]['faq_id']}")
                    print(f"유사도: {data['best_similarity']}")
                    print(f"답변 미리보기:\n{data['answer'][:200]}...")
                else:
                    print(f"답변: {data['answer']}")
            else:
                print(f"오류: {response.text}")
                
        except Exception as e:
            print(f"요청 실패: {e}")


def test_term_query():
    """용어 검색 테스트"""
    print("\n\n" + "=" * 80)
    print("용어 검색 테스트")
    print("=" * 80)
    
    test_cases = [
        {"query": "연회비"},
        {"query": "캐시백"},
        {"query": "신용 한도"},
        {"query": "한도"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] 쿼리: {case['query']}")
        try:
            response = requests.post(
                f"{BASE_URL}/tools/term-query",
                json=case,
                timeout=10
            )
            
            print(f"상태 코드: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"성공: {data['success']}")
                if data['success'] and data['term_info']:
                    print(f"검색된 용어: {data['term_info']['term']}")
                    print(f"용어 ID: {data['term_info']['term_id']}")
                    print(f"유사도: {data['similarity']}")
                    print(f"영문: {data['term_info']['english']}")
                    print(f"답변 미리보기:\n{data['answer'][:200]}...")
                else:
                    print(f"답변: {data['answer']}")
            else:
                print(f"오류: {response.text}")
                
        except Exception as e:
            print(f"요청 실패: {e}")


def test_health():
    """헬스 체크"""
    print("\n\n" + "=" * 80)
    print("헬스 체크")
    print("=" * 80)
    
    try:
        response = requests.get(f"{BASE_URL}/tools/health", timeout=5)
        print(f"상태 코드: {response.status_code}")
        print(f"응답: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"요청 실패: {e}")


if __name__ == "__main__":
    import time
    
    print("서버 시작 대기 중...")
    time.sleep(2)
    
    test_health()
    test_faq_query()
    test_term_query()
    
    print("\n\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)
