"""
FAQ Hybrid Search 모듈 (키워드 + 임베딩 + Reranker)

LLM Server의 operation_id: query_faq_database

검색 파이프라인:
1. Keyword Search (PostgreSQL GIN 인덱스) - 빠른 후보군 수집
2. Semantic Search (문장 임베딩 기반) - 의미론적 유사도
3. Hybrid Score (키워드 30% + 임베딩 70%) - 최종 점수 계산
4. Reranker (Cross-Encoder) - 정확도 재평가
"""

import re
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Tuple
from contextlib import contextmanager
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings

# ============================================================================
# 전역 모델 로드 (앱 시작 시 1회만 로드)
# ============================================================================

# 임베딩 모델: 다국어 지원 경량 모델 (약 420MB)
# 질문과 FAQ를 벡터로 변환하여 의미적 유사도 계산
_embedding_model = None

# Reranker 모델: 질문-FAQ 쌍의 관련성 재평가 (약 280MB)
# Cross-Encoder 방식으로 더 정확한 점수 산출
_reranker_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    임베딩 모델 Lazy Loading
    
    jhgan/ko-sbert-multitask:
    - 한국어 특화 BERT (KoBERT 기반)
    - 384차원 벡터
    - 약 110MB (매우 가벼움, 기존 420MB → 110MB)
    - 한국어 문장 유사도에 최적화
    """
    global _embedding_model
    if _embedding_model is None:
        print("[FAQ] 임베딩 모델 로드 중... (ko-sbert-multitask, ~110MB)")
        _embedding_model = SentenceTransformer(
            'jhgan/ko-sbert-multitask',
            device=settings.DEVICE
        )
        print(f"[FAQ] 임베딩 모델 로드 완료 (Device: {settings.DEVICE})")
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    """
    Reranker 모델 Lazy Loading
    
    cross-encoder/ms-marco-MiniLM-L-2-v2:
    - MS-MARCO 데이터로 학습된 초경량 모델
    - 질문-문서 쌍의 관련성 점수 (0~1)
    - 약 60MB (초경량, 기존 280MB → 60MB)
    - 2-layer Transformer (매우 빠른 추론)
    """
    global _reranker_model
    if _reranker_model is None:
        print("[FAQ] Reranker 모델 로드 중... (ms-marco-MiniLM-L-2-v2, ~60MB)")
        _reranker_model = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-2-v2',
            max_length=512,
            device=settings.DEVICE
        )
        print(f"[FAQ] Reranker 모델 로드 완료 (Device: {settings.DEVICE})")
    return _reranker_model


# ============================================================================
# 유틸리티 함수
# ============================================================================

def normalize_korean(text: str) -> str:
    """
    한글 텍스트 정규화 (키워드 검색용)
    
    띄어쓰기, 하이픈, 언더스코어 제거하여 검색 유연성 향상
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        정규화된 텍스트 (소문자 변환, 공백 제거)
        
    Examples:
        >>> normalize_korean("연 회비")
        "연회비"
        >>> normalize_korean("카드-발급")
        "카드발급"
    """
    if not text:
        return text
    # 공백, 하이픈, 언더스코어 제거 후 소문자 변환
    text = re.sub(r'[\s\-_]', '', text)
    return text.lower()


def extract_keywords_with_ngram(question: str) -> List[str]:
    """
    질문에서 키워드 추출 (3단계 전략)
    
    1. 원본 단어 분리
    2. 정규화된 단어
    3. N-gram (2-4글자) 조합
    
    Args:
        question: 사용자 질문
        
    Returns:
        중복 제거된 키워드 리스트
        
    Examples:
        >>> extract_keywords_with_ngram("카드 발급 방법")
        ["카드", "발급", "방법", "카드발급", "발급방법", "카드발", ...]
    """
    # 1단계: 원본 키워드 (2글자 이상)
    question_words = [
        word for word in question.replace('?', '').replace(',', '').split() 
        if len(word) > 1
    ]
    
    # 2단계: 정규화된 키워드 (띄어쓰기 제거)
    normalized_words = [normalize_korean(word) for word in question_words]
    
    # 3단계: N-gram 키워드 (2-4글자)
    # 전체 질문을 정규화한 후 n-gram 추출
    normalized_full = normalize_korean(question)
    additional_words = []
    for i in range(len(normalized_full)):
        for length in [2, 3, 4]:  # 2글자, 3글자, 4글자 조합
            if i + length <= len(normalized_full):
                word = normalized_full[i:i+length]
                if len(word) > 1:
                    additional_words.append(word)
    
    # 모든 키워드 합치고 중복 제거
    all_keywords = list(set(question_words + normalized_words + additional_words))
    
    return all_keywords


@contextmanager
def get_db_connection():
    """
    PostgreSQL 데이터베이스 연결 contextmanager
    
    자동으로 연결 관리 (종료 시 close)
    """
    conn = None
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        yield conn
    except psycopg2.Error as e:
        print(f"[FAQ] 데이터베이스 연결 오류: {e}")
        raise
    finally:
        if conn:
            conn.close()


# ============================================================================
# 1단계: Keyword Search (빠른 후보군 수집)
# ============================================================================

def keyword_search(question: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    키워드 배열 매칭으로 후보 FAQ 수집
    
    PostgreSQL GIN 인덱스를 활용한 고속 검색
    - 배열 교집합 연산자 (&&) 사용
    - 매칭 키워드 개수로 정렬
    
    Args:
        question: 사용자 질문
        top_k: 반환할 최대 FAQ 개수
        
    Returns:
        FAQ 리스트 (각 항목: faq_id, question, answer, keywords, match_count 등)
    """
    # 질문에서 키워드 추출
    all_keywords = extract_keywords_with_ngram(question)
    
    # SQL 쿼리: 키워드 매칭 + 매칭 개수 계산
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
                -- 매칭된 키워드 개수 계산
                SELECT COUNT(*)
                FROM unnest(f.keywords) AS k
                WHERE k = ANY(%s::text[])
            ) AS match_count
        FROM faqs f
        JOIN faq_categories c ON f.category_id = c.category_id
        WHERE f.keywords && %s::text[]  -- 배열 교집합 (최소 1개 이상 매칭)
        ORDER BY match_count DESC, f.priority DESC, f.views DESC
        LIMIT %s;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (all_keywords, all_keywords, top_k))
                results = cursor.fetchall()
                
                # 키워드 점수 계산 (0~1 정규화)
                faq_list = []
                max_match = max([row['match_count'] for row in results], default=1)
                
                for row in results:
                    faq = dict(row)
                    # 키워드 점수 = 매칭 개수 / 최대 매칭 개수
                    faq['keyword_score'] = faq['match_count'] / max_match if max_match > 0 else 0
                    faq_list.append(faq)
                
                print(f"[FAQ] 키워드 검색 완료: {len(faq_list)}개 후보")
                return faq_list
                
    except Exception as e:
        print(f"[FAQ] 키워드 검색 오류: {e}")
        return []


# ============================================================================
# 2단계: Semantic Search (의미론적 유사도 계산)
# ============================================================================

def semantic_search(question: str, faq_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    임베딩 기반 의미론적 유사도 계산
    
    질문과 FAQ 질문을 벡터로 변환 후 코사인 유사도 측정
    - 동의어/유사어 자동 인식
    - 문맥 이해 가능
    
    Args:
        question: 사용자 질문
        faq_candidates: 키워드 검색 결과 (후보 FAQ 리스트)
        
    Returns:
        semantic_score가 추가된 FAQ 리스트
    """
    if not faq_candidates:
        return []
    
    # 임베딩 모델 로드
    embedding_model = get_embedding_model()
    
    # 사용자 질문 임베딩 (384차원 벡터)
    question_embedding = embedding_model.encode(
        question, 
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 정규화 (코사인 유사도 계산 최적화)
    )
    
    # 모든 FAQ 질문 임베딩 (배치 처리로 속도 향상)
    faq_questions = [faq['question'] for faq in faq_candidates]
    faq_embeddings = embedding_model.encode(
        faq_questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    # 코사인 유사도 계산 (이미 정규화되어 있으므로 내적 = 코사인 유사도)
    # 결과: (n_faqs,) 형태의 유사도 배열
    similarities = cosine_similarity(
        question_embedding.reshape(1, -1),  # (1, 384)
        faq_embeddings  # (n_faqs, 384)
    )[0]
    
    # 각 FAQ에 semantic_score 추가
    for i, faq in enumerate(faq_candidates):
        faq['semantic_score'] = float(similarities[i])
    
    print(f"[FAQ] 의미론적 검색 완료: 유사도 범위 [{similarities.min():.3f}, {similarities.max():.3f}]")
    return faq_candidates


# ============================================================================
# 3단계: Hybrid Score (키워드 + 의미론적 점수 결합)
# ============================================================================

def calculate_hybrid_score(
    faq_candidates: List[Dict[str, Any]], 
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """
    하이브리드 점수 계산 및 재정렬
    
    최종 점수 = (키워드 점수 × 0.3) + (의미론적 점수 × 0.7)
    
    Args:
        faq_candidates: FAQ 리스트 (keyword_score, semantic_score 포함)
        keyword_weight: 키워드 점수 가중치 (기본값: 0.3)
        semantic_weight: 의미론적 점수 가중치 (기본값: 0.7)
        
    Returns:
        hybrid_score로 정렬된 FAQ 리스트
    """
    for faq in faq_candidates:
        # 하이브리드 점수 계산
        faq['hybrid_score'] = (
            faq.get('keyword_score', 0) * keyword_weight +
            faq.get('semantic_score', 0) * semantic_weight
        )
    
    # 하이브리드 점수 내림차순 정렬
    faq_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    print(f"[FAQ] 하이브리드 점수 계산 완료 (키워드 {keyword_weight} + 의미론적 {semantic_weight})")
    return faq_candidates


# ============================================================================
# 4단계: Reranker (Cross-Encoder로 정확도 재평가)
# ============================================================================

def rerank_faqs(question: str, faq_candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Cross-Encoder 기반 Reranking
    
    질문-FAQ 쌍을 직접 입력받아 관련성 점수 계산
    - 단방향 임베딩보다 정확도 높음
    - 최종 top-k 선정에 사용
    
    Args:
        question: 사용자 질문
        faq_candidates: 하이브리드 점수로 정렬된 FAQ 리스트
        top_k: 최종 반환할 FAQ 개수
        
    Returns:
        rerank_score로 재정렬된 상위 top_k FAQ 리스트
    """
    if not faq_candidates:
        return []
    
    # Reranker 모델 로드
    reranker_model = get_reranker_model()
    
    # 질문-FAQ 쌍 생성 (질문과 각 FAQ의 질문+답변 결합)
    pairs = [
        [question, f"{faq['question']} {faq['answer']}"] 
        for faq in faq_candidates
    ]
    
    # Reranking 점수 계산 (0~1 범위)
    rerank_scores = reranker_model.predict(pairs, show_progress_bar=False)
    
    # 각 FAQ에 rerank_score 추가
    for i, faq in enumerate(faq_candidates):
        faq['rerank_score'] = float(rerank_scores[i])
    
    # Rerank 점수로 재정렬
    faq_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # 상위 top_k만 반환
    top_faqs = faq_candidates[:top_k]
    
    print(f"[FAQ] Reranking 완료: Top-{top_k} 선정")
    for i, faq in enumerate(top_faqs, 1):
        print(f"  {i}. {faq['question'][:30]}... (Rerank: {faq['rerank_score']:.3f})")
    
    return top_faqs


# ============================================================================
# 통합 검색 파이프라인 (메인 함수)
# ============================================================================

def search_faq(question: str, top_k: int = 3) -> str:
    """
    FAQ Hybrid Search 파이프라인
    
    4단계 검색 전략:
    1. Keyword Search: PostgreSQL GIN 인덱스로 빠른 후보군 수집 (top-10)
    2. Semantic Search: 임베딩 기반 의미론적 유사도 계산
    3. Hybrid Score: 키워드(30%) + 의미론적(70%) 점수 결합
    4. Reranker: Cross-Encoder로 최종 정확도 향상 (top-k)
    
    Args:
        question: 사용자의 질문
        top_k: 최종 반환할 FAQ 개수 (기본값: 3)
        
    Returns:
        str: 최상위 FAQ 답변 + 관련 FAQ 목록
        
    Examples:
        >>> search_faq("카드 만들고 싶은데 어떻게 하나요?")
        "[카드 발급]\n\n질문: 카드 발급은 어떻게 신청하나요?\n\n답변: ..."
    """
    
    print(f"\n[FAQ] 검색 시작: '{question}'")
    print("="*80)
    
    try:
        # ====================================================================
        # 1단계: Keyword Search (후보군 수집)
        # ====================================================================
        # - PostgreSQL GIN 인덱스 활용 (빠른 속도)
        # - 키워드 매칭으로 상위 10개 후보 수집
        # - 띄어쓰기 오류 대응 (정규화 + n-gram)
        
        candidate_size = max(top_k * 3, 10)  # 최소 10개 후보 수집
        faq_candidates = keyword_search(question, top_k=candidate_size)
        
        if not faq_candidates:
            return "죄송합니다. 관련된 FAQ를 찾을 수 없습니다. 다른 질문을 해주시거나, 고객센터(1588-0000)로 문의해주세요."
        
        # ====================================================================
        # 2단계: Semantic Search (의미론적 유사도)
        # ====================================================================
        # - 임베딩 모델로 질문과 FAQ를 벡터로 변환
        # - 코사인 유사도 계산 (동의어/유사어 인식)
        # - 예: "카드 만들기" ≈ "카드 발급" (높은 유사도)
        
        faq_candidates = semantic_search(question, faq_candidates)
        
        # ====================================================================
        # 3단계: Hybrid Score (점수 결합)
        # ====================================================================
        # - 키워드 점수 30% + 의미론적 점수 70%
        # - 키워드 정확 매칭과 의미 유사성 모두 고려
        
        faq_candidates = calculate_hybrid_score(
            faq_candidates,
            keyword_weight=0.3,
            semantic_weight=0.7
        )
        
        # ====================================================================
        # 4단계: Reranker (최종 정확도 향상)
        # ====================================================================
        # - Cross-Encoder로 질문-FAQ 쌍 직접 평가
        # - 가장 관련성 높은 top-k 선정
        
        final_faqs = rerank_faqs(question, faq_candidates, top_k=top_k)
        
        if not final_faqs:
            return "죄송합니다. 적절한 FAQ를 찾을 수 없습니다. 고객센터로 문의해주세요."
        
        # ====================================================================
        # 답변 포맷팅
        # ====================================================================
        best_match = final_faqs[0]
        
        # 최상위 FAQ 상세 답변
        answer = f"[{best_match['category_name']}]\n\n"
        answer += f"질문: {best_match['question']}\n\n"
        answer += f"답변: {best_match['answer']}\n"
        
        # 관련 FAQ 목록 (2개 이상인 경우)
        if len(final_faqs) > 1:
            answer += "\n\n[관련 FAQ]\n"
            for idx, faq in enumerate(final_faqs[1:], 1):
                # Rerank 점수 표시 (개발/디버깅용)
                answer += f"{idx}. {faq['question']} (관련도: {faq['rerank_score']:.2f})\n"
        
        print("="*80)
        print(f"[FAQ] 검색 완료: {len(final_faqs)}개 반환\n")
        
        return answer
        
    except Exception as e:
        print(f"[FAQ] 검색 오류: {e}")
        import traceback
        traceback.print_exc()
        return f"검색 중 오류가 발생했습니다: {str(e)}"
