"""
FAQ 의미 기반 검색 모듈 (2단계 Re-ranking)

이 모듈은 Bi-Encoder + Cross-Encoder 2단계 검색을 사용하여 
사용자 질문과 FAQ 데이터베이스 간의 정확한 매칭을 수행합니다.

검색 프로세스:
    1단계 (Bi-Encoder): Jina v3로 빠른 후보 추출 (top 15)
    2단계 (Cross-Encoder): BGE Reranker로 정밀 재랭킹 (top 3)
    
주요 기능:
    - FAQ 임베딩 캐싱으로 빠른 1단계 검색
    - Cross-Encoder로 정확도 향상
    - 조회수(views) 자동 증가로 인기 FAQ 추적
    
성능:
    - Bi-Encoder: jinaai/jina-embeddings-v3 (다국어)
    - Cross-Encoder: BAAI/bge-reranker-v2-m3 (다국어)
    - 예상 응답시간: ~1초
    - 예상 정확도: 85-90%
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import psycopg2
import torch
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from core.config import settings


# ============================================================================
# 전역 캐시 변수
# ============================================================================

_embedding_model: SentenceTransformer = None  # Jina v3 Bi-Encoder
_cross_encoder: CrossEncoder = None           # BGE Reranker v2-m3
_faq_embeddings_cache: torch.Tensor = None    # FAQ 임베딩 벡터 캐시
_faq_data_cache: List[Dict[str, Any]] = None  # FAQ 메타데이터 캐시


# ============================================================================
# 유틸리티 함수
# ============================================================================

@contextmanager
def get_db_connection():
    """
    PostgreSQL 데이터베이스 연결을 컨텍스트 매니저로 관리합니다.
    
    자동으로 연결을 열고 닫아 리소스 누수를 방지합니다.
    
    Yields:
        psycopg2.connection: PostgreSQL 연결 객체
        
    Raises:
        psycopg2.Error: DB 연결 실패 시
    """
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


# ============================================================================
# 모델 및 데이터 로딩
# ============================================================================

def get_embedding_model() -> SentenceTransformer:
    """
    Jina Embeddings v3 Bi-Encoder 모델을 로드합니다 (Lazy Loading).
    
    첫 호출 시에만 모델을 로드하고, 이후에는 캐시된 모델을 재사용합니다.
    1단계 검색에서 질문과 FAQ 간의 의미적 유사도를 빠르게 계산합니다.
    
    Returns:
        SentenceTransformer: Jina v3 임베딩 모델 (다국어 지원)
        
    Note:
        - 첫 로딩: ~10초 (모델 다운로드 + 초기화)
        - 이후 호출: 즉시 반환 (캐시 사용)
    """
    global _embedding_model
    if _embedding_model is None:
        print("[FAQ] Bi-Encoder 로드 중...")
        _embedding_model = SentenceTransformer(
            'jinaai/jina-embeddings-v3',
            device=settings.DEVICE,
            trust_remote_code=True
        )
        print(f"[FAQ] Bi-Encoder 로드 완료 (Device: {settings.DEVICE})")
    return _embedding_model


def get_cross_encoder() -> CrossEncoder:
    """
    BGE Reranker v2-m3 Cross-Encoder 모델을 로드합니다 (Lazy Loading).
    
    첫 호출 시에만 모델을 로드하고, 이후에는 캐시된 모델을 재사용합니다.
    2단계 재랭킹에서 [질문, FAQ] 쌍의 정밀한 관련도 점수를 계산합니다.
    
    Returns:
        CrossEncoder: BGE Reranker v2-m3 모델 (다국어 지원, 한국어 최적화)
        
    Note:
        - 첫 로딩: ~15초 (모델 다운로드 + 초기화)
        - 이후 호출: 즉시 반환 (캐시 사용)
        - Bi-Encoder보다 느리지만 더 정확한 점수 계산
    """
    global _cross_encoder
    if _cross_encoder is None:
        print("[FAQ] Cross-Encoder 로드 중...")
        _cross_encoder = CrossEncoder(
            'BAAI/bge-reranker-v2-m3',
            device=settings.DEVICE,
            max_length=512
        )
        print(f"[FAQ] Cross-Encoder 로드 완료 (Device: {settings.DEVICE})")
    return _cross_encoder


def load_faq_embeddings() -> Dict[str, Any]:
    """
    FAQ 데이터를 로드하고 임베딩 벡터를 사전 계산하여 캐싱합니다.
    
    첫 호출 시 PostgreSQL에서 모든 FAQ를 로드한 후 Bi-Encoder로 임베딩을 생성하고,
    이후 호출에서는 캐시된 임베딩을 재사용하여 빠른 검색을 지원합니다.
    
    임베딩 생성 전략:
        - FAQ 질문 텍스트를 기본으로 사용
        - 상위 10개 키워드를 추가하여 검색 정확도 향상
        - 정규화(normalize)하여 코사인 유사도 계산 최적화
    
    Returns:
        Dict[str, Any]: {
            'embeddings': torch.Tensor - FAQ 임베딩 벡터 (N x 1024),
            'faqs': List[Dict] - FAQ 메타데이터 (faq_id, question, answer 등)
        }
        
    Note:
        - 첫 실행: ~2-5초 (DB 조회 + 임베딩 생성)
        - 이후 호출: 즉시 반환 (메모리 캐시)
    """
    global _faq_embeddings_cache, _faq_data_cache
    
    # 캐시된 데이터가 있으면 반환
    if _faq_embeddings_cache is not None:
        return {'embeddings': _faq_embeddings_cache, 'faqs': _faq_data_cache}
    
    start = time.time()
    try:
        # DB에서 FAQ 데이터 로드 
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT f.faq_id, f.question, f.answer, 
                           f.normalized_keywords, c.category_name
                    FROM faqs f
                    JOIN faq_categories c ON f.category_id = c.category_id
                    ORDER BY f.faq_id;
                """)
                faqs = [dict(row) for row in cursor.fetchall()]
        
        if not faqs:
            return {'embeddings': None, 'faqs': []}
        
        model = get_embedding_model()
        
        # 질문 + 키워드 결합 텍스트 생성
        texts = []
        for faq in faqs:
            text = faq['question']
            # 상위 10개 키워드만 사용하여 노이즈 감소
            if faq['normalized_keywords']:
                text += ' ' + ' '.join(faq['normalized_keywords'][:10])
            texts.append(text)
        
        # 임베딩 생성 (정규화 포함)
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        elapsed = (time.time() - start) * 1000
        print(f"[FAQ] FAQ 임베딩 완료: {len(faqs)}개 ({elapsed:.1f}ms)")
        
        # 캐시 저장
        _faq_embeddings_cache = embeddings
        _faq_data_cache = faqs
        
        return {'embeddings': embeddings, 'faqs': faqs}
        
    except Exception as e:
        print(f"[FAQ] 임베딩 캐싱 오류: {e}")
        import traceback
        traceback.print_exc()
        return {'embeddings': None, 'faqs': []}


# ============================================================================
# 메인 검색 함수
# ============================================================================

def search_faq(question: str, top_k: int = 3, top_k_retrieval: int = 15) -> Dict[str, Any]:
    """
    사용자 질문에 대해 가장 관련성 높은 FAQ를 2단계 Re-ranking으로 검색합니다.
    
    검색 알고리즘:
        Stage 1 (Bi-Encoder): 
            - Jina v3로 질문 임베딩 생성
            - 전체 FAQ와 코사인 유사도 계산
            - 상위 15개 후보 추출 (~300ms)
            
        Stage 2 (Cross-Encoder):
            - BGE Reranker로 [질문, FAQ] 쌍의 정밀한 관련도 점수 계산
            - 점수 기준 상위 3개 최종 선택 (~700ms)
            
        Stage 3 (조회수 업데이트):
            - 최상위 FAQ의 views 카운트 1 증가
    
    Args:
        question: 사용자의 자연어 질문 (예: "카드 잃어버렸어요")
        top_k: 최종 반환할 FAQ 개수 (기본값: 3)
        top_k_retrieval: Bi-Encoder로 추출할 후보 개수 (기본값: 15)
    
    Returns:
        Dict[str, Any]: {
            'answer': str - 최상위 FAQ 답변 (카테고리 + 질문 + 답변 포함),
            'relatedQuestions': List[str] - 나머지 관련 질문 목록 (최대 2개)
        }
        
    Example:
        >>> search_faq("카드 잃어버렸어요")
        {
            'answer': '[카드관리]\n\n질문: 재발급 신청방법을 알려주세요?\n\n답변: ...',
            'relatedQuestions': ['카드를 일시적으로 정지할수 있나요?', ...]
        }
        
    Performance:
        - 첫 호출: ~30초 (모델 로드 시간 포함)
        - 이후 호출: ~1초 (캐시 사용)
        - 정확도: 85-90% (Cross-Encoder re-ranking)
    """
    print(f"\n[FAQ] 2단계 검색 시작: '{question}'\n" + "=" * 80)
    start_time = time.time()
    
    try:
        # ========== Stage 1: Bi-Encoder 빠른 후보 추출 ==========
        print("[Stage 1] Bi-Encoder 후보 추출 중...")
        stage1_start = time.time()
        
        # 사전 계산된 FAQ 임베딩 로드 (캐시 사용)
        cache = load_faq_embeddings()
        embeddings = cache['embeddings']
        faqs = cache['faqs']
        
        if embeddings is None or len(faqs) == 0:
            return {
                "answer": "FAQ 데이터를 불러올 수 없습니다.",
                "relatedQuestions": []
            }
        
        bi_encoder = get_embedding_model()
        
        # 사용자 질문 임베딩 생성
        q_embed = bi_encoder.encode(
            question,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # 코사인 유사도 계산 및 상위 K개 후보 추출
        similarities = util.cos_sim(q_embed, embeddings)[0]
        top_retrieval_indices = torch.topk(
            similarities, 
            k=min(top_k_retrieval, len(faqs))
        )[1]
        
        # 후보 FAQ 리스트 구성
        candidates = []
        for idx in top_retrieval_indices:
            faq = faqs[idx.item()]
            candidates.append({
                'faq': faq,
                'bi_score': similarities[idx].item()
            })
        
        stage1_elapsed = (time.time() - stage1_start) * 1000
        print(f"[Stage 1] 완료: {len(candidates)}개 후보 추출 ({stage1_elapsed:.1f}ms)")
        
        # ========== Stage 2: Cross-Encoder 정밀 Re-ranking ==========
        print("[Stage 2] Cross-Encoder Re-ranking 중...")
        stage2_start = time.time()
        
        cross_encoder = get_cross_encoder()
        
        # [질문, FAQ 질문] 쌍을 생성하여 정확한 관련도 계산
        pairs = [[question, c['faq']['question']] for c in candidates]
        
        # Cross-Encoder로 각 쌍의 관련도 점수 계산 (0~1 범위)
        cross_scores = cross_encoder.predict(pairs, show_progress_bar=False)
        
        # 점수 기준 내림차순 정렬 후 상위 K개 선택
        sorted_indices = np.argsort(cross_scores)[::-1]
        top_indices = sorted_indices[:min(top_k, len(candidates))]
        
        stage2_elapsed = (time.time() - stage2_start) * 1000
        print(f"[Stage 2] 완료: 최종 {len(top_indices)}개 선택 ({stage2_elapsed:.1f}ms)")
        
        # ========== Stage 3: 결과 구성 및 조회수 업데이트 ==========
        # Cross-Encoder 점수 기준으로 정렬된 최종 FAQ 목록 생성
        top_faqs = []
        for rank_idx, candidate_idx in enumerate(top_indices):
            faq = candidates[candidate_idx]['faq'].copy()
            faq['bi_score'] = candidates[candidate_idx]['bi_score']
            faq['cross_score'] = float(cross_scores[candidate_idx])
            faq['final_rank'] = rank_idx + 1
            top_faqs.append(faq)
        
        # 성능 및 결과 로깅
        total_elapsed = (time.time() - start_time) * 1000
        print(f"\n[FAQ] 총 소요 시간: {total_elapsed:.1f}ms")
        print(f"  - Stage 1 (Bi-Encoder): {stage1_elapsed:.1f}ms")
        print(f"  - Stage 2 (Cross-Encoder): {stage2_elapsed:.1f}ms")
        print("\n[최종 결과]")
        for faq in top_faqs:
            print(f"  #{faq['final_rank']}: {faq['question'][:50]}...")
            print(f"    → Bi: {faq['bi_score']:.3f}, Cross: {faq['cross_score']:.3f}")
        print("=" * 80)
        
        # 최상위 FAQ 선택
        best = top_faqs[0]
        
        # 조회수(views) 카운트 1 증가 (인기 FAQ 추적용)
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE faqs SET views = views + 1 WHERE faq_id = %s;",
                        (best['faq_id'],)
                    )
                    conn.commit()
                    print(f"[FAQ] views 증가: faq_id={best['faq_id']}")
        except Exception as e:
            # views 업데이트 실패 시에도 검색 결과는 정상 반환
            print(f"[FAQ] views 업데이트 오류: {e}")
        
        return {
            "answer": f"[{best['category_name']}]\n\n질문: {best['question']}\n\n답변: {best['answer']}",
            "relatedQuestions": [faq['question'] for faq in top_faqs[1:]]
        }
        
    except Exception as e:
        print(f"[FAQ] 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": f"검색 중 오류 발생: {e}",
            "relatedQuestions": []
        }
