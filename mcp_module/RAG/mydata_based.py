import httpx
import psycopg2
from typing import Optional, Tuple, List, Dict

from core.config import settings
from Cluster.clustering import predict_cluster


# 클러스터별 소비 패턴 설명
CLUSTER_DESCRIPTIONS = {
    0: "균형 소비형 (다양한 카테고리 고른 지출)",
    1: "미니멀 소비형 (소액 필수 지출 위주)",
    2: "해외 활동형 (해외 결제 및 쇼핑 중심)",
    3: "실속 소비형 (일상 및 쇼핑 집중)",
    4: "납부 중심형 (공과금/납부 위주)",
    5: "교육 투자형 (교육비 지출 집중)",
    6: "프리미엄 라이프형 (고액 다양한 소비)",
    -1: "일반 소비형"
}


def _get_mydata_db_connection():
    """mydata 데이터베이스 연결"""
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database="mydata",
        user=settings.POSTGRES_USER,
        password="mcp_password"
    )


def _get_consumption_data(persona_id: int) -> Optional[Dict]:
    """
    persona_consumption_data 테이블에서 소비 패턴 조회
    
    Returns:
        Dict: 클러스터링 모델 입력용 12개 피처
    """
    try:
        conn = _get_mydata_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                usage_count_r3m,
                usage_amount_r3m,
                amount_shopping,
                amount_food,
                amount_transport,
                amount_medical,
                amount_payment,
                amount_education,
                amount_leisure,
                amount_social,
                amount_daily,
                amount_overseas
            FROM persona_consumption_data
            WHERE persona_id = %s
        """, (persona_id,))
        
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if row is None:
            return None
        
        # 클러스터링 모델 입력 형식으로 변환
        return {
            '이용건수_신판_R3M': row[0],
            '이용금액_신판_R3M': row[1],
            '이용금액_쇼핑': row[2],
            '이용금액_요식': row[3],
            '이용금액_교통': row[4],
            '이용금액_의료': row[5],
            '이용금액_납부': row[6],
            '이용금액_교육': row[7],
            '이용금액_여유생활': row[8],
            '이용금액_사교활동': row[9],
            '이용금액_일상생활': row[10],
            '이용금액_해외': row[11]
        }
        
    except Exception as e:
        print(f"[DB 오류] 소비 데이터 조회 실패: {e}")
        return None


def _get_recommended_cards(cluster_id: int) -> Tuple[List[str], str]:
    """
    클러스터별 추천 카드 조회
    - cluster_recommended_cards: 추천 카드 2개
    - user_card_usage: 해당 클러스터에서 가장 많이 사용되는 카드 1개
    
    Returns:
        Tuple[List[str], str]: (추천 카드 리스트, 주류 사용 카드)
    """
    try:
        conn = _get_mydata_db_connection()
        cur = conn.cursor()
        
        # 1. 추천 카드 2개 조회
        cur.execute("""
            SELECT recommended_card_1, recommended_card_2
            FROM cluster_recommended_cards
            WHERE cluster_id = %s
        """, (cluster_id,))
        
        rec_row = cur.fetchone()
        recommended_cards = []
        if rec_row:
            recommended_cards = [rec_row[0], rec_row[1]]
        
        # 2. 해당 클러스터에서 가장 많이 사용되는 카드 1개 조회
        cur.execute("""
            SELECT primary_card_id, COUNT(*) as cnt
            FROM user_card_usage
            WHERE cluster_id = %s
            GROUP BY primary_card_id
            ORDER BY cnt DESC
            LIMIT 1
        """, (cluster_id,))
        
        usage_row = cur.fetchone()
        popular_card = usage_row[0] if usage_row else None
        
        cur.close()
        conn.close()
        
        return recommended_cards, popular_card
        
    except Exception as e:
        print(f"[DB 오류] 추천 카드 조회 실패: {e}")
        return [], None


async def tabular_recommendation(session_id: str) -> Tuple[str, bool, List[str]]:
    """
    MyData 기반 소비패턴 분석 및 카드 추천 함수
    
    Args:
        session_id (str): 사용자 세션 ID
    
    Returns:
        Tuple[str, bool, List[str]]: (답변 메시지, 로그인 필요 여부, 추천 카드 리스트)
    """
    
    # ========================================================================
    # 1단계: 세션 ID로 Persona ID 조회
    # ========================================================================
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{settings.BACKEND_SERVER_URL}/api/login/persona_id",
            json={"session_id": str(session_id)}
        )
        try:
            persona_data = res.json()
        except ValueError as exc:
            raise RuntimeError("페르소나 조회 응답이 JSON 형식이 아닙니다.") from exc
        if not isinstance(persona_data, dict):
            persona_data = {}
        persona_id = persona_data.get("persona_id", None)
        print(f"[tabular_recommendation] 조회된 페르소나 ID: {persona_id}")
    
    # ========================================================================
    # 2단계: 로그인 체크
    # ========================================================================
    if persona_id is None:
        return (
            "로그인이 필요합니다. 로그인 후 다시 시도해주세요.",
            True,
            []
        )
    
    # ========================================================================
    # 3단계: DB에서 소비 패턴 데이터 조회
    # ========================================================================
    consumption_data = _get_consumption_data(persona_id)
    
    if consumption_data is None:
        return (
            "소비 데이터가 없습니다. 마이데이터 연동 후 이용해주세요.",
            False,
            []
        )
    
    print(f"[tabular_recommendation] 소비패턴 데이터: {consumption_data}")
    
    # ========================================================================
    # 4단계: 클러스터링 모델로 예측
    # ========================================================================
    try:
        cluster_id = predict_cluster(consumption_data)
        print(f"[tabular_recommendation] 예측된 클러스터: {cluster_id}")
    except Exception as e:
        print(f"[tabular_recommendation 오류] 클러스터 예측 실패: {e}")
        cluster_id = -1
    
    # ========================================================================
    # 5단계: 추천 카드 및 주류 카드 조회
    # ========================================================================
    recommended_cards, popular_card = _get_recommended_cards(cluster_id)
    
    # 카드 리스트 구성 (추천 카드 2개 + 주류 카드 1개)
    card_list = recommended_cards.copy()
    if popular_card and popular_card not in card_list:
        card_list.append(popular_card)
    
    # ========================================================================
    # 6단계: 응답 생성
    # ========================================================================
    cluster_desc = CLUSTER_DESCRIPTIONS.get(cluster_id, "일반 소비형")
    
    # 소비 패턴 요약 생성
    total_amount = consumption_data['이용금액_신판_R3M']
    top_categories = _get_top_categories(consumption_data)
    
    answer = f"""📊 **고객님의 소비 패턴 분석 결과**

🏷️ **소비 유형**: {cluster_desc}
💰 **최근 3개월 총 이용금액**: {total_amount:,}원
📈 **주요 소비 카테고리**: {', '.join(top_categories)}

---

💳 **고객님께 추천드리는 카드**:
"""
    
    for i, card in enumerate(card_list, 1):
        if i <= 2:
            answer += f"  {i}. {card} (맞춤 추천)\n"
        else:
            answer += f"  {i}. {card} (같은 유형 고객 인기 카드)\n"
    
    return (answer, False, card_list)


def _get_top_categories(consumption_data: Dict) -> List[str]:
    """소비 데이터에서 상위 3개 카테고리 추출"""
    category_mapping = {
        '이용금액_쇼핑': '쇼핑',
        '이용금액_요식': '요식',
        '이용금액_교통': '교통',
        '이용금액_의료': '의료',
        '이용금액_납부': '납부',
        '이용금액_교육': '교육',
        '이용금액_여유생활': '여유생활',
        '이용금액_사교활동': '사교활동',
        '이용금액_일상생활': '일상생활',
        '이용금액_해외': '해외'
    }
    
    # 금액 기준 정렬
    sorted_categories = sorted(
        [(k, v) for k, v in consumption_data.items() if k in category_mapping],
        key=lambda x: x[1],
        reverse=True
    )
    
    # 상위 3개 (0원 제외)
    top = [category_mapping[k] for k, v in sorted_categories[:3] if v > 0]
    
    return top if top else ['일반']