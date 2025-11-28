import httpx
import pandas as pd
from typing import Optional, Tuple, List

from core.config import settings
from Cluster.clustering import predict_cluster


# 클러스터별 추천 카드 매핑 (df_clustering_result.csv 기반)
CLUSTER_CARD_MAPPING = {
    0: ["우리카드 7CORE", "WONDER카드 (할인형)"],
    1: ["카드의정석 EVERY POINT", "우리WON모바일 체크카드"],
    2: ["카드의정석 EVERYDAY", "THE SIMPLE카드"],
    3: ["그랑블루 체크카드", "WE:SH 카드"],
    4: ["우리아이행복카드", "위비온카드"],
    5: ["신세계 우리카드", "WONDER카드 (포인트형)"],
    6: ["ROYAL BLUE MILEAGE", "우리V카드"],
    -1: ["우리카드 7CORE", "카드의정석 EVERY POINT"]  # 노이즈 클러스터 (기본 추천)
}


async def tabular_recommendation(session_id: str) -> Tuple[str, bool, List[str]]:
    """
    MyData 기반 소비패턴 분석 및 카드 추천 함수
    
    Args:
        session_id (str): 사용자 세션 ID
    
    Returns:
        Tuple[str, bool, List[str]]: (답변 메시지, 로그인 필요 여부, 추천 카드 리스트)
    """
    
    # ========================================================================
    # 1-2단계: 세션 ID로 Persona ID 조회 + 로그인 체크
    # ========================================================================
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{settings.BACKEND_SERVER_URL}/api/login/persona_id",
            json={"session_id": str(session_id)}
        )
        persona_id = res.json().get("persona_id", None)
        print(f"[tabular_recommendation] 조회된 페르소나 ID: {persona_id}")
        
        # 로그인되지 않은 경우
        if persona_id is None:
            return (
                "로그인이 필요합니다.",
                True,
                []
            )
    
    # ========================================================================
    # 3단계: DB에서 PersonalData 조회 (TODO: DB 구현 후 활성화)
    # ========================================================================
    # TODO: 백엔드에 다음 API 구현 필요
    # POST /api/mydata/consumption_pattern
    # Body: {"persona_id": int}
    # Response: {소비패턴 12개 컬럼}
    
    # 임시: 더미 데이터 사용 (DB 구현 전)
    # async with httpx.AsyncClient(timeout=30.0) as client:
    #     res = await client.post(
    #         f"{settings.BACKEND_SERVER_URL}/api/mydata/consumption_pattern",
    #         json={"persona_id": persona_id}
    #     )
    #     consumption_data = res.json()
    
    # 더미 데이터 (테스트용 - df_clustering_result.csv의 첫 번째 행)
    consumption_data = {
        '이용건수_신판_R3M': 5,
        '이용금액_신판_R3M': 382923,
        '이용금액_쇼핑': 0,
        '이용금액_요식': 0,
        '이용금액_교통': 0,
        '이용금액_의료': 0,
        '이용금액_납부': 197294,
        '이용금액_교육': 0,
        '이용금액_여유생활': 0,
        '이용금액_사교활동': 0,
        '이용금액_일상생활': 0,
        '이용금액_해외': 0
    }
    
    print(f"[tabular_recommendation] 소비패턴 데이터: {consumption_data}")
    
    # ========================================================================
    # 4단계: clustering.py로 클러스터 예측
    # ========================================================================
    try:
        cluster_id = predict_cluster(consumption_data)
        print(f"[tabular_recommendation] 예측된 클러스터: {cluster_id}")
    except Exception as e:
        print(f"[tabular_recommendation 오류] 클러스터 예측 실패: {e}")
        # 예측 실패 시 기본 추천
        cluster_id = -1
    
    # ========================================================================
    # 5-6단계: 클러스터별 추천 카드 반환
    # ========================================================================
    recommended_cards = CLUSTER_CARD_MAPPING.get(cluster_id, CLUSTER_CARD_MAPPING[-1])
    
    answer = (
        f"고객님의 최근 3개월 소비 패턴을 분석한 결과, "
        f"'{_get_cluster_description(cluster_id)}' 그룹으로 분류되었습니다.\n\n"
        f"이 그룹의 고객님들께서 가장 많이 선택하신 카드는 다음과 같습니다:"
    )
    
    return (
        answer,
        False,  # 로그인 완료
        recommended_cards
    )


def _get_cluster_description(cluster_id: int) -> str:
    """클러스터 ID에 맞는 설명 반환"""
    descriptions = {
        0: "고액 해외 집중형",
        1: "소액 일상 집중형",
        2: "중액 쇼핑 집중형",
        3: "중액 일상 집중형",
        4: "고액 납부 집중형",
        5: "고액 교육 집중형",
        6: "고액 요식 집중형",
        -1: "일반 소비"
    }
    return descriptions.get(cluster_id, "일반 소비")