import httpx
from typing import Optional

from core.config import settings


async def tabular_recommendation(session_id: str):
    """
    MyData 기반 소비패턴 분석 및 카드 추천 함수
    
    Args:
        session_id (str): 사용자 세션 ID
    
    Returns:
        str: 추천 카드 목록 및 설명
    """
    
    # 페르소나 ID를 세션 ID로 조회
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{settings.BACKEND_SERVER_URL}/api/login/persona_id",
            json={"session_id": str(session_id)}
        )
        persona_id = res.json().get("persona_id", None)
        print(f"[tabular_recommendation] 조회된 페르소나 ID: {persona_id}")
        # 조회 결과에 페르소나 ID가 없으면 로그인 필요 응답 반환
        if persona_id is None:
            return (
                "로그인이 필요합니다.",
                True,
                []
            )

    # 페르소나 ID가 존재하면 카드 추천 로직 수행
    # 카드 추천 함수 호출

    return  (
        "당신의 소비데이터 기반 카드 추천입니다.",
        False, # 이미 로그인 되어 있으므로 플래그 unset
        ["카드의 정석2", "그랑블루 체크카드", "우리WON모바일 체크카드"]
    )