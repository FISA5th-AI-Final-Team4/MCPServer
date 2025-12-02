"""
GraphRAG 기반 카드 추천 모듈

nano-graphrag를 활용하여 카드 상품 Knowledge Graph에서
사용자 질문에 맞는 카드를 추천합니다.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
import nano_graphrag.prompt as nano_prompts  # nano-graphrag 내장 프롬프트
from langchain_openai import ChatOpenAI

from core.config import settings
# 기존 카드 설명 RAG와 동일한 임베딩 모델 사용 (싱글톤)
from core.qdrant_upsert import embedding_model as bge_model

# 프롬프트 파일 로드 (Card_recommend/prompt.py)
PROMPT_PATH = Path(__file__).parent.parent.parent / "Card_recommend" / "prompt.py"

# 카드 별칭 맵 로드 (카드명 추출용)
ALIAS_MAP_PATH = Path(__file__).parent.parent.parent / "data" / "alias_map.json"
with open(ALIAS_MAP_PATH, "r", encoding="utf-8") as f:
    CARD_ID_TO_ALIASES = json.load(f)

# ============================================================================
# nano-graphrag 프롬프트 오버라이드 (Card_recommend/prompt.py 사용)
# ============================================================================
# prompt.py 파일을 동적으로 로드하여 nano-graphrag 내장 프롬프트 교체
if PROMPT_PATH.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("card_prompts", PROMPT_PATH)
    card_prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(card_prompts)
    
    # nano-graphrag의 PROMPTS 딕셔너리 오버라이드
    if hasattr(card_prompts, 'PROMPTS'):
        for key in ['local_rag_response', 'global_reduce_rag_response', 'global_map_rag_points']:
            if key in card_prompts.PROMPTS:
                nano_prompts.PROMPTS[key] = card_prompts.PROMPTS[key]
                logging.info(f"[GraphRAG] 프롬프트 오버라이드: {key}")

# ============================================================================
# 로깅 설정
# ============================================================================
logging.basicConfig(level=logging.INFO)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ============================================================================
# 전역 설정
# ============================================================================

# GraphRAG 인덱스 디렉토리 (Card_recommend/card_graphrag_db_real)
GRAPHRAG_WORKING_DIR = Path(__file__).parent.parent.parent / "Card_recommend" / "card_graphrag_db_real"

# 디버그 설정
DEBUG_PRINT_AUGMENTED_QUERY = False
DEBUG_PRINT_LLM_PROMPTS = False

# 대화 히스토리 관리 (세션별)
SESSION_CONVERSATION_HISTORY: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_TURNS = 2

# 토큰 사용량 추적
GLOBAL_TOKEN_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}
TOKEN_USAGE_TRACKING = False

# ============================================================================
# BGE-M3 임베딩 함수 (nano-graphrag용)
# 기존 qdrant_upsert.py의 embedding_model (bge_model) 재사용
# ============================================================================

@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def bge_embedding(texts: List[str]) -> np.ndarray:
    """
    nano-graphrag용 BGE-M3 임베딩 함수
    
    Args:
        texts: 임베딩할 텍스트 리스트
    
    Returns:
        (N, 1024) 형태의 numpy 배열
    """
    out = bge_model.encode(
        texts,
        batch_size=32,
        max_length=8192
    )
    dense_vecs = out["dense_vecs"]
    return np.array(dense_vecs, dtype=np.float32)


# ============================================================================
# JSON 복구 유틸리티
# ============================================================================

def add_json_safety_instruction(system_prompt: Optional[str]) -> Optional[str]:
    """JSON 출력 시 파싱 에러 방지 규칙 추가"""
    if not system_prompt:
        return system_prompt

    lower = system_prompt.lower()
    if "json" not in lower:
        return system_prompt

    extra = (
        "\n\n[IMPORTANT JSON OUTPUT RULES]\n"
        "- You MUST return syntactically valid JSON that can be parsed by Python json.loads.\n"
        "- Never put raw double quote characters inside JSON string values; they must always be escaped as \\\".\n"
        "- Do not add any text before or after the JSON object."
    )
    return system_prompt + extra


def sanitize_json_like_string(text: str) -> str:
    """LLM이 만든 깨진 JSON을 최대한 복구"""
    if not isinstance(text, str):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return text

    s = text[start:end + 1].replace("\r\n", "\n")
    lines = s.split("\n")

    key_re_with_quotes = re.compile(r'^(\s*)"(?P<key>[^"]+)"\s*:\s*(?P<value>.*)$')
    key_re_no_quotes = re.compile(r'^(\s*)(?P<key>[A-Za-z0-9_가-힣]+)\s*:\s*(?P<value>.*)$')
    number_re = re.compile(r'-?\d+(\.\d+)?$')

    def is_number(tok: str) -> bool:
        return bool(number_re.fullmatch(tok))

    new_lines: List[str] = []

    for line in lines:
        m = key_re_with_quotes.match(line) or key_re_no_quotes.match(line)
        if not m:
            new_lines.append(line)
            continue

        indent = m.group(1)
        key = m.group("key")
        value = m.group("value").rstrip("\n")

        trailing_comma = ""
        if value.endswith(","):
            value_core = value[:-1].rstrip()
            trailing_comma = ","
        else:
            value_core = value.rstrip()

        leading_ws_len = len(value_core) - len(value_core.lstrip())
        leading_ws = value_core[:leading_ws_len]
        v = value_core[leading_ws_len:]

        if not v:
            new_v = leading_ws + '""'
        elif v in ("true", "false", "null") or is_number(v):
            new_v = leading_ws + v
        elif v.startswith("{") or v.startswith("["):
            new_v = leading_ws + v
        elif v.startswith('"') and v.endswith('"'):
            inner = v[1:-1]
            json_str = json.dumps(inner, ensure_ascii=False)
            new_v = leading_ws + json_str
        else:
            json_str = json.dumps(v, ensure_ascii=False)
            new_v = leading_ws + json_str

        new_line = f'{indent}"{key}": {new_v}{trailing_comma}'
        new_lines.append(new_line)

    return "\n".join(new_lines)


def normalize_json_if_possible(text: str) -> str:
    """JSON 정규화 시도"""
    if not isinstance(text, str):
        return text

    stripped = text.strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            return stripped
        except Exception:
            pass

        sanitized = sanitize_json_like_string(stripped)
        try:
            json.loads(sanitized)
            return sanitized
        except Exception:
            return stripped

    return text


# ============================================================================
# LLM 래퍼 함수
# ============================================================================

async def custom_llm_complete_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict]] = None,
    *,
    model_name: str,
    **kwargs,
) -> str:
    """
    nano-graphrag용 LLM 완성 함수
    
    Args:
        prompt: 사용자 프롬프트
        system_prompt: 시스템 프롬프트
        history_messages: 대화 히스토리
        model_name: 사용할 모델명
    
    Returns:
        LLM 응답 텍스트
    """
    global GLOBAL_TOKEN_USAGE, TOKEN_USAGE_TRACKING

    if history_messages is None:
        history_messages = []

    system_prompt = add_json_safety_instruction(system_prompt)

    if DEBUG_PRINT_LLM_PROMPTS:
        logger.info(f"\n[GraphRAG LLM] model={model_name}")
        if system_prompt:
            logger.info(f"[SYSTEM] {system_prompt[:300]}...")
        logger.info(f"[USER] {prompt[:300]}...")

    # 히스토리 텍스트로 변환
    history_text_parts = []
    for m in history_messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if content:
            history_text_parts.append(f"[{role}]: {content}")
    history_text = "\n".join(history_text_parts)

    # 전체 프롬프트 조합
    full_prompt_parts = []
    if system_prompt:
        full_prompt_parts.append(system_prompt.strip())
    if history_text:
        full_prompt_parts.append(history_text.strip())
    full_prompt_parts.append(prompt.strip())
    full_prompt = "\n\n".join(full_prompt_parts)

    # LLM 호출
    llm = ChatOpenAI(model=model_name, api_key=settings.OPENAI_API_KEY)
    
    allowed_keys = {"temperature", "max_tokens", "top_p"}
    llm_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

    response = await llm.ainvoke(full_prompt, **llm_kwargs)

    content = response.content or ""
    content = normalize_json_if_possible(content)

    # 토큰 사용량 집계
    usage = getattr(response, "usage_metadata", None)
    if usage is None and hasattr(response, "response_metadata"):
        token_usage = (response.response_metadata or {}).get("token_usage")
        if token_usage:
            usage = {
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
            }

    if usage and TOKEN_USAGE_TRACKING:
        GLOBAL_TOKEN_USAGE["input_tokens"] += usage.get("input_tokens", 0)
        GLOBAL_TOKEN_USAGE["output_tokens"] += usage.get("output_tokens", 0)
        GLOBAL_TOKEN_USAGE["total_tokens"] += usage.get("total_tokens", 0)

    return content


# ============================================================================
# 모델 함수 정의 (nano-graphrag용)
# ============================================================================

async def BEST_MODEL_INFER(prompt, system_prompt=None, history_messages=None, **kwargs):
    """최종 답변 생성용 모델 (GPT-5-mini)"""
    return await custom_llm_complete_async(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        model_name="gpt-5-mini",
        **kwargs,
    )


async def CHEAP_MODEL_INFER(prompt, system_prompt=None, history_messages=None, **kwargs):
    """커뮤니티 관련성 판단용 모델 (GPT-5-nano)"""
    return await custom_llm_complete_async(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        model_name="gpt-5-nano",
        **kwargs,
    )


# ============================================================================
# GraphRAG 인스턴스 관리
# ============================================================================

_graphrag_instance = None

def get_graphrag_instance() -> GraphRAG:
    """GraphRAG 인스턴스 싱글톤 로드"""
    global _graphrag_instance
    
    if _graphrag_instance is None:
        if not GRAPHRAG_WORKING_DIR.exists():
            raise FileNotFoundError(
                f"GraphRAG 인덱스 디렉토리를 찾을 수 없습니다: {GRAPHRAG_WORKING_DIR}\n"
                "먼저 인덱싱을 수행해주세요."
            )
        
        logger.info(f"[GraphRAG] Loading from {GRAPHRAG_WORKING_DIR}")
        
        _graphrag_instance = GraphRAG(
            working_dir=str(GRAPHRAG_WORKING_DIR),
            best_model_func=BEST_MODEL_INFER,
            cheap_model_func=CHEAP_MODEL_INFER,
            embedding_func=bge_embedding,
            embedding_batch_num=16,
            embedding_func_max_async=4,
            enable_llm_cache=False,
        )
        
        logger.info("[GraphRAG] Instance loaded successfully.")
    
    return _graphrag_instance


# ============================================================================
# 메인 추천 함수
# ============================================================================

def _extract_card_ids_from_answer(answer: str) -> List[str]:
    """
    GraphRAG 답변에서 TOP3 추천 카드명(카드 ID) 추출
    
    "N순위 : 카드명" 형식에서 카드명을 추출하고,
    alias_map.json의 key(카드 ID)와 직접 매칭합니다.
    """
    card_ids = []
    seen = set()  # 중복 방지
    
    # "N순위 : 카드명" 패턴 매칭 (1순위, 2순위, 3순위)
    pattern = r'([1-3])순위\s*[:：]\s*(.+?)(?:\n|$)'
    matches = re.findall(pattern, answer)
    
    for rank, card_name in matches:
        card_name = card_name.strip()
        if not card_name:
            continue
        
        # alias_map의 key(카드 ID)와 직접 매칭
        matched_card_id = None
        for card_id in CARD_ID_TO_ALIASES.keys():
            if card_id in card_name:
                matched_card_id = card_id
                break
        
        if matched_card_id and matched_card_id not in seen:
            card_ids.append(matched_card_id)
            seen.add(matched_card_id)
    
    return card_ids  # 최대 3개


async def get_card_recommendation(
    query: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    GraphRAG 기반 카드 추천 수행
    
    Args:
        query: 사용자 질문 (예: "편의점 할인 많은 카드 추천해줘")
        session_id: 세션 ID (대화 히스토리 관리용)
    
    Returns:
        Dict containing:
            - answer: 카드 추천 답변
            - card_ids: 추천된 카드명 리스트
    
    Note:
        mode는 항상 "global"로 고정 (커뮤니티 리포트 기반 전체 조망)
    """
    global TOKEN_USAGE_TRACKING, GLOBAL_TOKEN_USAGE
    
    # mode 항상 global 고정
    mode = "global"
    
    logger.info(f"[GraphRAG] 카드 추천 요청: {query[:50]}... (mode={mode})")
    
    # 토큰 추적 초기화
    TOKEN_USAGE_TRACKING = True
    GLOBAL_TOKEN_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    try:
        # 대화 히스토리 가져오기
        if session_id:
            conversation_history = SESSION_CONVERSATION_HISTORY.get(session_id, [])
        else:
            conversation_history = []
        
        # 히스토리가 있으면 쿼리 확장
        if conversation_history:
            history_snippets = []
            for turn in conversation_history[-MAX_HISTORY_TURNS:]:
                prev_q = turn.get("user", "")
                if prev_q:
                    history_snippets.append(f"- {prev_q}")
            
            history_block = "\n".join(history_snippets)
            
            augmented_query = (
                "아래는 사용자가 지금까지 했던 카드 상담 관련 질문 목록이다.\n"
                "이전 질문들의 맥락을 참고하되, 맨 마지막에 주어진 '현재 질문'에 가장 잘 맞는 카드 추천/설명을 우선해서 답변하라.\n\n"
                f"[이전 질문 목록]\n{history_block}\n\n"
                f"[현재 질문]\n{query}"
            )
        else:
            augmented_query = query
        
        if DEBUG_PRINT_AUGMENTED_QUERY:
            logger.info(f"[GraphRAG] Augmented query: {augmented_query[:500]}...")
        
        # GraphRAG 인스턴스 로드 및 쿼리 실행
        graph_func = get_graphrag_instance()
        
        result = await graph_func.aquery(
            augmented_query,
            param=QueryParam(
                mode=mode,
                global_max_consider_community=64,
            ),
        )
        
        # 응답 텍스트 추출
        if isinstance(result, dict):
            answer_text = result.get("answer", str(result))
        else:
            answer_text = str(result)
        
        # 답변에서 카드명 추출
        card_list = _extract_card_ids_from_answer(answer_text)
        
        # 히스토리에 추가
        if session_id:
            if session_id not in SESSION_CONVERSATION_HISTORY:
                SESSION_CONVERSATION_HISTORY[session_id] = []
            
            SESSION_CONVERSATION_HISTORY[session_id].append({
                "user": query,
                "answer": answer_text
            })
            
            # 최대 히스토리 유지
            if len(SESSION_CONVERSATION_HISTORY[session_id]) > MAX_HISTORY_TURNS * 2:
                SESSION_CONVERSATION_HISTORY[session_id] = \
                    SESSION_CONVERSATION_HISTORY[session_id][-MAX_HISTORY_TURNS * 2:]
        
        TOKEN_USAGE_TRACKING = False
        
        # 토큰 사용량 로그 출력
        logger.info(f"[GraphRAG] 토큰 사용량 - input: {GLOBAL_TOKEN_USAGE['input_tokens']}, output: {GLOBAL_TOKEN_USAGE['output_tokens']}, total: {GLOBAL_TOKEN_USAGE['total_tokens']}")
        logger.info(f"[GraphRAG] 카드 추천 완료 - 추천 카드: {card_list}")
        
        return {
            "answer": answer_text,
            "card_list": card_list
        }
        
    except Exception as e:
        TOKEN_USAGE_TRACKING = False
        logger.error(f"[GraphRAG] 카드 추천 오류: {e}")
        raise


def clear_session_history(session_id: str) -> None:
    """세션의 대화 히스토리 삭제"""
    if session_id in SESSION_CONVERSATION_HISTORY:
        del SESSION_CONVERSATION_HISTORY[session_id]
        logger.info(f"[GraphRAG] Session history cleared: {session_id}")
