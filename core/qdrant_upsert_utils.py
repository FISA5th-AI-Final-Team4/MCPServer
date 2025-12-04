from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, PointStruct

import numpy as np

from typing import List, Dict, Tuple
from uuid import uuid5, NAMESPACE_DNS


def _to_list(x):
    """ numpy.ndarray를 리스트로 변환하는 헬퍼 함수 """
    return x.tolist() if isinstance(x, np.ndarray) else x

def _extract_sparse(out: dict) -> SparseVector:
    """ 희소 벡터 추출 헬퍼 함수 """
    # 희소 벡터 정보가 있는지 확인 후 SparseVector 인스턴스 반환
    if "sparse_vecs" in out and out["sparse_vecs"]:
        sp = out["sparse_vecs"][0]
        return SparseVector(indices=sp["indices"], values=sp["values"])
    
    # 희소 벡터가 없는 경우 lexical_weights 이용
    if "lexical_weights" in out and out["lexical_weights"]:
        lw = out["lexical_weights"][0]
        # 해시 충돌 방지: dict로 중복 인덱스 제거 (같은 인덱스는 값 합산)
        index_map = {}
        for tok, w in lw.items():
            idx = abs(hash(tok)) % (2**20)
            index_map[idx] = index_map.get(idx, 0.0) + w
        
        # 정렬 및 리스트 변환
        items = sorted(index_map.items())
        if items:
            indices, values = zip(*items)
            return SparseVector(indices=list(indices), values=list(values))
    
    # 희소 벡터 정보가 전혀 없는 경우 빈 SparseVector 반환
    return SparseVector(indices=[], values=[])

def encode_doc(embed_model: BGEM3FlagModel, text: str) -> Tuple[List[float], SparseVector]:
    """
    Args:
        - embed_model: 임베딩 모델 인스턴스
        - text: 임베딩할 텍스트 문자열
    Returns:
        - dense: 밀집 벡터 리스트
        - sparse: 희소 벡터 SparseVector 인스턴스
    주어진 텍스트 문자열을 임베딩 모델을 이용하여 밀집 벡터와 희소 벡터로 변환합니다.
    """
    # 임베딩 수행
    out = embed_model.encode([text], return_dense=True, return_sparse=True)

    # 결과 파싱
    dense = _to_list(out["dense_vecs"][0]) # 밀집 벡터 리스트로 변환
    sparse = _extract_sparse(out) # 희소 벡터 추출

    return dense, sparse

def build_text(doc: Dict) -> str:
    """
    Args:
        - doc: 카드 설명 문서
    Returns:
        - str: 임베딩할 텍스트 문자열
    카드 설명 문서 임베딩하기 전 json 데이터를 파싱하여 텍스트 문자열로 변환합니다.
    """
    # 문서 헤더 정보 구성
    head = f"[{doc['card_id']}] 경로: {doc['tag_major']} > {doc.get('tag_middle')} > {doc.get('tag_minor')} | gran: {('coarse' if '|coarse' in doc['doc_id'] else 'fine')}"

    # 텍스트 변환 결과 반환
    return f"{head}\n요약: {doc['text_dense']}\n핵심키워드: {doc['text_sparse']}"

def upsert_docs(
    client: QdrantClient,
    embed_model: BGEM3FlagModel,
    target_collection:str,
    embed_docs: List[Dict]
) -> None:
    """
    Args:
        - client: QdrantClient 인스턴스
        - target_collection: Qdrant에 데이터를 적재할 컬렉션 이름
        - embed_docs: 임베딩할 문서들의 리스트
    
    JSON 리스트를 임베딩하여 Qdrant DB에 적재합니다.
    """
    points=[] # Qdrant에 적재할 PointStruct(임베딩 결과물) 리스트

    # 각 문서를 순회하며 임베딩 및 PointStruct 생성
    for d in embed_docs:
        text = build_text(d) # json 형태의 문서를 텍스트(string) 형태로 변환
        dense, sparse = encode_doc(embed_model, text) # 임베딩 수행
        # 메타 데이터 페이로드 구성
        payload = {
            "card_id": d["card_id"], "doc_id": d["doc_id"],
            "tag_major": d["tag_major"], "tag_middle": d.get("tag_middle"),
            "tag_minor": d.get("tag_minor"), "section_canonical": d["section_canonical"],
            "granularity": "coarse" if "|coarse" in d["doc_id"] else "fine",
            "has_numbers": any(ch.isdigit() for ch in d["text_dense"]),
            "preview": d["text_dense"][:160],
            "full_text": text # ⬅️ [중요] full_text 저장
        }
        points.append(PointStruct(
            id=str(uuid5(NAMESPACE_DNS, d["doc_id"])),
            vector={"": dense, "sparse": sparse},
            payload=payload
        ))
    print(f"Upserting {len(points)} points to '{target_collection}'...")
    client.upsert(collection_name=target_collection, points=points, wait=True)
    print("Upsert complete.")