from FlagEmbedding import BGEM3FlagModel # 임베딩 모델
from qdrant_client import QdrantClient # Qdrant 클라이언트
from qdrant_client.models import (
    VectorParams, Distance, SparseVector, PointStruct, SparseVectorParams
)

from pathlib import Path # 파일 경로 처리용
import json

from core.qdrant_upsert_utils import upsert_docs
from core.config import settings

# 임베딩 및 upsert 대상 json 파일 경로 (정제된 카드 설명 데이터)
INPUT_JSON_BASE = "./data/card_desc_refined_json"
COL = "woori_card_description" # Qdrant에 upsert할 컬렉션 이름

# json데이터를 임베딩할 모델 선언
embedding_model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True,
    device=settings.DEVICE,
    normalize_embeddings=True
)

# Qdrant 클라이언트 선언
qdrant_connection = QdrantClient(
    host=settings.QDRANT_HOST, # Qdrant 서버 IP
    port=settings.QDRANT_PORT  # Qdrant 서버 Port
)

def upsert_card_desc_data():
    global qdrant_connection, embedding_model
    global COL, INPUT_JSON_BASE

    print(f"===== [START] Qdrant '{COL}' 컬렉션에 카드 설명 데이터 upsert =====")

    try:
        # Qdrant 컬렉션이 존재하지 않으면 생성
        if not qdrant_connection.collection_exists(collection_name=COL):
            print(f"[INFO] Qdrant 컬렉션 '{COL}'이 존재하지 않아 새로 생성합니다.")
            qdrant_connection.create_collection(
                collection_name=COL,
                vectors_config=VectorParams(
                    size=1024,               # BAAI/bge-m3 임베딩 벡터 크기
                    distance=Distance.COSINE # 코사인 유사도
                ),
                # 희소 벡터도 지원하도록 설정 (키워드 기반 검색용)
                sparse_vectors_config={"sparse": SparseVectorParams()}
            )
        # 데이터가 없으면 생성
        if qdrant_connection.count(collection_name=COL, exact=True).count == 0:
            print(f"[INFO] Qdrant 컬렉션 '{COL}'이 존재하지만 데이터가 없어 새로 생성합니다.")
            
            # upsert 대상 파일 탐색 및 전처리
            embed_docs = list()
            input_json_dir_path = Path(INPUT_JSON_BASE)
            input_json_file_paths = list(input_json_dir_path.glob("*.json"))
            if not input_json_file_paths: # 해당 경로에 json 파일이 없는 경우
                raise FileNotFoundError(f"입력 JSON 파일이 '{INPUT_JSON_BASE}' 경로에 존재하지 않습니다.")

            # 각 json 파일을 순회 하며 임베딩 및 upsert 준비
            for json_path in input_json_file_paths:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                embed_docs.extend(data["embed_docs"])
            print(f"[INFO] 총 {len(embed_docs)}개의 문서를 찾았습니다.")
            # 문서 임베딩 및 Qdrant에 upsert
            upsert_docs(
                client=qdrant_connection,
                embed_model=embedding_model,
                target_collection=COL,
                embed_docs=embed_docs
            )
    except Exception as e:
        print(f"[ERROR] upsert 중 오류 발생: {e}")
        return