# 챗봇의 정석 MCP Server


| 소개 | 관련 링크 |
| ------ | ------ |
| 챗봇의 정석의 핵심 기능을 담당하는 **MCP 툴 서버 원격 저장소**입니다.<br>GraphRAG, 벡터 DB, PostgreSQL 기반 QnA, 군집화 모델 등을 하나의 MCP 인터페이스로 묶어,<br> LLM 서버의 **AI 에이전트**를 통해 필요한 툴을 선택적으로 호출할 수 있도록 제공합니다.<br>사용가능한 MCP 툴은 카드 혜택 설명/비교, 혜택 기반 카드 추천,<br> 개인 소비 패턴 기반 추천, FAQ·금융 용어 QnA 입니다. | 🔗[챗봇의 정석](https://github.com/FISA5th-AI-Final-Team4) <br>🔗[FrontEnd](https://github.com/FISA5th-AI-Final-Team4/FrontEnd) <br>🔗[BackEnd](https://github.com/FISA5th-AI-Final-Team4/BackEnd) <br>🔗[LLM서버](https://github.com/FISA5th-AI-Final-Team4/LLMServer) <br>🔗[DB서버](https://github.com/FISA5th-AI-Final-Team4/LocalDbSetup) |



## 🛠️ MCP 도구 목록
| 도구 이름               | 설명                                      | 경로                       |
|----------------------|-----------------------------------------|-----------------------------|
| get_card_description      | 카드 혜택 설명·비교 RAG 툴             | `/mcp/card-description`      |
| get_card_recommendation   | GraphRAG 기반 혜택 기반 카드 추천 툴   | `/mcp/card-recommendation`   |
| query_faq_database        | 카드 상품 및 가입 관련 FAQ 툴          | `/mcp/faq-query`             |
|  query_term_database      | 금융 용어 QnA 유사도 검색 툴           | `/mcp/term-query`            |
| consumption_recommend     | 개인 소비 데이터 기반 카드 추천 툴      | `/mcp/consumption_recommend` |

## 🗂️ MCP 툴 내부 로직 다이어그램

<details>
<summary><strong>get_card_description</strong></summary>

<img width="1200" alt="image" src="https://github.com/user-attachments/assets/8b80f269-f236-45d4-9426-9ccf15284b13" />

</details> 

<details>
  <summary><strong>query_faq_database</strong></summary>

  <img width="923" alt="image" src="https://github.com/user-attachments/assets/0594469c-317b-4795-9e91-93827179c1ed" />
  
</details>


<details>
  <summary><strong>query_term_database</strong></summary>

  <img width="582" alt="image" src="https://github.com/user-attachments/assets/38be2643-b792-4b60-a763-580641003542" />
</details>



<details>
  <summary><strong>consumption_recommend</strong></summary>

  <img width="2209" alt="image" src="https://github.com/user-attachments/assets/86de9a10-5f43-4039-a102-45167c4042d7" />
</details>




## ⚒️ 기술 스택
- FastAPI, pydantic, MCP, PostgreSQL, Qdrant

  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white"/> <img src="https://img.shields.io/badge/pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white"> <img src="https://img.shields.io/badge/MCP-1C3C3C?style=for-the-badge&logo=modelcontextprotocol&logoColor=white"/> <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=PostgreSQL&logoColor=white"/> <img src="https://img.shields.io/badge/Qdrant-DC244C?style=for-the-badge">

- langchain-core, Ollama, scikit-learn

  <img src="https://img.shields.io/badge/langchain-core-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/> <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=Ollama&logoColor=white"/> <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">

- Docker + uvicorn + EC2

    <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white"> <img src="https://img.shields.io/badge/uvicorn-E4CCFF?style=for-the-badge"> <img src="https://img.shields.io/badge/AWS EC2-F58536?style=for-the-badge"/>

## 📁 프로젝트 구조
```bash
.
├── Card_recommend                                  # GraphRAG 기반 카드 추천 관련 작업 디렉토리
│   ├── card_graphrag_db_real                       # nano-graphrag 인덱스 및 그래프 데이터
│   │   └── graph_chunk_entity_relation.graphml     # 카드 설명서에서 추출한 엔티티·관계 그래프
│   └── prompt.py                                   # GraphRAG용 카드 추천/응답 프롬프트 정의
├── Cluster                                         # 개인 소비 데이터 군집화(클러스터링) 관련 모듈
│   ├── clustering.py                               # HDBSCAN + UMAP 기반 클러스터링 스크립트
│   ├── hdbscan_model.pkl                           # 학습된 HDBSCAN 군집화 모델
│   └── umap_reducer_5d.pkl                         # 차원 축소(UMAP 5차원) 모델
├── Dockerfile                                      # MCP 서버 컨테이너 빌드 설정
├── api                                             # FastAPI 라우터 및 MCP 툴 HTTP API 레이어
│   ├── router.py                                   # 공통 APIRouter 설정 및 엔드포인트 등록
│   └── routes                                      # 개별 툴/엔드포인트 구현
│       ├── tools.py                                # LLM 서버에서 호출할 MCP 툴 HTTP 엔드포인트
│       └── weather.py                              # 날씨 툴 예시 엔드포인트
├── core                                            # 공통 설정 및 벡터 DB 연동 모듈
│   ├── config.py                                   # 환경 변수/설정 로딩 (Qdrant, DB 등)
│   ├── qdrant_upsert.py                            # 카드 설명서 벡터 임베딩 Qdrant 업서트 로직
│   ├── qdrant_upsert_utils.py                      # Qdrant 업서트/전처리 유틸 함수
│   └── setup.py                                    # 애플리케이션 초기화 및 의존성 설정
├── data                                            # RAG 및 QnA용 정적 데이터/벡터 인덱스
│   ├── alias_map.json                              # 카드 ID ↔ 카드명/별칭 매핑 정보
│   └── card_desc_refined_json                      # 정제된 카드 상품 설명서 JSON 모음
│       └── *.json                                  # 우리카드 상품 설명서 정제본
├── docker-compose.yml                              # MCP/툴 서버 실행용 Docker Compose 설정
├── hf_cache                                        # HuggingFace 임베딩/모델 로컬 캐시 디렉토리 (Docker 마운트 용)
├── main.py                                         # FastAPI 앱 엔트리포인트 (툴 서버 실행)
├── mcp_module                                      # MCP 서버 및 툴 로직 구현
│   ├── RAG                                         # 각 도메인별 MCP 모듈
│   │   ├── card_desc_rag_based.py                  # 카드 혜택 설명·비교 RAG 파이프라인
│   │   ├── card_recommend_based.py                 # GraphRAG 기반 혜택 추천 카드 파이프라인
│   │   ├── faq_rag_based.py                        # FAQ 의미 기반 검색(RAG) 파이프라인
│   │   ├── mydata_based.py                         # 개인 소비(MyData) 기반 카드 추천 파이프라인
│   │   ├── retrieval_qdrant.py                     # Qdrant 벡터 검색 공통 래퍼/유틸
│   │   └── term_rag_based.py                       # 금융 용어 RAG/DB 검색 파이프라인
│   └── server.py                                   # MCP 서버 엔트리/툴 등록 및 실행 로직
├── requirements.in                                 # 의존성 원본 목록 (pip-compile 입력용)
├── requirements.txt                                # 고정 버전 의존성 목록 (배포/빌드용)
└── schemas                                         # Pydantic 스키마 정의
    ├── card_description.py                         # 카드 혜택 설명/비교 응답 스키마
    ├── card_recommendation.py                      # 카드 추천 결과(카드 리스트 등) 스키마
    ├── qna.py                                      # FAQ/금융 용어 QnA 응답 스키마
    └── tabular_recommand.py                        # 테이블/탭형 추천 결과 스키마 (군집/소비분석용)
```

## ⚙️ 환경 변수 & 서버 실행
- 환경 변수 (`.env`)
    ```bash
    OLLAMA_BASE_URL=http://127.0.0.1:11434  # OLLAMA 서버 주소
    OLLAMA_MODEL_NAME=qwen3:1.7b            # OLLAMA 모델 이름 (Generation LLM)

    # 문장 임베딩 모델 이름 (HuggingFace)
    EMBEDDING_MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # PostgreSQL 설정
    POSTGRES_HOST=host.docker.internal
    POSTGRES_PORT=5432
    POSTGRES_DB=card_qna_db
    POSTGRES_USER=your_pg_username
    POSTGRES_PASSWORD=your_pg_password

    # Qdrant 서버 설정
    QDRANT_HOST=host.docker.internal
    QDRANT_PORT=6333
    
    BACKEND_SERVER_URL=http://127.0.0.1:8001 # 백엔드 서버 주소
    ```

- 서버 실행 
    ```bash
    # Docker 이미지 빌드 및 컨테이너 실행
    docker compose --env-file .env  up --build -d 
    ```
    - 브라우저에서 `http://127.0.0.1:8011/docs` 로 OpenAPI 문서를 확인할 수 있습니다.
