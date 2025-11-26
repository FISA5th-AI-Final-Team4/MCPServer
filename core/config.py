from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.cuda import is_available as cuda_is_available

class Settings(BaseSettings):
    # 시스템 환경변수 적용
    WEATHER_API_BASE_URL: str
    WEATHER_API_KEY: str
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL_NAME: str

    
    # RAG 전역 설정
    VECTOR_DB_PATH: Path = Path(__file__).parent.parent / "data" / "VectorDB"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # PostgreSQL 설정 (환경변수로 관리)
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    # Qdrant 서버 설정 (환경변수로 관리)
    QDRANT_HOST: str
    QDRANT_PORT: int

    # GPU 사용 가능 여부
    DEVICE: str = "cuda" if cuda_is_available() else "cpu"
    
    # 백엔드 서버 주소
    BACKEND_SERVER_URL: str

    @property
    def POSTGRES_PASSWORD(self) -> str:
        """POSTGRES_PASSWORD를 PostgreSQL 비밀번호로 사용"""
        return self.POSTGRESM_PASSWORD
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # .env 환경변수 파일 로드
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True
    )

settings = Settings()