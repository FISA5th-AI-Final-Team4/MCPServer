from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 시스템 환경변수 적용
    WEATHER_API_BASE_URL: str
    WEATHER_API_KEY: str
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL_NAME: str
    
    # RAG 전역 설정
    VECTOR_DB_PATH: Path = Path(__file__).parent.parent / "data" / "VectorDB"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


    # .env 환경변수 파일 로드
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True
    )

settings = Settings()