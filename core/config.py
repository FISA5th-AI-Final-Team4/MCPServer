from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 시스템 환경변수 적용
    WEATHER_API_BASE_URL: str
    WEATHER_API_KEY: str

    # .env 환경변수 파일 로드
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True
    )

settings = Settings()