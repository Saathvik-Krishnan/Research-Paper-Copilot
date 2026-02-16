from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"


    class Config:
        env_file = ".env"

settings = Settings()
