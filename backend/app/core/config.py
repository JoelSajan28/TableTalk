from pydantic import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str
    MONGO_DB: str = "tabularag"
    FILE_BASE_URL: str = ""  # For doclinks
    class Config:
        env_file = ".env"

settings = Settings()
