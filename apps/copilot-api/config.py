import os

class Settings:
    PROJECT_NAME: str = "Copilot API"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "changeme")

settings = Settings()
