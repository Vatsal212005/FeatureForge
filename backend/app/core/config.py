from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "FeatureForge"
    environment: str = "development"

    database_url: str = "sqlite:///./featureforge.db"

    redis_host: str = "localhost"
    redis_port: int = 6379

    class Config:
        env_file = ".env"


settings = Settings()
