from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- API設定 ---
    api_port: int = Field(default=8000, description="APIサーバーのポート番号")

    # --- Ollama設定 ---
    llm_model: str = Field(default="qwen2.5-coder:7b", description="LLMモデル名")
    embeddings_model: str = Field(default="bge-m3", description="読み込みモデル名")
    ollama_port: str = Field(
        default="http://localhost:11434", description="Ollamaサーバーのポート番号"
    )

    # --- Qdrant設定 ---
    qdrant_host: str = Field(
        default="localhost", description="QdrantのIPアドレスまたはホスト名"
    )
    qdrant_port: int = Field(default=6333, description="Qdrantのポート番号")
    qdrant_collection_name: str = Field(
        default="code_knowledge_base", description="Qdrantのコレクション名"
    )

    # --- プロジェクト設定 ---
    repository_id: str = Field(
        default="default-repo", description="インジェスト対象のリポジトリID"
    )

    # --- CORS設定 ---
    cors_origins: str = Field(
        default="http://localhost:5173", description="許可されるオリジン"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
