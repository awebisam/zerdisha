"""Configuration models."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Neo4j database configuration."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str
    database: str = "neo4j"


class LLMConfig(BaseModel):
    """LLM configuration."""
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_type: str = "openai"  # "openai" or "azure"
    api_version: str = "2024-02-15-preview"  # Azure API version
    deployment_name: Optional[str] = None  # Azure deployment name


class PersonaConfig(BaseModel):
    """Persona configuration."""
    path: str = "docs/external/persona.md"
    update_frequency: int = 5  # Update persona every N interactions
    adaptation_enabled: bool = True


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # Database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j" 
    neo4j_password: str
    neo4j_database: str = "neo4j"
    
    # OpenAI/Azure OpenAI
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4"
    openai_api_type: str = "openai"  # "openai" or "azure"
    openai_api_version: str = "2024-02-15-preview"
    openai_deployment_name: Optional[str] = None  # For Azure
    
    # Application
    log_level: str = "INFO"
    persona_path: str = "docs/external/persona.md"
    knowledge_graphs_path: str = "knowledge_graphs/"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @property
    def database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return DatabaseConfig(
            uri=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            database=self.neo4j_database
        )
    
    @property
    def llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return LLMConfig(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            model=self.openai_model,
            api_type=self.openai_api_type,
            api_version=self.openai_api_version,
            deployment_name=self.openai_deployment_name
        )
    
    @property
    def persona_config(self) -> PersonaConfig:
        """Get persona configuration."""
        return PersonaConfig(path=self.persona_path)