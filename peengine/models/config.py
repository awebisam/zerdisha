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
    """LLM configuration for Azure OpenAI."""
    # Azure OpenAI Primary
    azure_openai_key: str
    azure_openai_endpoint: str
    azure_openai_deployment_name: str
    azure_openai_model_name: str
    azure_openai_api_version: str = "2024-12-01-preview"
    
    # Azure AI Foundry Fallback
    azure_ai_foundry_key: Optional[str] = None
    azure_ai_foundry_endpoint: Optional[str] = None
    
    # Model Selection
    primary_model: str = "gpt-4.1-mini"
    fallback_model: str = "gpt-4o-mini"  
    pattern_model: str = "gpt-3.5-turbo"
    
    # Generation Parameters
    temperature: float = 0.7
    max_tokens: int = 2000


class PersonaConfig(BaseModel):
    """Persona configuration."""
    path: str = "docs/external/persona.md"
    update_frequency: int = 5  # Update persona every N interactions
    adaptation_enabled: bool = True


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # Azure OpenAI Configuration (Primary)
    azure_openai_key: str
    azure_openai_endpoint: str
    azure_openai_deployment_name: str
    azure_openai_model_name: str
    azure_openai_api_version: str = "2024-12-01-preview"
    
    # Azure AI Foundry Configuration (Fallback)
    azure_ai_foundry_api_key: Optional[str] = None
    azure_ai_foundry_endpoint: Optional[str] = None
    
    # Model Selection
    primary_model: str = "gpt-4.1-mini"
    fallback_model: str = "gpt-4o-mini"
    pattern_model: str = "gpt-3.5-turbo"
    
    # Database Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    neo4j_database: str = "neo4j"
    
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "socratic_lab"
    
    # Application Configuration
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
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database
        )
    
    @property
    def llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return LLMConfig(
            azure_openai_key=self.azure_openai_key,
            azure_openai_endpoint=self.azure_openai_endpoint,
            azure_openai_deployment_name=self.azure_openai_deployment_name,
            azure_openai_model_name=self.azure_openai_model_name,
            azure_openai_api_version=self.azure_openai_api_version,
            azure_ai_foundry_key=self.azure_ai_foundry_api_key,
            azure_ai_foundry_endpoint=self.azure_ai_foundry_endpoint,
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
            pattern_model=self.pattern_model
        )
    
    @property
    def persona_config(self) -> PersonaConfig:
        """Get persona configuration."""
        return PersonaConfig(path=self.persona_path)