from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for running HITL experiences via the ursa CLI."""

    model_config = SettingsConfigDict(
        env_prefix="URSA_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        validate_default=True,
        extra="ignore",
    )

    workspace: Path = Field(
        default=Path("ursa_workspace"),
        description="Directory used to persist checkpoints, logs, and caches.",
    )
    llm_model_name: str = Field(
        default="openai:gpt-5-mini",
        description="Model provider and name for the LLM.",
        validation_alias=AliasChoices("LLM_MODEL_NAME", "LLM_NAME"),
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description="Override base URL for the LLM provider.",
        validation_alias="LLM_BASE_URL",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key used to authenticate with the LLM provider.",
        validation_alias="LLM_API_KEY",
    )
    max_completion_tokens: int = Field(
        default=50_000,
        description="Maximum number of tokens that can be generated per completion.",
        validation_alias="MAX_COMPLETION_TOKENS",
    )
    emb_model_name: str = Field(
        default="openai:text-embedding-3-small",
        description="Model provider and embedding model name.",
        validation_alias=AliasChoices("EMB_MODEL_NAME", "EMB_NAME"),
    )
    emb_base_url: Optional[str] = Field(
        default=None,
        description="Override base URL for the embedding provider.",
        validation_alias="EMB_BASE_URL",
    )
    emb_api_key: Optional[str] = Field(
        default=None,
        description="API key used to authenticate with the embedding provider.",
        validation_alias="EMB_API_KEY",
    )
    share_key: bool = Field(
        default=False,
        description="Indicates whether LLM and embedding providers share a single API key.",
        validation_alias="SHARE_KEY",
    )
    thread_id: str = Field(
        default="ursa_cli",
        description="Base identifier used for agent memory checkpoints.",
        validation_alias="THREAD_ID",
    )
    arxiv_summarize: bool = Field(
        default=True,
        description="Allow the ArxivAgent to summarize results.",
        validation_alias="ARXIV_SUMMARIZE",
    )
    arxiv_process_images: bool = Field(
        default=False,
        description="Allow the ArxivAgent to download and process images.",
        validation_alias="ARXIV_PROCESS_IMAGES",
    )
    arxiv_max_results: int = Field(
        default=10,
        description="Maximum number of papers retrieved per ArXiv search.",
        validation_alias="ARXIV_MAX_RESULTS",
    )
    arxiv_database_path: Optional[Path] = Field(
        default=None,
        description="Directory used to cache downloaded ArXiv documents.",
        validation_alias="ARXIV_DATABASE_PATH",
    )
    arxiv_summaries_path: Optional[Path] = Field(
        default=None,
        description="Directory used to store generated ArXiv summaries.",
        validation_alias="ARXIV_SUMMARIES_PATH",
    )
    arxiv_vectorstore_path: Optional[Path] = Field(
        default=None,
        description="Directory used to persist ArXiv vector stores.",
        validation_alias="ARXIV_VECTORSTORE_PATH",
    )
    arxiv_download_papers: bool = Field(
        default=True,
        description="Allow ArxivAgent to download full paper content.",
        validation_alias="ARXIV_DOWNLOAD_PAPERS",
    )
    ssl_verify: bool = Field(
        default=True,
        description="Verify SSL certificates when connecting to remote providers.",
        validation_alias="SSL_VERIFY",
    )

