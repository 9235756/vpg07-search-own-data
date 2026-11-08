"""Конфиг: понятные ошибки о пустых и неверных переменных."""

import pytest

from hay_v2_bot import config

FULL_ENV = {
    "TELEGRAM_BOT_TOKEN": "token",
    "OPENAI_API_KEY": "key",
    "OPENAI_BASE_URL": "https://api.proxyapi.ru/openai/v1/",
    "PINECONE_API_KEY": "pc-key",
    "PINECONE_INDEX": "hay-v2",
}
OPTIONAL = (
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "PINECONE_NAMESPACE_DOCS",
    "PINECONE_NAMESPACE_CHAT",
    "MAX_FILE_SIZE_MB",
    "DOCS_TOP_K",
    "MEMORY_TOP_K",
    "DOCS_MIN_SCORE",
    "SUMMARY_CHUNKS",
)


@pytest.fixture
def env(monkeypatch, tmp_path):
    # .env.local разработчика не должен влиять на тесты.
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env.local")
    for name in (*config.REQUIRED_VARS, *OPTIONAL):
        monkeypatch.delenv(name, raising=False)
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def test_full_config_with_defaults(env):
    cfg = config.load_config()

    assert cfg.openai_base_url == "https://api.proxyapi.ru/openai/v1"
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.embedding_dimension == 1536
    assert (cfg.namespace_docs, cfg.namespace_chat) == ("docs", "chat")
    assert cfg.max_file_size_mb == 20
    assert cfg.max_file_size_bytes == 20 * 1024 * 1024


def test_missing_vars_are_listed(env):
    env.delenv("OPENAI_API_KEY")
    env.setenv("PINECONE_INDEX", "   ")

    with pytest.raises(config.ConfigError) as exc:
        config.load_config()

    message = str(exc.value)
    assert "OPENAI_API_KEY" in message and "PINECONE_INDEX" in message
    assert "TELEGRAM_BOT_TOKEN" not in message
    assert ".env.local" in message


def test_base_url_is_required(env):
    env.setenv("OPENAI_BASE_URL", "")

    with pytest.raises(config.ConfigError, match="OPENAI_BASE_URL"):
        config.load_config()


def test_base_url_must_be_http(env):
    env.setenv("OPENAI_BASE_URL", "api.proxyapi.ru/openai/v1")

    with pytest.raises(config.ConfigError, match="http"):
        config.load_config()


def test_namespaces_must_differ(env):
    env.setenv("PINECONE_NAMESPACE_DOCS", "same")
    env.setenv("PINECONE_NAMESPACE_CHAT", "same")

    with pytest.raises(config.ConfigError, match="различаться"):
        config.load_config()


@pytest.mark.parametrize("value", ["двадцать", "0", "-5"])
def test_bad_file_size(env, value):
    env.setenv("MAX_FILE_SIZE_MB", value)

    with pytest.raises(config.ConfigError, match="MAX_FILE_SIZE_MB"):
        config.load_config()


def test_unknown_embedding_model(env):
    env.setenv("OPENAI_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    with pytest.raises(config.ConfigError, match="модель эмбеддингов"):
        config.load_config()
