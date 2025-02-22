import os

# LLM API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# PGVector
PG_DB_USER = os.getenv("PG_DB_USER")
PG_DB_TOKEN = os.getenv("PG_DB_TOKEN")
PG_DB_HOST = os.getenv("PG_DB_HOST")
PG_DB_NAME = os.getenv("PG_DB_NAME")

PGVECTOR_CONNECTION_STR = (
    f"postgresql+psycopg://{PG_DB_USER}:{PG_DB_TOKEN}@{PG_DB_HOST}:5432/{PG_DB_NAME}"
)
