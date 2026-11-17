
import os
from pathlib import Path

import psycopg


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


def load_environment(env_file: Path = ENV_FILE) -> None:
    """Load environment variables from the project's .env file."""

    if not env_file.is_file():
        raise FileNotFoundError(
            f"Environment file not found: {env_file}"
        )

    with env_file.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip()

            if len(value) >= 2 and value[0] == value[-1]:
                if value[0] in ("'", '"'):
                    value = value[1:-1]

            if key:
                os.environ.setdefault(key, value)


def get_connection() -> psycopg.Connection:
    """Create a PostgreSQL connection using project environment settings."""

    load_environment()

    required = (
        "POSTGRES_HOST",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    )

    missing = [
        key for key in required
        if key not in os.environ
    ]

    if missing:
        raise RuntimeError(
            f"Missing database environment variables: {', '.join(missing)}"
        )

    try:
        port = int(os.environ.get("POSTGRES_PORT", "5432"))
    except ValueError as exc:
        raise ValueError("POSTGRES_PORT must be an integer") from exc

    return psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=port,
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
