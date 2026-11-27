"""Check that Postgres is reachable and pgvector is enabled."""

import psycopg
from selfrag.core.config import settings


def main():
    with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            print("SELECT 1 ->", cur.fetchone())

            cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
            print("pgvector ->", cur.fetchone())

if __name__ == '__main__':
    main()