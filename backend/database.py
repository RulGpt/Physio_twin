from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config import DATABASE_DIR


DATABASE_URL = (
    f"sqlite:///{DATABASE_DIR / 'physio_twin.db'}"
)


class Base(DeclarativeBase):
    pass


engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
    },
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def _migrate_users_table() -> None:
    """
    Add profile fields to the existing users table without
    deleting or recreating existing users.

    Existing users receive deterministic legacy usernames.
    """

    inspector = inspect(engine)

    if "users" not in inspector.get_table_names():
        return

    columns = {
        column["name"]
        for column in inspector.get_columns("users")
    }

    with engine.begin() as connection:

        # -------------------------------------------------------------
        # Add missing columns
        # -------------------------------------------------------------

        if "name" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE users "
                    "ADD COLUMN name VARCHAR(100)"
                )
            )

        if "username" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE users "
                    "ADD COLUMN username VARCHAR(50)"
                )
            )

        # -------------------------------------------------------------
        # Give existing users deterministic legacy profiles.
        #
        # This does NOT change their UUID or any associated data.
        # -------------------------------------------------------------

        rows = connection.execute(
            text(
                "SELECT id FROM users "
                "WHERE username IS NULL OR username = ''"
            )
        ).fetchall()

        for row in rows:
            user_id = str(row[0])

            legacy_username = (
                f"legacy_{user_id.replace('-', '')[:8]}"
            )

            legacy_name = (
                f"Legacy User {user_id.replace('-', '')[:8]}"
            )

            connection.execute(
                text(
                    "UPDATE users "
                    "SET username = :username, "
                    "name = :name "
                    "WHERE id = :user_id"
                ),
                {
                    "username": legacy_username,
                    "name": legacy_name,
                    "user_id": user_id,
                },
            )

        # -------------------------------------------------------------
        # Unique username index
        # -------------------------------------------------------------

        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "ix_users_username_unique "
                "ON users(username)"
            )
        )


def init_db() -> None:
    from backend.models import (
        User,
        DailyRecordDB,
        StateEstimateDB,
    )

    Base.metadata.create_all(
        bind=engine
    )

    _migrate_users_table()


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()