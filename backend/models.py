from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    username: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    daily_records: Mapped[list["DailyRecordDB"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    state_estimates: Mapped[list["StateEstimateDB"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class DailyRecordDB(Base):
    __tablename__ = "daily_records"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "date",
            name="uq_daily_record_user_date",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"),
        index=True,
    )

    date: Mapped[date] = mapped_column(
        Date,
        index=True,
    )

    sleep_duration_hours: Mapped[float] = mapped_column(
        Float,
    )

    sleep_quality: Mapped[int] = mapped_column(
        Integer,
    )

    exercise_duration_minutes: Mapped[float] = mapped_column(
        Float,
    )

    exercise_intensity: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    exercise_type: Mapped[str] = mapped_column(
        String(50),
    )

    stress: Mapped[int] = mapped_column(
        Integer,
    )

    fatigue_observed: Mapped[int] = mapped_column(
        Integer,
    )

    recovery_observed: Mapped[int] = mapped_column(
        Integer,
    )

    user: Mapped["User"] = relationship(
        back_populates="daily_records",
    )


class StateEstimateDB(Base):
    __tablename__ = "state_estimates"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "date",
            name="uq_state_estimate_user_date",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"),
        index=True,
    )

    date: Mapped[date] = mapped_column(
        Date,
        index=True,
    )

    fatigue: Mapped[float] = mapped_column(
        Float,
    )

    recovery: Mapped[float] = mapped_column(
        Float,
    )

    load: Mapped[float] = mapped_column(
        Float,
    )

    fatigue_variance: Mapped[float] = mapped_column(
        Float,
    )

    recovery_variance: Mapped[float] = mapped_column(
        Float,
    )

    load_variance: Mapped[float] = mapped_column(
        Float,
    )

    sleep_deficit: Mapped[float] = mapped_column(
        Float,
    )

    exercise_load_raw: Mapped[float] = mapped_column(
        Float,
    )

    exercise_load_model: Mapped[float] = mapped_column(
        Float,
    )

    stress: Mapped[float] = mapped_column(
        Float,
    )

    user: Mapped["User"] = relationship(
        back_populates="state_estimates",
    )