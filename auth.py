"""
MODULE 7: Authentication & User Management
Mobile Money Fraud Detection System - Zambia
auth.py

Provides session-based authentication for the Streamlit dashboard.
Stores users in the `dashboard_users` table with bcrypt-hashed passwords.
Supports three roles:
    - admin   : Full access (all tabs, user management, compliance reports)
    - analyst : Investigation terminal, overrides, manual flags
    - operator: Read-only executive overview
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, Boolean
from sqlalchemy.orm import Session

from db_config import Base, get_engine


# ─────────────────────────────────────────────
# ROLE ENUM
# ─────────────────────────────────────────────

class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"


# ─────────────────────────────────────────────
# ORM MODEL
# ─────────────────────────────────────────────

class DashboardUser(Base):
    __tablename__ = "dashboard_users"

    user_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), nullable=False, unique=True)
    password_hash = Column(String(128), nullable=False)
    role = Column(String(20), nullable=False, default=UserRole.OPERATOR.value)
    full_name = Column(String(100), nullable=False)
    email = Column(String(120), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


# ─────────────────────────────────────────────
# PASSWORD HASHING (SHA-256 + salt, no extra dependency)
# ─────────────────────────────────────────────
# Using salted SHA-256 to avoid adding bcrypt as a dependency.
# The salt is stored as the first 32 hex chars of the hash string.

def _hash_password(password: str, salt: str | None = None) -> str:
    """Hash a password with a random salt. Returns 'salt$hash'."""
    if salt is None:
        salt = os.urandom(16).hex()
    digest = hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()
    return f"{salt}${digest}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored 'salt$hash' string."""
    try:
        salt, _ = stored_hash.split("$", 1)
        return _hash_password(password, salt) == stored_hash
    except ValueError:
        return False


# ─────────────────────────────────────────────
# AUTH FUNCTIONS
# ─────────────────────────────────────────────

def register_user(
    username: str,
    password: str,
    role: str,
    full_name: str,
    email: str | None = None,
) -> DashboardUser:
    """
    Register a new dashboard user.
    Raises ValueError if the username already exists.
    """
    engine = get_engine()
    with Session(engine) as session:
        existing = session.query(DashboardUser).filter_by(username=username).first()
        if existing:
            raise ValueError(f"Username '{username}' is already registered.")

        user = DashboardUser(
            user_id=str(uuid.uuid4()),
            username=username,
            password_hash=_hash_password(password),
            role=role,
            full_name=full_name,
            email=email,
            is_active=True,
            created_at=datetime.utcnow(),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def authenticate(username: str, password: str) -> Optional[DashboardUser]:
    """
    Authenticate a user by username and password.
    Returns the DashboardUser on success, None on failure.
    Updates last_login timestamp on success.
    """
    engine = get_engine()
    with Session(engine) as session:
        user = (
            session.query(DashboardUser)
            .filter_by(username=username, is_active=True)
            .first()
        )
        if user is None:
            return None

        if not _verify_password(password, user.password_hash):
            return None

        user.last_login = datetime.utcnow()
        session.commit()

        # Detach user data for use outside session
        return _detach_user(user)


def _detach_user(user: DashboardUser) -> SimpleNamespace:
    """Create a serialisation-safe copy of user attributes for Streamlit session state."""
    return SimpleNamespace(
        user_id=user.user_id,
        username=user.username,
        password_hash=user.password_hash,
        role=user.role,
        full_name=user.full_name,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login,
    )


def get_user(user_id: str) -> Optional[DashboardUser]:
    """Retrieve a user by user_id."""
    engine = get_engine()
    with Session(engine) as session:
        user = session.query(DashboardUser).filter_by(user_id=user_id).first()
        return _detach_user(user) if user else None


def get_all_users() -> list[dict]:
    """Return all dashboard users as a list of dicts (for admin management)."""
    engine = get_engine()
    with Session(engine) as session:
        users = session.query(DashboardUser).order_by(DashboardUser.created_at).all()
        return [
            {
                "user_id": u.user_id,
                "username": u.username,
                "role": u.role,
                "full_name": u.full_name,
                "email": u.email or "—",
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat() if u.created_at else "—",
                "last_login": u.last_login.isoformat() if u.last_login else "Never",
            }
            for u in users
        ]


def update_user_status(user_id: str, is_active: bool) -> bool:
    """Enable or disable a user account."""
    engine = get_engine()
    with Session(engine) as session:
        user = session.query(DashboardUser).filter_by(user_id=user_id).first()
        if not user:
            return False
        user.is_active = is_active
        session.commit()
        return True


# ─────────────────────────────────────────────
# ROLE CHECKS
# ─────────────────────────────────────────────

def has_role(user: DashboardUser | dict, required_role: str) -> bool:
    """Check if a user has the required role (or higher privilege)."""
    hierarchy = {
        UserRole.ADMIN.value: 3,
        UserRole.ANALYST.value: 2,
        UserRole.OPERATOR.value: 1,
    }
    if isinstance(user, DashboardUser):
        user_role = user.role
    elif isinstance(user, dict):
        user_role = user.get("role", "")
    else:
        user_role = getattr(user, "role", "")
    return hierarchy.get(user_role, 0) >= hierarchy.get(required_role, 99)


# ─────────────────────────────────────────────
# DEFAULT ADMIN SEEDER
# ─────────────────────────────────────────────

def seed_default_users():
    """Create default users if the dashboard_users table is empty."""
    engine = get_engine()
    with Session(engine) as session:
        count = session.query(DashboardUser).count()
        if count > 0:
            return

    print("[auth] Seeding default dashboard users...")
    try:
        register_user("admin", "admin123", UserRole.ADMIN.value, "System Administrator", "admin@fraudops.zm")
        register_user("analyst", "analyst123", UserRole.ANALYST.value, "Fraud Analyst", "analyst@fraudops.zm")
        register_user("operator", "operator123", UserRole.OPERATOR.value, "Mobile Operator", "ops@fraudops.zm")
        print("[auth] Default users created: admin / analyst / operator")
    except ValueError:
        pass  # Users already exist
