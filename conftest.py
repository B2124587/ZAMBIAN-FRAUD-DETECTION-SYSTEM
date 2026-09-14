"""
Shared test fixtures. Sets dummy env vars before anything imports
db_config/app, since those modules require env vars to be present at
import time (they fail fast rather than silently falling back to
hardcoded credentials). No real database connection is opened by these
tests — SQLAlchemy's create_engine() is lazy.
"""

import os

os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_HOST", "127.0.0.1")
os.environ.setdefault("DB_PORT", "3306")
os.environ.setdefault("DB_NAME", "test_db")
os.environ.setdefault("MSISDN_HASH_SECRET", "ci-test-secret-do-not-use-in-prod")
os.environ.setdefault("FRAUD_API_KEYS", "ci-test-key-123")
