import os

from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Load .env from the anpr-backend folder (parent of this file's
# app/ folder), not just the cwd. load_dotenv() on its own only
# looks in the cwd and its parents, so starting uvicorn from the
# repo root used to miss anpr-backend/.env entirely and raise
# "DATABASE_URL is missing". Real environment variables still
# take precedence (override=False).
_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env", override=False)

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from .env file")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

class Base(DeclarativeBase):
    pass
