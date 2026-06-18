import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

UDAHUB_DB_PATH   = os.getenv("UDAHUB_DB_PATH",  "data/core/udahub.db")
CULTPASS_DB_PATH = os.getenv("CULTPASS_DB_PATH", "data/external/cultpass.db")
CHROMA_PATH      = os.getenv("CHROMA_PATH",      "data/core/chroma")
MCP_SERVER_URL   = os.getenv("MCP_SERVER_URL",   "http://localhost:8000")

udahub_engine = create_engine(
    f"sqlite:///{UDAHUB_DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)

cultpass_engine = create_engine(
    f"sqlite:///{CULTPASS_DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)

UdaHubSessionLocal   = sessionmaker(bind=udahub_engine,   autocommit=False, autoflush=False)
CultPassSessionLocal = sessionmaker(bind=cultpass_engine, autocommit=False, autoflush=False)

@contextmanager
def get_udahub_db() -> Session:
    db = UdaHubSessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_cultpass_db() -> Session:
    db = CultPassSessionLocal()
    try:
        yield db
    finally:
        db.close()
