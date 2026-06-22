from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Table
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime

# Define the database file
DATABASE_URL = "sqlite:///./rag_app.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Association table for User and Role (many-to-many)
user_roles = Table('user_roles', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('role_id', Integer, ForeignKey('roles.id'))
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    roles = relationship("Role", secondary=user_roles, back_populates="users")

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    users = relationship("User", secondary=user_roles, back_populates="roles")

# --- Chat History and Logging Tables ---

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, index=True) # e.g., UUID
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

class AppLog(Base):
    __tablename__ = "app_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String)
    message = Column(String)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=True)